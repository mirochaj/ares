"""

VolumeLocal.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Dec 31 11:16:59 2012

Description:

"""

import copy
import numpy as np
from ..util import ParameterFile
from ..physics.SecondaryElectrons import SecondaryElectrons
from ..physics.Constants import erg_per_ev, E_LyA, ev_per_hz

class LocalVolume:
    def __init__(self, grid, sources, **kwargs):
        self.pf = ParameterFile(**kwargs)
        self.grid = grid
        self.srcs = sources
        self.esec = SecondaryElectrons(method=self.pf['secondary_ionization'])
                
        if self.srcs is not None:
            self._initialize()
    
    def _initialize(self):
        self.Ns = len(self.srcs)
    
        self.E_th = {}
        for absorber in self.grid.absorbers:
            self.E_th[absorber] = self.grid.ioniz_thresholds[absorber]
    
        # Array of cross-sections to match grid size 
        self.sigma = []
        for src in self.srcs:
            if src.continuous:
                self.sigma.append(None)
                continue
    
            self.sigma.append((np.ones([self.grid.dims, src.Nfreq]) \
                   * src.sigma).T)
    
        # Calculate correction to normalization factor if plane_parallel    
        if self.pf['optically_thin']:
            if self.pf['plane_parallel']:
                self.pp_corr = 4. * np.pi * self.grid.r_mid**2
            else:
                self.pp_corr = 1.0
        else:
            if self.pf['photon_conserving']:
                self.pp_corr = self.grid.Vsh / self.grid.dr
            else:
                self.A_npc = source.Lbol / 4. / np.pi / self.grid.r_mid**2
                self.pp_corr = 4. * np.pi * self.grid.r_mid**2

    @property
    def rates_no_RT(self):
        if not hasattr(self, '_rates_no_RT'):
            self._rates_no_RT = \
                {'k_ion': np.zeros((self.Ns, self.grid.dims, 
                    self.grid.N_absorbers)),
                 'k_heat': np.zeros((self.Ns, self.grid.dims, 
                    self.grid.N_absorbers)),
                 'k_ion2': np.zeros((self.Ns, self.grid.dims, 
                    self.grid.N_absorbers, self.grid.N_absorbers)),
                }
        
        return self._rates_no_RT

    def update_rate_coefficients(self, data, t, rfield):
        """
        Get rate coefficients for ionization and heating. Sort into dictionary.
        
        Parameters
        ----------
        rfield : RadialField instance
            Contains attributes representing column densities and such.
        """
    
        # Return zeros for everything if RT is off
        if not self.pf['radiative_transfer']:
            self.kwargs = self.rates_no_RT.copy()
            return self.kwargs
        else:
            self.kwargs = {}

        # Parse column densities, set attributes
        for attribute in ['logN_by_cell', 'logNdN', 'n', 'N', 'Nc']:
            val = getattr(rfield, attribute)
            setattr(self, attribute, val)

        # Make data globally accessible
        self.data = data.copy()

        # Compute source dependent rate coefficients
        k_ion_src, k_ion2_src, k_heat_src, Ja_src = \
            self._get_coefficients(data, t)
                    
        # Unpack source-specific rates if necessary            
        if len(self.srcs) > 1:
            for i, src in enumerate(self.srcs):
                self.kwargs.update({'k_ion_{}'.format(i): k_ion_src[i], 
                    'k_ion2_{}'.format(i): k_ion2_src[i],
                    'k_heat_{}'.format(i): k_heat_src[i]})
                    
                if False:
                    self.kwargs.update({'Ja_{}'.format(i): Ja_src[i]})
        
                    #if self.pf['secondary_lya']:
                    #    self.kwargs.update({'Ja_X_{}'.format(i): Ja_src[i]})
        
        # Sum over sources
        k_ion = np.sum(k_ion_src, axis=0)
        k_ion2 = np.sum(k_ion2_src, axis=0)
        k_heat = np.sum(k_heat_src, axis=0)
            
        # Compute Lyman-Alpha emission
        if False:
            Ja = np.sum(Ja_src, axis=0)
        
        #if self.pf['secondary_lya']:
        #    Ja_X = np.sum(Ja_src, axis=0)
        
        # Each is grid x absorbers, or grid x [absorbers, absorbers] for gamma
        self.kwargs.update({'k_ion': k_ion, 'k_heat': k_heat, 'k_ion2': k_ion2})
        
        # Ja just has len(grid) 
        if False:
            self.kwargs.update({'Ja': Ja})
        
        return self.kwargs
               
    def _get_coefficients(self, data, t):
        """
        Compute rate coefficients for ionization and heating.
        
        Parameters
        ----------
        data : dict
            Data for current snapshot.
        t : int, float  
            Current time (needed to make sure sources are on).
        
        """
        
        self.k_ion = np.zeros((self.Ns, self.grid.dims, self.grid.N_absorbers))
        self.k_heat = np.zeros((self.Ns, self.grid.dims, self.grid.N_absorbers))
        self.k_ion2 = np.zeros((self.Ns, self.grid.dims, self.grid.N_absorbers, 
            self.grid.N_absorbers))
        
        if True:
            self.Ja = [None] * self.Ns
        else:
            self.Ja = np.array(self.Ns * [np.zeros(self.grid.dims)])

        # Loop over sources
        for h, src in enumerate(self.srcs):      

            if not src.SourceOn(t):
                continue
                
            self.h = h
            self.src = src
                
            # If we're operating under the optically thin assumption, 
            # return pre-computed source-dependent values.    
            if self.pf['optically_thin']:
                self.tau_tot = np.zeros(self.grid.dims) # by definition
                self.k_ion[h] = src.k_ion_bar * self.pp_corr
                self.k_heat[h] = src.k_heat_bar * self.pp_corr
                self.k_ion2[h] = src.k_ion2_bar * self.pp_corr
                continue

            # Normalizations
            self.A = {}
            for absorber in self.grid.absorbers:          
                
                if self.pf['photon_conserving']:
                    self.A[absorber] = self.src.Lbol(t) \
                        / self.n[absorber] / self.grid.Vsh
                else:
                    self.A[absorber] = self.A_npc
                    
                # Correct normalizations if radiation field is plane-parallel
                if self.pf['plane_parallel']:
                    self.A[absorber] = self.A[absorber] * self.pp_corr
                                
            """
            For sources with discrete SEDs.
            """
            if self.src.discrete:
            
                # Loop over absorbing species
                for i, absorber in enumerate(self.grid.absorbers):
                                    
                    # Discrete spectrum (multi-freq approach)
                    if self.src.multi_freq:
                        r1, r2, r3 = self.MultiFreqCoefficients(data, absorber, t)
                        self.k_ion[h,:,i], self.k_ion2[h,:,i,:], \
                        self.k_heat[h,:,i] = \
                            self.MultiFreqCoefficients(data, absorber, t)
                    
                    # Discrete spectrum (multi-grp approach)
                    elif self.src.multi_group:
                        pass
                
                continue
                
            """
            For sources with continuous SEDs.
            """
            
            # This could be post-processed, but eventually may be more
            # sophisticated
            if True:
                self.Ja = None
            else:
                self.Ja[h] = src.Spectrum(E_LyA) * ev_per_hz \
                    * src.Lbol(t) / 4. / np.pi / self.grid.r_mid**2 \
                    / E_LyA / erg_per_ev 
            
            # Initialize some arrays/dicts
            self.PhiN = {}
            self.PhiNdN = {}
            self.fheat = 1.0
            self.fion = dict([(absorber, 1.0) for absorber in self.grid.absorbers])
            
            self.PsiN = {}
            self.PsiNdN = {}
            if not self.pf['isothermal'] and self.pf['secondary_ionization'] < 2:
                self.fheat = self.esec.DepositionFraction(data['h_2'], 
                    channel='heat')
                
            self.logx = None            
            if self.pf['secondary_ionization'] > 1:
                
                self.logx = np.log10(data['h_2'])
                
                self.PhiWiggleN = {}
                self.PhiWiggleNdN = {}
                self.PhiHatN = {}
                self.PhiHatNdN = {}
                self.PsiWiggleN = {}
                self.PsiWiggleNdN = {}
                self.PsiHatN = {}
                self.PsiHatNdN = {}
                
                for absorber in self.grid.absorbers:
                    self.PhiWiggleN[absorber] = {}
                    self.PhiWiggleNdN[absorber] = {}
                    self.PsiWiggleN[absorber] = {}
                    self.PsiWiggleNdN[absorber] = {}
                
            else:
                self.fion = {}
                for absorber in self.grid.absorbers:
                    self.fion[absorber] = \
                        self.esec.DepositionFraction(xHII=data['h_2'], 
                            channel=absorber)
                            
            # Loop over absorbing species, compute tabulated quantities
            for i, absorber in enumerate(self.grid.absorbers):
                                           
                self.PhiN[absorber] = \
                    10**self.src.tables["logPhi_{!s}".format(absorber)](self.logN_by_cell,
                    self.logx, t)
                
                if (not self.pf['isothermal']) and (self.pf['secondary_ionization'] < 2):
                    self.PsiN[absorber] = \
                        10**self.src.tables["logPsi_{!s}".format(absorber)](self.logN_by_cell,
                        self.logx, t)
                    
                if self.pf['photon_conserving']:
                    self.PhiNdN[absorber] = \
                        10**self.src.tables["logPhi_{!s}".format(absorber)](self.logNdN[i],
                        self.logx, t)
                    
                    if (not self.pf['isothermal']) and (self.pf['secondary_ionization'] < 2):
                        self.PsiNdN[absorber] = \
                            10**self.src.tables["logPsi_{!s}".format(absorber)](self.logNdN[i],
                            self.logx, t)
            
                if self.pf['secondary_ionization'] > 1:
                    
                    self.PhiHatN[absorber] = \
                        10**self.src.tables["logPhiHat_{!s}".format(absorber)](self.logN_by_cell,
                        self.logx, t)    
                                        
                    if not self.pf['isothermal']:
                        self.PsiHatN[absorber] = \
                            10**self.src.tables["logPsiHat_{!s}".format(absorber)](self.logN_by_cell,
                            self.logx, t)  
                                                
                        if self.pf['photon_conserving']:    
                            self.PhiHatNdN[absorber] = \
                                10**self.src.tables["logPhiHat_{!s}".format(absorber)](self.logNdN[i],
                                self.logx, t)
                            self.PsiHatNdN[absorber] = \
                                10**self.src.tables["logPsiHat_{!s}".format(absorber)](self.logNdN[i],
                                self.logx, t)     
                    
                    for j, donor in enumerate(self.grid.absorbers):
                        
                        suffix = '{0!s}_{1!s}'.format(absorber, donor)
                        
                        self.PhiWiggleN[absorber][donor] = \
                            10**self.src.tables["logPhiWiggle_{!s}".format(suffix)](self.logN_by_cell,
                                self.logx, t)    
                        
                        self.PsiWiggleN[absorber][donor] = \
                            10**self.src.tables["logPsiWiggle_{!s}".format(suffix)](self.logN_by_cell,
                            self.logx, t)
                            
                        if not self.pf['photon_conserving']:
                            continue
                        
                        self.PhiWiggleNdN[absorber][donor] = \
                            10**self.src.tables["logPhiWiggle_{!s}".format(suffix)](self.logNdN[j],
                            self.logx, t)
                        self.PsiWiggleNdN[absorber][donor] = \
                            10**self.src.tables["logPsiWiggle_{!s}".format(suffix)](self.logNdN[j],
                            self.logx, t)

            # Now, go ahead and calculate the rate coefficients
            for k, absorber in enumerate(self.grid.absorbers):
                self.k_ion[h][...,k] = self.PhotoIonizationRate(absorber)
                self.k_heat[h][...,k] = self.PhotoHeatingRate(absorber)

                for j, donor in enumerate(self.grid.absorbers):
                    self.k_ion2[h][...,k,j] = \
                        self.SecondaryIonizationRate(absorber, donor)
                       
            # Compute total optical depth too
            self.tau_tot = 10**self.src.tables["logTau"](self.logN_by_cell)
            
        return self.k_ion, self.k_ion2, self.k_heat, self.Ja
        
    def MultiFreqCoefficients(self, data, absorber, t=None):
        """
        Compute all source-dependent rates.
        
        (For given absorber assuming a multi-frequency SED)
        
        """
        
        k_heat = np.zeros(self.grid.dims)
        k_ion2 = np.zeros_like(self.grid.zeros_grid_x_absorbers)
        
        i = self.grid.absorbers.index(absorber)
        n = self.n[absorber]
        N = self.N[absorber]
               
        # Optical depth up to cells at energy E
        N = np.ones([self.src.Nfreq, self.grid.dims]) * self.N[absorber]
        
        self.tau_r = N * self.sigma[self.h]
        self.tau_tot = np.sum(self.tau_r, axis=1)
        
        Qdot = self.src.Qdot(t=t)
                        
        # Loop over energy groups
        k_ion_E = np.zeros([self.grid.dims, self.src.Nfreq])
        for j, E in enumerate(self.src.E):
            
            if E < self.E_th[absorber]:
                continue    
            
            # Optical depth of cells (at this photon energy)                                                           
            tau_c = self.Nc[absorber] * self.src.sigma[j]
                                                            
            # Photo-ionization by *this* energy group
            k_ion_E[...,j] = \
                self.PhotoIonizationRateMultiFreq(Qdot[j], n,
                self.tau_r[j], tau_c)     
                                          
            # Heating
            if self.grid.isothermal:
                continue

            fheat = self.esec.DepositionFraction(xHII=data['h_2'], 
                E=E, channel='heat')

            # Total energy deposition rate per atom i via photo-electrons 
            # due to ionizations by *this* energy group. 
            ee = k_ion_E[...,j] * (E - self.E_th[absorber]) \
               * erg_per_ev

            k_heat += ee * fheat

            if not self.pf['secondary_ionization']:
                continue
                                        
            # Ionizations of species k by photoelectrons from species i
            # Neglect HeII until somebody figures out how that works
            for k, otherabsorber in enumerate(self.grid.absorbers):
            
                # If these photo-electrons don't have enough 
                # energy to ionize species k, continue    
                if (E - self.E_th[absorber]) < \
                    self.E_th[otherabsorber]:
                    continue    
                
                fion = self.esec.DepositionFraction(xHII=data['h_2'], 
                    E=E, channel=absorber)

                # (This k) = i from paper, and (this i) = j from paper
                k_ion2[...,k] += ee * fion \
                    / (self.E_th[otherabsorber] * erg_per_ev)
                                                                           
        # Total photo-ionization tally
        k_ion = np.sum(k_ion_E, axis=1)
                
        return k_ion, k_ion2, k_heat
    
    def PhotoIonizationRateMultiFreq(self, qdot, n, tau_r_E, tau_c):
        """
        Returns photo-ionization rate coefficient for single frequency over
        the entire grid.
        """     
                                        
        q0 = qdot * np.exp(-tau_r_E)             # number of photons entering cell per sec
        dq = q0 * (1. - np.exp(-tau_c))          # number of photons absorbed in cell per sec
        IonizationRate = dq / n / self.grid.Vsh  # ionizations / sec / atom        
                                  
        if self.pf['plane_parallel']:
            IonizationRate *= self.pp_corr
        
        return IonizationRate
        
    def PhotoIonizationRateMultiGroup(self):
        pass
        
    def PhotoIonizationRate(self, absorber):
        """
        Returns photo-ionization rate coefficient for continuous source.
        """                                     
            
        IonizationRate = self.PhiN[absorber].copy()
        if self.pf['photon_conserving']:
            IonizationRate -= self.PhiNdN[absorber]
        
        return self.A[absorber] * IonizationRate
        
    def PhotoHeatingRate(self, absorber):
        """
        Photo-electric heating rate coefficient due to photo-electrons previously 
        bound to `species.'  If this method is called, it means TabulateIntegrals = 1.
        """

        if self.pf['isothermal']:
            return 0.0

        if self.esec.method < 2:
            HeatingRate = self.PsiN[absorber].copy()
            HeatingRate -= self.E_th[absorber] * erg_per_ev  \
                * self.PhiN[absorber]
            if self.pf['photon_conserving']:
                HeatingRate -= self.PsiNdN[absorber]
                HeatingRate += erg_per_ev \
                    * self.E_th[absorber] \
                    * self.PhiNdN[absorber]
        else:
            HeatingRate = self.PsiHatN[absorber].copy()
            HeatingRate -= self.E_th[absorber] * erg_per_ev  \
                * self.PhiHatN[absorber]
            if self.pf['photon_conserving']:
                HeatingRate -= self.PsiHatNdN[absorber]
                HeatingRate += erg_per_ev \
                    * self.E_th[absorber] \
                    * self.PhiHatNdN[absorber]

        return self.A[absorber] * self.fheat * HeatingRate
            
    def SecondaryIonizationRate(self, absorber, donor):
        """
        Secondary ionization rate which we denote elsewhere as gamma (note little g).
        
            absorber = species being ionized by photo-electron
            donor = species the photo-electron came from
            
        If this routine is called, it means TabulateIntegrals = 1.
        """    
        
        if self.esec.method < 2:
            IonizationRate = self.PsiN[donor].copy()
            IonizationRate -= self.E_th[donor] \
                * erg_per_ev * self.PhiN[donor]
            if self.pf['photon_conserving']:
                IonizationRate -= self.PsiNdN[donor]
                IonizationRate += self.E_th[donor] \
                    * erg_per_ev * self.PhiNdN[donor]
                            
        else:
            IonizationRate = self.PsiWiggleN[absorber][donor] \
                - self.E_th[donor] \
                * erg_per_ev * self.PhiWiggleN[absorber][donor]
            if self.pf['photon_conserving']:
                IonizationRate -= self.PsiWiggleNdN[absorber][donor]
                IonizationRate += self.E_th[donor] \
                    * erg_per_ev * self.PhiWiggleNdN[absorber][donor]            
                        
        # Normalization (by number densities) will be applied in 
        # chemistry solver    
        return self.A[donor] * self.fion[absorber] * IonizationRate \
                / self.E_th[absorber] / erg_per_ev    
        
        
        
