"""

RadiationField.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Dec 31 11:16:59 2012

Description:

"""

import copy
import numpy as np
from .Chemistry import Chemistry
from ..util.Misc import parse_kwargs
from ..physics.SecondaryElectrons import SecondaryElectrons
from ..physics.Constants import erg_per_ev, E_LyA, ev_per_hz

class RadiationField: # maybe RadiationNearSource
    def __init__(self, grid, sources, **kwargs):
        self.pf = parse_kwargs(**kwargs)
        self.grid = grid
        self.srcs = sources
        self.esec = SecondaryElectrons(method=self.pf['secondary_ionization'])
                
        self.chem = Chemistry(grid, rt=self.pf['radiative_transfer'],
            rate_src=self.pf['rate_source'], rtol=self.pf['solver_rtol'], 
            atol=self.pf['solver_atol'], 
            recombination=self.pf['recombination'])        
                
        if self.srcs is not None:
            self._initialize()
    
    def _initialize(self):
        self.Ns = len(self.srcs)
    
        # See if all sources are diffuse
        self.all_diffuse = 1
        for src in self.srcs:
            self.all_diffuse *= int(src.SourcePars['type'] == 'diffuse')
    
        if self.all_diffuse:
            return
    
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
    def finite_c(self):
        if self.pf['infinite_c']:
            return False
        return True      
    
    @property
    def helium(self):
        return 2 in self.grid.Z      
    
    def Evolve(self, data, t, dt, z=None, **kwargs):
        """
        This routine calls our solvers and updates 'data' -> 'newdata'
    
        PhotonPackage guide: 
        pack = [EmissionTime, EmissionTimeInterval, NHI, NHeI, NHeII, E]
    
        """
    
        # Make data globally accessible
        self.data = data.copy()
    
        # Figure out which processors will solve which cells and create newdata dict
        #self.solve_arr, newdata = self.control.DistributeDataAcrossProcessors(data, lb)
    
        # Set up photon packages    
        if self.finite_c and t == 0:
            self.data['photon_packages'] = []
    
        # Compute source dependent rate coefficients
        self.kwargs = {}
        if self.pf['radiative_transfer']:
    
            if self.finite_c:
                raise NotImplementedError('Finite speed-of-light solver not implemented.')
    
            else:    
                Gamma_src, gamma_src, Heat_src, Ja_src = \
                    self.SourceDependentCoefficients(data, t, z, 
                        **kwargs)    
    
            if len(self.srcs) > 1:
                for i, src in enumerate(self.srcs):
                    self.kwargs.update({'Gamma_%i' % i: Gamma_src[i], 
                        'gamma_%i' % i: gamma_src[i],
                        'Heat_%i' % i: Heat_src[i]})
    
                    if not self.pf['approx_lya']:
                        self.kwargs.update({'Ja_%i' % i: Ja_src[i]})
    
            # Sum over sources
            Gamma = np.sum(Gamma_src, axis=0)
            gamma = np.sum(gamma_src, axis=0)
            Heat = np.sum(Heat_src, axis=0)
    
            # Compute Lyman-Alpha emission
            if not self.pf['approx_lya']:
                Ja = np.sum(Ja_src, axis=0)
    
            # Molecule destruction
            #kdiss = np.sum(kdiss_src, axis=0)
    
            # Each is grid x absorbers, or grid x [absorbers, absorbers] for gamma
            self.kwargs.update({'Gamma': Gamma, 'Heat': Heat, 'gamma': gamma})
            
            # Ja just has len(grid)                
            if not self.pf['approx_lya']:
                self.kwargs.update({'Ja': Ja})
    
        # Compute source independent rate coefficients
        if (not self.grid.isothermal) or (t == 0):
            self.kwargs.update(self.chem.chemnet.SourceIndependentCoefficients(data['Tk']))
    
        # SOLVE
        newdata = self.chem.Evolve(data, t, dt, **self.kwargs)
    
        return newdata
    
    def EvolvePhotonsAtFiniteSpeed(self, newdata, t, dt, h):
        """
        Solver for InfiniteSpeedOfLight = 0.
        
        PhotonPackage guide: 
            pack = [EmissionTime, EmissionTimeInterval, ncolHI, ncolHeI, ncolHeII, Energy]
        """ 
        
        # Set up timestep array for use on next cycle
        if self.AdaptiveGlobalStep:
            dtphot = 1.e50 * np.ones_like(self.grid)
        else:
            dtphot = dt
        
        Lbol = self.rs.BolometricLuminosity(t)    
        
        # Photon packages going from oldest to youngest - will have to create it on first timestep
        if t == 0: 
            packs = []
        else: 
            packs = list(self.data['PhotonPackages']) 
        
        # Add one for this timestep
        packs.append(np.array([t, dt, neglible_column, neglible_column, neglible_column, Lbol * dt]))
        
        # Loop over photon packages, updating values in cells: data -> newdata
        for j, pack in enumerate(packs):
            t_birth = pack[0]
            r_pack = (t - t_birth) * c        # Position of package before evolving photons
            r_max = r_pack + dt * c           # Furthest this package will get this timestep
        
            # Cells we need to know about - not necessarily integer
            cell_pack = (r_pack - self.R0) * self.GridDimensions / self.pf['LengthUnits']
            cell_pack_max = (r_max - self.R0) * self.GridDimensions / self.pf['LengthUnits'] - 1
            cell_t = t  
        
            Lbol = pack[-1] / pack[1]          
        
            # Advance this photon package as far as it will go on this global timestep  
            while cell_pack < cell_pack_max:
        
                # What cell are we in
                if cell_pack < 0:
                    cell = -1
                else:    
                    cell = int(cell_pack)
        
                if cell >= self.GridDimensions: 
                    break
        
                # Compute dc (like dx but in fractional cell units)
                # Really how far this photon can go in this step
                if cell_pack % 1 == 0: 
                    dc = min(cell_pack_max - cell_pack, 1)
                else: 
                    dc = min(math.ceil(cell_pack) - cell_pack, cell_pack_max - cell_pack)        
        
                # We really need to evolve this cell until the next photon package arrives, which
                # is probably longer than a cell crossing time unless the global dt is vv small.
                if (len(packs) > 1) and ((j + 1) < len(packs)): 
                    subdt = min(dt, packs[j + 1][0] - pack[0])
                else: 
                    subdt = dt
        
                # If photons haven't hit first cell interface yet, evolve in time                
                if cell < 0:
                    cell_pack += dc
                    cell_t += subdt
                    continue        
        
                # Current radius in code units                                                                                                                                                                                                                                                                                                                          
                r = cell_pack * self.pf['LengthUnits'] / self.pf['GridDimensions']
        
                # These quantities will be different (in general) for each step
                # of the while loop
                n_e = newdata["ElectronDensity"][cell]
                n_HI = newdata["HIDensity"][cell]
                n_HII = newdata["HIIDensity"][cell]
                n_HeI = newdata["HeIDensity"][cell]
                n_HeII = newdata["HeIIDensity"][cell]
                n_HeIII = newdata["HeIIIDensity"][cell] 
                n_H = n_HI + n_HII
                n_He = n_HeI + n_HeII + n_HeIII
        
                # Read in ionized fractions for this cell
                x_HI = n_HI / n_H
                x_HII = n_HII / n_H
                x_HeI = n_HeI / n_He
                x_HeII = n_HeII = n_He
                x_HeIII = n_HeIII = n_He
        
                # Compute mean molecular weight for this cell
                mu = 1. / (self.cosm.X * (1. + x_HII) + self.cosm.Y * (1. + x_HeII + x_HeIII) / 4.)
        
                # Retrieve path length through this cell
                dx = self.dx[cell]     
        
                # Crossing time
                dct = self.CellCrossingTime[cell]                        
        
                # For convenience     
                nabs = np.array([n_HI, n_HeI, n_HeII])
                nion = np.array([n_HII, n_HeII, n_HeIII])
                n_H = n_HI + n_HII
                n_He = n_HeI + n_HeII + n_HeIII
                n_B = n_H + n_He + n_e
        
                # Compute internal energy for this cell
                T = newdata["Temperature"][cell]
                E = 3. * k_B * T * n_B / mu / 2.
        
                q_cell = [n_HII, n_HeII, n_HeIII, E]
        
                # Add columns of this cell
                packs[j][2] += newdata['HIDensity'][cell] * dc * dx
                packs[j][3] += newdata['HeIDensity'][cell] * dc * dx 
                packs[j][4] += newdata['HeIIDensity'][cell] * dc * dx
                ncol = np.log10(packs[j][2:5])
        
                ######################################
                ######## Solve Rate Equations ########
                ######################################
        
                # Retrieve indices used for interpolation
                indices = self.coeff.Interpolate.GetIndices(ncol)
        
                # Retrieve coefficients and what not.
                args = [nabs, nion, n_H, n_He, n_e]                
                args.extend(self.coeff.ConstructArgs(args, indices, Lbol, r, ncol, T, dx * dc, t, self.z, cell))
        
                # Unpack so we have everything by name
                nabs, nion, n_H, n_He, n_e, Gamma, gamma, Beta, alpha, \
                    k_H, zeta, eta, psi, xi, omega, hubble, compton = args        
        
                ######################################
                ######## Solve Rate Equations ########
                ######################################                          
        
                tarr, qnew, h, odeitr, rootitr = self.solver.integrate(self.RateEquations, 
                    q_cell, cell_t, cell_t + subdt, self.z, self.z - self.dz, None, h, *args)                 
        
                # Unpack results of coupled equations - Remember: these are lists and we only need the last entry 
                newHII, newHeII, newHeIII, newE = qnew    
        
                # Weight by volume if less than a cell traversed
                if dc < 1:
                    dnHII = newHII - newdata['HIIDensity'][cell]
                    dnHeII = newHeII - newdata['HeIIDensity'][cell]
                    dnHeIII = newHeIII - newdata['HeIIIDensity'][cell]
                    dV = self.coeff.ShellVolume(r, dx * dc) / self.coeff.ShellVolume(self.r[cell], dx)
                    newHII = newdata['HIIDensity'][cell] + dnHII * dV 
                    newHeII = newdata['HeIIDensity'][cell] + dnHeII * dV
                    newHeIII = newdata['HeIIIDensity'][cell] + dnHeIII * dV
        
                # Calculate neutral fractions
                newHI = n_H - newHII
                newHeI = n_He - newHeII - newHeIII               
        
                # Convert from internal energy back to temperature
                newT = newE * 2. * mu / 3. / k_B / n_B     
        
                # Store data
                newdata = self.StoreData(newdata, cell, newHI, newHII, newHeI, newHeII, newHeIII, newT,
                    self.tau_all[cell], odeitr, h, rootitr, Gamma, k_H, Beta, alpha, zeta, eta, psi, gamma, xi, 
                    omega, hubble, compton)                    
        
                cell_pack += dc
                cell_t += subdt
        
                ######################################
                ################ DONE ################     
                ######################################  
        
        # Adjust timestep for next cycle
        if self.AdaptiveGlobalStep:
            n_HI = newdata['HIDensity']
            n_HII = newdata['HIIDensity']
            n_H_all = n_HI + n_HII
            n_HeI = newdata['HeIDensity']
            n_HeII = newdata['HeIIDensity']
            n_HeIII = newdata['HeIIIDensity']
            n_He_all = n_HeI + n_HeII + n_HeIII
            n_e_all = n_HII + n_HeII + 2 * n_HeIII
            T = newdata['Temperature']
            n_B_all = n_H_all + n_He_all + n_e_all
        
            ncol_HI = np.roll(np.cumsum(n_HI * self.dx), 1)
            ncol_HeI = np.roll(np.cumsum(n_HeI * self.dx), 1)
            ncol_HeII = np.roll(np.cumsum(n_HeII * self.dx), 1)
            ncol_HI[0] = ncol_HeI[0] = ncol_HeII[0] = neglible_column
            ncol = np.transpose(np.log10([ncol_HI, ncol_HeI, ncol_HeII]))   
        
            tau = self.ComputeOpticalDepths([ncol_HI, ncol_HeI, ncol_HeII])
        
            for cell in self.grid:
                r = self.r[cell]
                dx = self.dx[cell]
                nabs = np.array([n_HI[cell], n_HeI[cell], n_HeII[cell]])
                nion = np.array([n_HII[cell], n_HeII[cell], n_HeIII[cell]])
        
                indices = self.coeff.Interpolate.GetIndices(ncol[cell])
        
                args = [nabs, nion, n_H_all[cell], n_He_all[cell], n_e_all[cell]]  
                args.extend(self.coeff.ConstructArgs(args, indices, Lbol, r, ncol[cell], T[cell], dx, t, self.z))
        
                # Unpack so we have everything by name
                nabs, nion, n_H, n_He, n_e, Gamma, gamma, Beta, alpha, \
                    k_H, zeta, eta, psi, xi, omega, hubble, compton = args 
        
                dtphot[cell] = self.control.ComputePhotonTimestep(tau[:,cell], 
                    nabs, nion, ncol[cell], n_H, n_He, 
                    n_e, n_B_all[cell], Gamma, gamma, Beta, alpha, k_H, zeta, 
                    eta, psi, xi, omega, hubble, compton, T[cell], self.z, dt) 
        
                if self.pf['LightCrossingTimeRestrictedTimestep']: 
                    dtphot[cell] = min(dtphot[cell], 
                        self.LightCrossingTimeRestrictedTimestep * self.CellCrossingTime[cell])    
        
        # Update photon packages   
        newdata['PhotonPackages'] = np.array(self.UpdatePhotonPackages(packs, t + dt))
        
        return newdata, dtphot    
           
    def SourceDependentCoefficients(self, data, t, z=None, **kwargs):
        """
        Compute rate coefficients for photo-ionization, secondary ionization, 
        and photo-heating.
        """
        
        self.k_H = np.array(self.Ns*[np.zeros_like(self.grid.zeros_grid_x_absorbers)])
        self.Gamma = np.array(self.Ns*[np.zeros_like(self.grid.zeros_grid_x_absorbers)])
        self.gamma = np.array(self.Ns*[np.zeros_like(self.grid.zeros_grid_x_absorbers2)])

        if self.pf['approx_lya']:
            self.Ja = [None] * self.Ns
        else:
            self.Ja = np.array(self.Ns * [np.zeros(self.grid.dims)])

        # H2 dissociation
        #self.kdiss = np.array(self.Ns*[np.zeros_like(self.grid.zeros_grid_x_absorbers)])

        # Column density to cells (N) and of cells (Nc)
        if not self.pf['optically_thin'] or self.all_diffuse:
            self.N, self.logN, self.Nc = self.grid.ColumnDensity(data)
            
            # Column densities (of all absorbers) sorted by cell 
            # (i.e. an array with shape = grid cells x # of absorbers)
            self.N_by_cell = np.zeros([self.grid.dims, len(self.grid.absorbers)])
            self.Nc_by_cell = np.zeros([self.grid.dims, len(self.grid.absorbers)])
            for i, absorber in enumerate(self.grid.absorbers):
                self.N_by_cell[...,i] = self.N[absorber]
                self.Nc_by_cell[...,i] = self.Nc[absorber]

            self.logN_by_cell = np.log10(self.N_by_cell)
            self.logNc_by_cell = np.log10(self.N_by_cell)
            
            # Number densities
            self.n = {}
            for absorber in self.grid.absorbers:
                self.n[absorber] = data[absorber] * self.grid.x_to_n[absorber]
                          
            # Compute column densities up to and of cells        
            if self.pf['photon_conserving']:
                self.NdN = self.grid.N_absorbers \
                    * [np.zeros_like(self.grid.zeros_grid_x_absorbers)]
                for i, absorber in enumerate(self.grid.absorbers):
                    tmp = self.N_by_cell.copy()
                    tmp[..., i] += self.Nc_by_cell[..., i]
                    self.NdN[i] = tmp
                    del tmp
                
                self.logNdN = np.log10(self.NdN)
                          
        # Loop over sources
        for h, src in enumerate(self.srcs):        
            
            if not src.SourceOn(t):
                continue
                
            # "Diffuse" sources have parameterized rates    
            if src.SourcePars['type'] == 'diffuse':
                
                # If rate coefficients are in kwargs, 
                # it means we're working on an IGM grid patch
                if 'Gamma' in kwargs:
                    
                    # These if/else's are for glorb!
                    # Have to be sure we don't double count ionization/heat...
                    if src.pf['is_ion_src_igm']:
                        self.Gamma[h] = kwargs['Gamma']
                    else:
                        self.Gamma[h] = 0.0
                else:
                    for i, absorber in enumerate(self.grid.absorbers):
                        if self.pf['approx_helium'] and i > 0:
                            continue
                        
                        self.Gamma[h,:,i] = \
                            src.ionization_rate(z, species=i, **kwargs)
                
                if 'gamma' in kwargs:
                    if src.pf['is_ion_src_igm']:
                        self.gamma[h] = kwargs['gamma']
                    else:
                        self.gamma[h] = 0.0
                else:
                    for i, absorber in enumerate(self.grid.absorbers):
                        if self.pf['approx_helium'] and i > 0:
                            continue
                            
                        self.gamma[h,:,i] = \
                            src.secondary_ionization_rate(z, species=i, **kwargs)
                
                if 'epsilon_X' in kwargs:
                    if src.pf['is_heat_src_igm']:
                        self.k_H[h] = kwargs['epsilon_X']
                    else:
                        self.k_H[h] = 0.0
                else:
                    for i, absorber in enumerate(self.grid.absorbers):
                        if self.pf['approx_helium'] and i > 0:
                            continue
                            
                        self.k_H[h,:,i] = src.heating_rate(z, species=i, **kwargs)
            
                continue    
                
            self.h = h
            self.src = src
                
            # If we're operating under the optically thin assumption, 
            # return pre-computed source-dependent values.    
            if self.pf['optically_thin']:
                self.tau_tot = np.zeros(self.grid.dims) # by definition
                self.Gamma[h] = src.Gamma_bar * self.pp_corr
                self.k_H[h] = src.Heat_bar * self.pp_corr
                self.gamma[h] = src.gamma_bar * self.pp_corr
                continue

            # Normalizations
            self.A = {}
            for absorber in self.grid.absorbers:          
                
                if self.pf['photon_conserving']:
                    self.A[absorber] = self.src.Lbol \
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
                        self.Gamma[h,:,i], self.gamma[h,:,i], self.k_H[h,:,i] = \
                            self.MultiFreqCoefficients(data, absorber)
                    
                    # Discrete spectrum (multi-grp approach)
                    elif self.src.multi_group:
                        pass
                
                continue
                
            """
            For sources with continuous SEDs.
            """
            
            # This could be post-processed, but eventually may be more
            # sophisticated
            if self.pf['approx_lya']:
                self.Ja = None
            else:
                self.Ja[h] = src.Spectrum(E_LyA) * ev_per_hz \
                    * src.Lbol / 4. / np.pi / self.grid.r_mid**2 \
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
                    10**self.src.tables["logPhi_%s" % absorber](self.logN_by_cell,
                    self.logx, t)
                
                if (not self.pf['isothermal']) and (self.pf['secondary_ionization'] < 2):
                    self.PsiN[absorber] = \
                        10**self.src.tables["logPsi_%s" % absorber](self.logN_by_cell,
                        self.logx, t)
                    
                if self.pf['photon_conserving']:
                    self.PhiNdN[absorber] = \
                        10**self.src.tables["logPhi_%s" % absorber](self.logNdN[i],
                        self.logx, t)
                    
                    if (not self.pf['isothermal']) and (self.pf['secondary_ionization'] < 2):
                        self.PsiNdN[absorber] = \
                            10**self.src.tables["logPsi_%s" % absorber](self.logNdN[i],
                            self.logx, t)
            
                if self.pf['secondary_ionization'] > 1:
                    
                    self.PhiHatN[absorber] = \
                        10**self.src.tables["logPhiHat_%s" % absorber](self.logN_by_cell,
                        self.logx, t)    
                                        
                    if not self.pf['isothermal']:
                        self.PsiHatN[absorber] = \
                            10**self.src.tables["logPsiHat_%s" % absorber](self.logN_by_cell,
                            self.logx, t)  
                                                
                        if self.pf['photon_conserving']:    
                            self.PhiHatNdN[absorber] = \
                                10**self.src.tables["logPhiHat_%s" % absorber](self.logNdN[i],
                                self.logx, t)
                            self.PsiHatNdN[absorber] = \
                                10**self.src.tables["logPsiHat_%s" % absorber](self.logNdN[i],
                                self.logx, t)     
                    
                    for j, donor in enumerate(self.grid.absorbers):
                        
                        suffix = '%s_%s' % (absorber, donor)
                        
                        self.PhiWiggleN[absorber][donor] = \
                            10**self.src.tables["logPhiWiggle_%s" % suffix](self.logN_by_cell,
                                self.logx, t)    
                        
                        self.PsiWiggleN[absorber][donor] = \
                            10**self.src.tables["logPsiWiggle_%s" % suffix](self.logN_by_cell,
                            self.logx, t)
                            
                        if not self.pf['photon_conserving']:
                            continue
                        
                        self.PhiWiggleNdN[absorber][donor] = \
                            10**self.src.tables["logPhiWiggle_%s" % suffix](self.logNdN[j],
                            self.logx, t)
                        self.PsiWiggleNdN[absorber][donor] = \
                            10**self.src.tables["logPsiWiggle_%s" % suffix](self.logNdN[j],
                            self.logx, t)

            # Now, go ahead and calculate the rate coefficients
            for k, absorber in enumerate(self.grid.absorbers):
                self.Gamma[h][...,k] = self.PhotoIonizationRate(absorber)
                self.k_H[h][...,k] = self.PhotoHeatingRate(absorber)

                for j, donor in enumerate(self.grid.absorbers):
                    self.gamma[h][...,k,j] = \
                        self.SecondaryIonizationRate(absorber, donor)
                       
            # Compute total optical depth too
            self.tau_tot = 10**self.src.tables["logTau"](self.logN_by_cell)
            
        return self.Gamma, self.gamma, self.k_H, self.Ja
        
    def MultiFreqCoefficients(self, data, absorber):
        """
        Compute all source-dependent rates for given absorber assuming a
        multi-frequency SED.
        """
        
        #k_H = np.zeros_like(self.grid.zeros_grid_x_absorbers)
        #Gamma = np.zeros_like(self.grid.zeros_grid_x_absorbers)
        #gamma = np.zeros_like(self.grid.zeros_grid_x_absorbers2)
        
        k_H = np.zeros(self.grid.dims)
        #Gamma = np.zeros(self.grid.dims)
        gamma = np.zeros_like(self.grid.zeros_grid_x_absorbers)
        
        i = self.grid.absorbers.index(absorber)
        n = self.n[absorber]
        N = self.N[absorber]
               
        # Optical depth up to cells at energy E
        N = np.ones([self.src.Nfreq, self.grid.dims]) * self.N[absorber]
        
        self.tau_r = N * self.sigma[self.h]
        self.tau_tot = np.sum(self.tau_r, axis = 1)
                        
        # Loop over energy groups
        Gamma_E = np.zeros([self.grid.dims, self.src.Nfreq])
        for j, E in enumerate(self.src.E):
            
            if E < self.E_th[absorber]:
                continue    
            
            # Optical depth of cells (at this photon energy)                                                           
            tau_c = self.Nc[absorber] * self.src.sigma[j]
                                                            
            # Photo-ionization by *this* energy group
            Gamma_E[...,j] = \
                self.PhotoIonizationRateMultiFreq(self.src.Qdot[j], n,
                self.tau_r[j], tau_c)            
                          
            # Heating
            if self.grid.isothermal:
                continue
                 
            fheat = self.esec.DepositionFraction(xHII=data['h_2'], 
                E=E, channel='heat')
            
            # Total energy deposition rate per atom i via photo-electrons 
            # due to ionizations by *this* energy group. 
            ee = Gamma_E[...,j] * (E - self.E_th[absorber]) \
               * erg_per_ev
            
            k_H += ee * fheat
                
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
                gamma[...,k,i] += ee * fion \
                    / (self.E_th[otherabsorber] * erg_per_ev)
                                                                           
        # Total photo-ionization tally
        Gamma = np.sum(Gamma_E, axis=1)
        
        return Gamma, gamma, k_H
    
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

        if self.esec.Method < 2:
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
        
        if self.esec.Method < 2:
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
        
        
        