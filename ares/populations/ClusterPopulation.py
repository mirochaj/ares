"""

ClusterPopulation.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Wed Jan  3 15:08:08 PST 2018

Description: 

"""

import os
import re
import inspect
import numpy as np
from ..util import read_lit
from types import FunctionType
from ..util.Stats import rebin
from ..util.Math import interp1d
from .Population import Population
from ..util import MagnitudeSystem
from ..util.ParameterFile import get_pq_pars
from ..phenom.ParameterizedQuantity import ParameterizedQuantity
from ..physics.Constants import s_per_yr, s_per_myr, ev_per_hz, g_per_msun, \
    cm_per_mpc

class ClusterPopulation(Population):
    
    @property
    def dust(self):
        if not hasattr(self, '_dust'):
            self._dust = DustCorrection(**self.pf)
        return self._dust
    
    @property
    def magsys(self):
        if not hasattr(self, '_magsys'):
            self._magsys = MagnitudeSystem(**self.pf)
        return self._magsys
        
    def LuminosityDensity(self):
        pass
        
    def _sfrd_func(self, z):
        # This is a cheat so that the SFRD spline isn't constructed
        # until CALLED. Used only when linking populations.
        return self.SFRD(z)
        
    def SFRD(self, z):
        on = self.on(z)
        if not np.any(on):
            return z * on

        sfrd = self.FRD(z=z) * self.Mavg(z) * g_per_msun

        return sfrd / cm_per_mpc**3 / s_per_yr
    
    def FRD(self, **kwargs):
        if 'z' in kwargs:
            z = kwargs['z']
            on = self.on(z)
            if not np.any(on):
                return z * on
        else: 
            on = 1        
            
        return on * self._frd(**kwargs)
    
    @property
    def _frd(self):
        """
        Formation rate density in # of clusters / Myr / cMpc^3.
        """
        if not hasattr(self, '_frd_'):
            if self.pf['pop_frd'] is None:
                self._frd_ = None
            if type(self.pf['pop_frd']) in [int, float, np.float]:
                self._frd_ = lambda **kwargs: self.pf['pop_frd']
            elif type(self.pf['pop_frd']) is FunctionType:
                self._frd_ = self.pf['pop_frd']
            elif inspect.ismethod(self.pf['pop_frd']):
                self._frd_ = self.pf['pop_frd']
            elif isinstance(self.pf['pop_frd'], interp1d):
                self._frd_ = self.pf['pop_frd']  
            elif self.pf['pop_frd'][0:2] == 'pq':
                pars = get_pq_pars(self.pf['pop_frd'], self.pf)
                self._frd_ = ParameterizedQuantity(**pars)    
            else:
                tmp = read_lit(self.pf['pop_frd'], verbose=self.pf['verbose'])
                self._frd_ = lambda z: tmp.FRD(z, **self.pf['pop_kwargs'])
    
        return self._frd_
                
    def MassFunction(self, **kwargs):
        """
        Un-normalized.
        """
        
        return self._mdist(**kwargs)
        
        #iz = np.argmin(np.abs(kwargs['z'] - self.zarr))
        #
        #frd = np.array([self.FRD(z=z) for z in self.zarr])
        #mdist = np.array([self._mdist(z=z, M=kwargs['M']) for z in self.zarr])
        #y = frd * mdist / self._mdist_norm
        #
        #return np.trapz(y[iz:], x=self.tarr[iz:])

    @property
    def _tab_massfunc(self):
        if not hasattr(self, '_tab_massfunc_'):
            self._tab_massfunc_ = np.zeros((len(self.zarr), len(self.Marr)))
            
            # Loop over formation redshifts.
            for i, z in enumerate(self.zarr):
                
                frd = np.array([self.FRD(z=zz) \
                    for zz in self.zarr[i:]]) * 1e6 # since tarr in Myr
                mdist = np.array([self._mdist(z=zz, M=self.Marr) \
                    for zz in self.zarr[i:]]) / self._mdist_norm
                
                for j, M in enumerate(self.Marr):
                    #self._tab_agefunc_[i,i:] = self.ages
                    self._tab_massfunc_[i,j] = np.trapz(frd * mdist[:,j], 
                        x=self.tarr[i:])
                
                    # Luminosity function integrates along age, not mass.
                    #self._tab_lumfunc[i,i:] = np.trapz()
                
        return self._tab_massfunc_    
    
    #@property
    #def _tab_agefunc(self):
    #    if not hasattr(self, '_tab_agefunc_'):
    #        self._tab_agefunc_ = np.zeros((len(self.zarr), len(self.zarr)))
    #                
    #        
    
    @property
    def _tab_rho_L(self):
        if not hasattr(self, '_tab_rho_L_'):
            lf = self._tab_lf
        return self._tab_rho_L_
        
    @property
    def _tab_Nc(self):
        if not hasattr(self, '_tab_Nc_'):
            lf = self._tab_lf
        return self._tab_Nc_
            
    @property
    def _tab_lf(self):
        if not hasattr(self, '_tab_lf_'):
            
            if self.is_aging:
                self._tab_lf_ = np.zeros((len(self.zarr), len(self.Larr)))
            else:
                self._tab_lf_ = np.zeros((len(self.zarr), len(self.Marr)))
                                            
            dt = self.pf['pop_age_res']
            dz_all = np.diff(self.zarr)
                        
            # Number of clusters as a function of (zobs, age, mass)
            self._tab_Nc_ = np.zeros((len(self.zarr), len(self.Marr), 
                len(self.zarr)))
                
            self._tab_ages = np.zeros((len(self.zarr), len(self.Marr),
                len(self.zarr)))
            
            # These are observed redshifts, so we must integrate over
            # all higher redshifts to get the luminosity function.
            for i, z in enumerate(self.zarr):
             
                if i == len(self.zarr) - 2:
                    # Do this
                    break
                                                
                # If we're not allowing this population to age, 
                # things get a lot easier.
                if not self.is_aging:         
                    self._tab_lf_[i] = self.FRD(z=z) * dt * 1e6 \
                        * self._mdist(z=z, M=self.Marr)
                                     
                    continue
                
                ##
                # If we're here, it means this population can age
                ##
                
                # At this redshift of observation, we're seeing clusters of
                # a range of ages. At each age, we need to weight by the 
                # formation rate density * mass function at the corresponding
                # birth redshift to get the UV luminosity now.
                    
                # First, calculate number of clusters as a function of 
                # mass (luminosity) and age.
                
                zarr = self.zarr[0:i+1]       
                frd =  self.FRD(z=zarr)
                mdist = np.array([self._mdist(z=z, M=self.Marr) for z in zarr])
                                                                
                # Integral (sum) over differential redshift elements
                # to obtain number of clusters formed in each redshift interval
                Nc_of_M_z = frd * dt * 1e6 * mdist.T #* np.diff(dz_all)[0:i+1]
                # This has shape (M, z)
                 
                self._tab_Nc_[i,:,0:i+1] = Nc_of_M_z
                                                                                
                # Age distribution of clusters formed at z.
                zpre = self.zarr[0:i]
                
                # Age of all clusters formed between now and first formation
                # redshift.
                ages = dt * np.arange(len(zarr))[-1::-1]
                                                
                L = np.interp(ages, self.ages, self._tab_L1600)
                                                    
                # For each source population formed at z > znow, determine
                # luminosity (as function of age) and scale by mass.
                Lnow = np.array([L[k] * self.Marr \
                    for k in range(len(ages))]).T
                    
                # At this point, we have an array Nc_of_M_z that represents
                # the number of clusters as a function of (mass, age).
                # So, we convert from age to luminosity, weight by mass, 
                # and then histogram in luminosity.

                # It seems odd to histogram here, but I think we must, since
                # mass and age can combine to produce a continuum of 
                # luminosities, i.e., we can't just integrate along one 
                # dimension.    
                    
                # Histogram: number of clusters in given luminosity bins.
                lf, bin_e = np.histogram(Lnow.flatten(), bins=self.Larr_e,
                    weights=Nc_of_M_z.flatten())

                self._tab_lf_[i] = lf
                
        return self._tab_lf_
        
    def rho_N(self, Emin=None, Emax=None):
        if not hasattr(self, '_rho_N'):
            self._rho_N = {}
            
        # If we've already figured it out, just return    
        if (Emin, Emax) in self._rho_N:
            return self._rho_N[(Emin, Emax)]
        
        rho_L = self.rho_L(Emin, Emax)
        return self._rho_N[(Emin, Emax)]
        
    def rho_L(self, Emin=None, Emax=None):
        
        if not hasattr(self, '_rho_L'):
            self._rho_L = {}
            self._rho_N = {}
            
        # If we've already figured it out, just return    
        if (Emin, Emax) in self._rho_L:
            return self._rho_L[(Emin, Emax)]
            
        # Important change needed: not L1600, but integrated luminosity
        # at each age.
        
        # This is in [erg / s / g]. Must convert to Msun.
        tmp = self.src.rad_yield(Emin, Emax) * g_per_msun
        yield_per_M = np.interp(self.ages, self.src.times, tmp)
        
        tmp = self.src.erg_per_phot(Emin, Emax)
        erg_per_phot = np.interp(self.ages, self.src.times, tmp)
        
        #Lmin = np.log10(self.Marr.min() * yield_per_M.min())
        #Lmax = np.log10(self.Marr.max() * yield_per_M.max())
        #Larr = np.logspace(Lmin, Lmax, self.Larr.size)
        #
        #dlogL = np.diff(np.log10(Larr))[0]
        #edges = 10**np.arange(np.log10(Larr[0]) - 0.5 * dlogL,
        #            np.log10(Larr[-1]) + 0.5 * dlogL, dlogL)
        
        #dLdL = np.diff(self.Larr) / np.diff(Larr)
    
        dt = self.pf['pop_age_res']
    
        _tab_rho_L_ = np.zeros_like(self.zarr)
        _tab_rho_N_ = np.zeros_like(self.zarr)
        
        # Loop over redshift
        for i, z in enumerate(self.zarr):
            
            # Should have shape (Marr, len(zarr[0:i+1]))                  
            Nc = self._tab_Nc[i,:,0:i+1]
            
            if not self.is_aging:
                _tab_rho_L_[i] = yield_per_M[0] * self.FRD(z=z) * np.mean(self.Marr)
                continue

            if i == 0:
                continue

            if i == len(self.zarr) - 2:
                # Do this
                break

            zarr = self.zarr[0:i+1]    
            zpre = zarr[0:-1]

            ages = dt * np.arange(len(zarr))[-1::-1]
                                        
            L = np.interp(ages, self.ages, yield_per_M)
            N = np.interp(ages, self.ages, yield_per_M / erg_per_phot)
                                        
            # Compute the emissivity as an integral over the LF.
            # Be careful since conversion factor is age-dependent.
            Larr = np.array([L[k] * self.Marr \
                for k in range(len(ages))]).T    

            dlogL = self.pf['pop_dlogM']
                            
            edges = 10**np.arange(np.log10(Larr.min()) - 0.5 * dlogL,
                        np.log10(Larr.max()) + 0.5 * dlogL, dlogL)
                        
            # Compute LF
            lf, bin_e = np.histogram(Larr.flatten(), bins=edges,
                weights=Nc.flatten())
                        
            Lnow = rebin(edges)
                        
            # Compute luminosity density.
            _tab_rho_L_[i] = np.trapz(lf * Lnow, dx=dlogL)
            
            Narr = np.array([N[k] * self.Marr \
                for k in range(len(ages))]).T
            
            edges = 10**np.arange(np.log10(Narr.min()) - 0.5 * dlogL,
                        np.log10(Narr.max()) + 0.5 * dlogL, dlogL)

            # Compute luminosity density in units of photon number.            
            lf, bin_e = np.histogram(Narr.flatten(), bins=edges,
                weights=Nc.flatten())
            
            Nnow = rebin(edges)
            
            # Photon number as well.
            _tab_rho_N_[i] = np.trapz(lf * Nnow, dx=dlogL)
        
        
        # Not as general as it could be right now...
        if (Emin, Emax) == (13.6, 24.6):
            _tab_rho_L_ *= self.pf['pop_fesc']
            _tab_rho_N_ *= self.pf['pop_fesc']
            
        self._rho_L[(Emin, Emax)] = interp1d(self.zarr[-1::-1], 
            _tab_rho_L_[-1::-1] / cm_per_mpc**3,
            kind=self.pf['pop_interp_sfrd'], bounds_error=False,
            fill_value=0.0)
        
        self._rho_N[(Emin, Emax)] = interp1d(self.zarr[-1::-1], 
            _tab_rho_N_[-1::-1] / cm_per_mpc**3,
            kind=self.pf['pop_interp_sfrd'], bounds_error=False,
            fill_value=0.0)
            
        return self._rho_L[(Emin, Emax)]
                
    def LuminosityFunction(self, z):
        
        iz = np.argmin(np.abs(self.zarr - z))
        
        mags = self.mags(z=z)
        
        # Remember that this is a histogram in log10(L) bins.
        phi = self._tab_lf[iz]
                
        dLdmag = np.diff(np.log10(self.Larr)) / np.diff(mags)
        
        return mags[0:-1], phi[0:-1] * np.abs(dLdmag)
        
    def rho_GC(self, z):
        mags, phi = self.LuminosityFunction(z)
        
        return np.trapz(phi, dx=abs(np.diff(mags)[0]))
        
    @property
    def _mdist_norm(self):
        if not hasattr(self, '_mdist_norm_'):
            ##
            # Wont' work if mdist is redshift-dependent.
            ## HELP
            self._mdist_norm_ = np.trapz(self._mdist(M=self.Marr), x=self.Marr)
    
        return self._mdist_norm_
    
    @property
    def _mdist(self):
        if not hasattr(self, '_mdist_'):
            if self.pf['pop_mdist'] is None:
                self._mdist_ = None
            if type(self.pf['pop_mdist']) in [int, float, np.float]:
                self._mdist_ = lambda **kw: self.pf['pop_mdist']
            elif type(self.pf['pop_mdist']) is FunctionType:
                self._mdist_ = self.pf['pop_mdist']
            elif inspect.ismethod(self.pf['pop_mdist']):
                self._mdist_ = self.pf['pop_mdist']
            elif self.pf['pop_mdist'][0:2] == 'pq':
                pars = get_pq_pars(self.pf['pop_mdist'], self.pf)
                self._mdist_ = ParameterizedQuantity(**pars)    
            elif isinstance(self.pf['pop_mdist'], interp1d):
                self._mdist_ = self.pf['pop_mdist']
            else:
                raise NotImplemented('help')
                tmp = read_lit(self.pf['pop_mdist'], verbose=self.pf['verbose'])
                self._mdist_ = lambda z: tmp.SFRD(z, **self.pf['pop_kwargs'])
        
        return self._mdist_
        
    @property
    def Larr(self):
        if not hasattr(self, '_Larr'):
            # Setup array of luminosities spanning full range of possibilities
            # from youngest to oldest, least massive cluster to most massive
            # cluster allowed. Unless we're not allowing this cluster to age,
            # in which case the luminosity is easily related to mass function.
            
            if self.is_aging:
                Lmin = np.log10(self.Marr.min() * self._tab_L1600.min())
                Lmax = np.log10(self.Marr.max() * self._tab_L1600.max())
                dlogL = self.pf['pop_dlogM']
                self._Larr = 10**np.arange(Lmin, Lmax+dlogL, dlogL)
            else:
                self._Larr = self._tab_L1600[0] * self.Marr
                
        return self._Larr
        
    @property
    def Larr_e(self):
        if not hasattr(self, '_Larr_e'):
            dlogL = self.pf['pop_dlogM']
            edges = 10**np.arange(np.log10(self.Larr[0]) - 0.5 * dlogL,
                        np.log10(self.Larr[-1]) + 0.5 * dlogL, dlogL)    
        
            self._Larr_e = edges
        
        return self._Larr_e
        
    def mags(self, z):
        return self.magsys.L_to_MAB(self.Larr, z=z)
        
    @property
    def Marr(self):
        if not hasattr(self, '_Marr'):
            lMmin = np.log10(self.pf['pop_Mmin'])
            lMmax = np.log10(self.pf['pop_Mmax'])
            dlogM = self.pf['pop_dlogM']
            self._Marr = 10**np.arange(lMmin, lMmax+dlogM, dlogM)
        
        return self._Marr
        
    def Mavg(self, z):
        pdf = self._mdist(z=z, M=self.Marr)
        norm = np.trapz(pdf, x=self.Marr)
        
        return np.trapz(pdf * self.Marr, x=self.Marr) / norm
        
    @property
    def zarr(self):
        if not hasattr(self, '_zarr'):
            ages = self.ages
        return self._zarr
    
    @property
    def tarr(self):
        if not hasattr(self, '_zarr'):
            ages = self.ages
        return self._tarr

    @property
    def ages(self):
        """
        Array of ages corresponding to redshifts at which we tabulate LF.
        """
        if not hasattr(self, '_ages'):
            zf = self.pf['final_redshift']
            ti = self.cosm.t_of_z(self.zform) / s_per_myr
            tf = self.cosm.t_of_z(zf) / s_per_myr
            # Time since Big Bang
            dt = self.pf['pop_age_res']
            self._tarr = np.arange(ti, tf+2*dt, dt)
            self._zarr = self.cosm.z_of_t(self._tarr * s_per_myr)

            if self._zarr[-1] > zf:
                self._zarr[-1] = zf
                self._tarr[-1] = self.cosm.t_of_z(zf) / s_per_myr

            # Of clusters formed at corresponding element of zarr
            self._ages = self._tarr - ti

        return self._ages

    @property
    def _tab_L1600(self):
        if not hasattr(self, '_tab_L1600_'):
            self._tab_L1600_ = np.interp(self.ages, self.src.times,
                self.src.L_per_SFR_of_t())

        return self._tab_L1600_
    
    def Emissivity(self, z, E=None, Emin=None, Emax=None):
        """
        Compute the emissivity of this population as a function of redshift
        and rest-frame photon energy [eV].
    
        Parameters
        ----------
        z : int, float
    
        Returns
        -------
        Emissivity in units of erg / s / c-cm**3 [/ eV]

        """
        
        if not self.is_aging:
            on = self.on(z)
            if not np.any(on):
                return z * on
        else:
            on = 1.
            
        if self.pf['pop_sed_model'] and (Emin is not None) and (Emax is not None):
            if (Emin > self.pf['pop_Emax']):
                return 0.0
            if (Emax < self.pf['pop_Emin']):
                return 0.0    
    
        if self.is_emissivity_separable:
            # The table is in L1600, so we need to convert to broad-band
            # emissivity.
            rhoL = self.rho_L(Emin=Emin, Emax=Emax)(z)
        else:
            raise NotImplemented('help!')
        
        #if not self.pf['pop_sed_model']:
        #    if (Emin, Emax) == (10.2, 13.6):
        #        return rhoL * self.pf['pop_Nlw'] * self.pf['pop_fesc_LW']
        #    elif (Emin, Emax) == (13.6, 24.6):
        #        return rhoL * self.pf['pop_Nion'] * self.pf['pop_fesc']
        #    else:
        #        return rhoL
        
        # Convert from reference band to arbitrary band
        #rhoL *= self._convert_band(Emin, Emax)
        #if (Emax is None) or (Emin is None):
        #    pass
        #elif Emax > 13.6 and Emin < self.pf['pop_Emin_xray']:
        #    rhoL *= self.pf['pop_fesc']
        #elif Emax <= 13.6:
        #    rhoL *= self.pf['pop_fesc_LW']    
        
        if E is not None:
            return rhoL * self.src.Spectrum(E)
        else:
            return rhoL


    def PhotonLuminosityDensity(self, z, Emin=None, Emax=None):
        """
        Return the photon luminosity density in the (Emin, Emax) band.
    
        Parameters
        ----------
        z : int, flot
            Redshift of interest.
    
        Returns
        -------
        Photon luminosity density in photons / s / c-cm**3.

        """

        # erg / s / cm**3
        if self.is_emissivity_scalable:
            rhoL = self.Emissivity(z, E=None, Emin=Emin, Emax=Emax)
            erg_per_phot = self._get_energy_per_photon(Emin, Emax) * erg_per_ev
    
            return rhoL / erg_per_phot
        else:
            return self.rho_N(Emin, Emax)(z)
            
            
            