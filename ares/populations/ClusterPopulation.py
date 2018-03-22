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
from .Population import Population
from ..util import MagnitudeSystem
from scipy.interpolate import interp1d
#from .GalaxyAggregate import GalaxyAggregate
from ..util.ParameterFile import get_pq_pars
from ..physics.Constants import s_per_yr, s_per_myr
from ..phenom.ParameterizedQuantity import ParameterizedQuantity

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
    
    def FRD(self, **kwargs):
        if kwargs['z'] < self.zdead:
            return 0.0
            
        return self._frd(**kwargs)
    
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
        
        iz = np.argmin(np.abs(kwargs['z'] - self.zarr))
        
        frd = np.array([self.FRD(z=z) for z in self.zarr]) * 1e6
        mdist = np.array([self._mdist(z=z, M=kwargs['M']) for z in self.zarr])
        y = frd * mdist / self._mdist_norm

        return np.trapz(y[iz:], x=self.tarr[iz:])

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
    def _tab_lf(self):
        if not hasattr(self, '_tab_lf_'):
            self._tab_lf_ = np.zeros((len(self.zarr), len(self.Larr)))
            
            dlogL = self.pf['pop_dlogM']
            edges = 10**np.arange(np.log10(self.Larr[0]) - 0.5 * dlogL,
                              np.log10(self.Larr[-1]) + 0.5 * dlogL, dlogL)
            
            # These are observed redshifts, so we must integrate over
            # all higher redshifts to get the luminosity function.
            for i, z in enumerate(self.zarr):
                
                if i == 0:
                    continue
                    
                if i == len(self.zarr) - 2:
                    # Do this
                    continue
                
                # Convert FRD to Myr^-1 units.
                mdist = np.array([self._mdist(z=zz, M=self.Marr) \
                    * self.FRD(z=zz) * 1e6 \
                    for zz in self.zarr[0:i+1]]) / self._mdist_norm
                    
                # Clusters / Myr / cMpc^3 / mass-bin    
                
                # Age distribution of clusters formed at z.
                zpre = self.zarr[0:i]
                dt = self.pf['pop_age_res']
                ages = dt * np.arange(len(self.zarr[0:i+1]))[-1::-1]
                
                L = np.interp(ages, self.src.times, self.src.L_per_SFR_of_t())
                
                # For each source population formed at z > znow, determine
                # luminosity (as function of age) and scale by mass.
                Lnow = np.array([L[k] * self.Marr \
                    for k in range(len(ages))])

                # Histogram: number of galaxies in given luminosity bins.
                hist, bin_e = np.histogram(Lnow.flatten(), bins=edges,
                    weights=mdist.flatten())

                # Norm to get total number right
                norm = np.trapz(hist, x=np.log10(self.Larr))
                tot = np.trapz(self._tab_massfunc[i], x=self.Marr)

                self._tab_lf_[i] = hist * tot / norm

        return self._tab_lf_
                
    def LuminosityFunction(self, z):
        
        iz = np.argmin(np.abs(self.zarr - z))
        
        mags = self.mags(z=z)
        phi = self._tab_lf[iz]
        
        return mags, phi
        
        dLdmag = np.diff(self.Larr) / np.diff(mags)
        
        return mags[0:-1], phi[0:-1] * np.abs(dLdmag)
        
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
            elif isinstance(self.pf['pop_mdist'], interp1d):
                self._mdist_ = self.pf['pop_mdist']  
            elif self.pf['pop_mdist'][0:2] == 'pq':
                pars = get_pq_pars(self.pf['pop_mdist'], self.pf)
                self._mdist_ = ParameterizedQuantity(**pars)    
            else:
                raise NotImplemented('help')
                tmp = read_lit(self.pf['pop_mdist'], verbose=self.pf['verbose'])
                self._mdist_ = lambda z: tmp.SFRD(z, **self.pf['pop_kwargs'])
        
        return self._mdist_
        
    @property
    def Larr(self):
        if not hasattr(self, '_Larr'):
            Lmin = np.log10(self.Marr.min() * self._tab_L1600.min())
            Lmax = np.log10(self.Marr.max() * self._tab_L1600.max())
            dlogL = self.pf['pop_dlogM']
            self._Larr = 10**np.arange(Lmin, Lmax+dlogL, dlogL)
        return self._Larr
        
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
        if not hasattr(self, '_ages'):
            zf = self.pf['final_redshift']
            ti = self.cosm.t_of_z(self.zform) / s_per_myr
            tf = self.cosm.t_of_z(zf) / s_per_myr
            # Time since Big Bang
            self._tarr = np.arange(ti, tf+self.pf['pop_age_res'],
                self.pf['pop_age_res'])
            self._zarr = self.cosm.z_of_t(self._tarr * s_per_myr)

            if self._zarr[-1] != zf:
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
    

