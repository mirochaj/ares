"""

Halo.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu May 28 16:22:44 MDT 2015

Description: 

"""

import numpy as np
from ..util import read_lit
from inspect import ismethod
from types import FunctionType
from .Population import Population
from scipy.integrate import cumtrapz
from ..util.PrintInfo import print_pop
from scipy.interpolate import interp1d
from ..physics.HaloModel import HaloModel
from ..physics.HaloMassFunction import HaloMassFunction
from ..util.Math import central_difference, forward_difference
from ..physics.Constants import cm_per_mpc, s_per_yr, g_per_msun

class HaloPopulation(Population):
    def __init__(self, **kwargs):
        
        # This is basically just initializing an instance of the cosmology
        # class. Also creates the parameter file attribute ``pf``.
        Population.__init__(self, **kwargs)

    @property
    def parameterized(self):
        if not hasattr(self, '_parameterized'):
            not_parameterized = (self.pf['pop_k_ion_igm']) is None
            not_parameterized &= (self.pf['pop_k_ion_cgm']) is None
            not_parameterized &= (self.pf['pop_k_heat_igm']) is None
            self._parameterized = not not_parameterized

        return self._parameterized

    #@property
    #def dndm(self):
    #    if not hasattr(self, '_fcoll'):
    #        self._init_fcoll()
    #
    #    return self._dndm

    @property
    def fcoll(self):
        if not hasattr(self, '_fcoll'):
            self._init_fcoll(return_fcoll=True)
    
        return self._fcoll

    @property
    def dfcolldz(self):
        if not hasattr(self, '_dfcolldz'):
            self._init_fcoll()

        return self._dfcolldz

    def dfcolldt(self, z):
        return self.dfcolldz(z) / self.cosm.dtdz(z)    

    def _set_fcoll(self, Tmin, mu, return_fcoll=False):
        self._fcoll, self._dfcolldz, self._d2fcolldz2 = \
            self.halos.build_1d_splines(Tmin, mu, return_fcoll=return_fcoll)

    @property
    def gf_spline(self):
        if not hasattr(self, '_gf_spline'):
            gf = self.halos.growth_factor
            self._gf_spline = interp1d(self.halos.z, gf, 
                kind='linear', bounds_error=False)
        
        return self._gf_spline
            
    def growth_factor(self, z):
        return self.gf_spline(z)
                
    @property
    def halos(self):
        if not hasattr(self, '_halos'):
            if self.pf['hmf_instance'] is not None:
                self._halos = self.pf['hmf_instance']
            else:
                self._halos = HaloModel(**self.pf)
                #self._halos = HaloMassFunction(**self.pf)
                
        return self._halos
        
    def _init_fcoll(self, return_fcoll=False):
        # Halo stuff
        if self.pf['pop_sfrd'] is not None:
            return

        if self.pf['pop_fcoll'] is None:
            self._set_fcoll(self.pf['pop_Tmin'], self.pf['mu'],
                return_fcoll=return_fcoll)
        else:
            self._fcoll, self._dfcolldz = \
                self.pf['pop_fcoll'], self.pf['pop_dfcolldz']
    
    @property
    def MGR(self):
        """
        Mass growth rate of halos of mass M at redshift z.
    
        ..note:: This is the *DM* mass accretion rate. To obtain the baryonic 
            accretion rate, multiply by Cosmology.fbaryon.
    
        """
        if not hasattr(self, '_MAR'):
            if self.pf['pop_MAR'] is None:
                self._MAR = None
            elif type(self.pf['pop_MAR']) is FunctionType \
                or ismethod(self.pf['pop_MAR']):
                self._MAR = self.pf['pop_MAR']
            elif self.pf['pop_MAR'] == 'pl':
                raise NotImplemented('do this')
            elif self.pf['pop_MAR'] == 'hmf':
                # Would be nice if this were a pointer...
                self._MAR = self.halos.MAR_func
            else:
                self._MAR = read_lit(self.pf['pop_MAR'], 
                    verbose=self.pf['verbose']).MAR
    
        return self._MAR
    
    def MGR_integrated(self, z, source=None):
        """
        The integrated DM accretion rate.
    
        Parameters
        ----------
        z : int, float
            Redshift
        source : str
            Can be a litdata module, e.g., 'mcbride2009'.
    
        Returns
        -------
        Integrated DM mass accretion rate in units of Msun/yr/cMpc**3.
    
        """    
    
        return self.cosm.rho_m_z0 * self.dfcolldt(z) * cm_per_mpc**3 \
                * s_per_yr / g_per_msun
    

        
        
