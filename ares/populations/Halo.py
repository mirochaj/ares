"""

Halo.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu May 28 16:22:44 MDT 2015

Description: 

"""

import numpy as np
from .Population import Population
from scipy.integrate import cumtrapz
from ..physics import HaloMassFunction
from ..util.PrintInfo import print_pop
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

    @property
    def info(self):
        if not self.parameterized:
            try:
                print_pop(self)
            except AttributeError:
                pass

    @property
    def dndm(self):
        if not hasattr(self, '_fcoll'):
            self._init_fcoll()
    
        return self._dndm

    @property
    def fcoll(self):
        if not hasattr(self, '_fcoll'):
            self._init_fcoll()
    
        return self._fcoll

    @property
    def dfcolldz(self):
        if not hasattr(self, '_dfcolldz'):
            self._init_fcoll()

        return self._dfcolldz

    def dfcolldt(self, z):
        return self.dfcolldz(z) / self.cosm.dtdz(z)    

    def _set_fcoll(self, Tmin, mu):
        self._fcoll, self._dfcolldz, self._d2fcolldz2 = \
            self.halos.build_1d_splines(Tmin, mu)

    @property
    def halos(self):
        if not hasattr(self, '_halos'):
            if self.pf['hmf_instance'] is not None:
                self._halos = self.pf['hmf_instance']
            else:
                self._halos = HaloMassFunction(**self.pf)
                
        return self._halos

    def update_Mmin(self, z, Mmin):
        """
        Given the redshift and minimum mass, create a new _dfcolldz function.
        """
        
        if not hasattr(self, '_counter'):
            self._counter = 0
            self._update_Mmin = True
            
        if not self._update_Mmin:
            return
            
        # Data containers    
        if not hasattr(self, '_z_list'):
            self._z_list = []
            #self._Mmin_list = []            
            self._fcoll_list = []
        
        # Brute-force
        if self._counter < 5:
            Tmin = self.halos.VirialTemperature(Mmin, z, mu=0.6)
            self._set_fcoll(Tmin, mu=0.6)
            
            self._z_list.append(z)
            self._fcoll_list.append(self.fcoll(z))
                            
        else:
            
            if self.pf['pop_Mmax'] is not None:
                if Mmin >= self.pf['pop_Mmax']:
                    self._dfcolldz = lambda z: 0.0
                    self._counter += 1
                    self._update_Mmin = False
                    return
                    
            elif self.pf['pop_Tmax'] is not None:
                Mmax = self.halos.VirialMass(self.pf['pop_Tmax'], z, mu=0.6)
                if Mmin >= Mmax:
                       self._dfcolldz = lambda z: 0.0
                       self._counter += 1
                       self._update_Mmin = False
                       return
                        
            # Do something cool!
            
            # Step 1: Calculate fcoll(z, Mmin)
            
            fcoll = self.halos.fcoll_2d(z, np.log10(Mmin))
            
            # Account for halos crossing threshold?
            
            
            # Step 2: Append to lists
            
            #self._Mmin_list.append(Mmin)
            self._z_list.append(z)
            self._fcoll_list.append(fcoll)
            
            # Step 2.5: Take derivative!
            
            
            #_ztab, _dfcolldz_tab = \
            #    central_difference(self._z_list, self._fcoll_list)
            
            _ztab, _dfcolldz_tab = \
                forward_difference(self._z_list, self._fcoll_list)
            
            
            
            #import matplotlib.pyplot as pl
            #
            #print z, Mmin / 1e5, self._fcoll_list[-3:], _dfcolldz_tab[-3:]
            #
            ##pl.scatter(_ztab[-3:], np.abs(_dfcolldz_tab[-3:]))
            #pl.scatter(self._z_list[-3:], self._fcoll_list[-3:])
            #pl.ylim(1e-10, 1e-2)
            #pl.yscale('log')
            
            
            #raw_input('<enter>')
            
            # Step 3: Create extrapolants
            z_hi, z_mid, z_lo = _ztab[-3:]
            dfcdz_hi, dfcdz_mid, dfcdz_lo = _dfcolldz_tab[-3:]
            
            dfcdz_p = (dfcdz_lo - dfcdz_hi) / (z_lo - z_hi)
            dfcdz_a = (dfcdz_mid - dfcdz_hi) / (z_mid - z_hi)
            dfcdz_b = (dfcdz_lo - dfcdz_mid) / (z_lo - z_mid)
                                                  
            # This should be positive!
            if self._counter % 40 == 0:
                delattr(self, '_dfcolldz')
                self._dfcolldz = lambda zz: abs(dfcdz_p * (zz - z_lo) + dfcdz_lo)            
                
        self._counter += 1
        
    def _init_fcoll(self):
        # Halo stuff
        if self.pf['pop_sfrd'] is not None:
            return

        if self.pf['pop_fcoll'] is None:
            self._set_fcoll(self.pf['pop_Tmin'], self.pf['mu'])
        else:
            self._fcoll, self._dfcolldz = \
                self.pf['pop_fcoll'], self.pf['pop_dfcolldz']
    
    def iMAR(self, z, source=None):
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
    

        
        