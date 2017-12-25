"""

Population.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu May 28 13:59:41 MDT 2015

Description: 

"""

from ..physics import Cosmology
from ..util import ParameterFile
from ..physics.Constants import E_LyA, E_LL
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

_multi_pop_error_msg = "Parameters for more than one population detected! "
_multi_pop_error_msg += "Population objects are by definition for single populations."
_multi_pop_error_msg += 'This population: '

class Population(object):
    def __init__(self, grid=None, **kwargs):

        # why is this necessary?
        if 'problem_type' in kwargs:
            del kwargs['problem_type']

        self.pf = ParameterFile(**kwargs)
        
        assert self.pf.Npops == 1, _multi_pop_error_msg + str(self.id_num)
        
        self.grid = grid

        self.zform = self.pf['pop_zform']
        self.zdead = self.pf['pop_zdead']

    @property
    def id_num(self):
        if not hasattr(self, '_id_num'):
            self._id_num = None
        return self._id_num

    @id_num.setter
    def id_num(self, value):
        self._id_num = int(value)    

    @property
    def cosm(self):
        if not hasattr(self, '_cosm'):    
            if self.grid is None:
                self._cosm = Cosmology(
                    omega_m_0=self.pf['omega_m_0'], 
                    omega_l_0=self.pf['omega_l_0'], 
                    omega_b_0=self.pf['omega_b_0'],  
                    hubble_0=self.pf['hubble_0'],  
                    helium_by_number=self.pf['helium_by_number'],
                    cmb_temp_0=self.pf['cmb_temp_0'],
                    approx_highz=self.pf['approx_highz'],
                    sigma_8=self.pf['sigma_8'],
                    primordial_index=self.pf['primordial_index'])
            else:
                self._cosm = grid.cosm

        return self._cosm

    @property
    def zone(self):
        if not hasattr(self, '_zone'):
            if self.affects_cgm and (not self.affects_igm):
                self._zone = 'cgm'
            elif self.affects_igm and (not self.affects_cgm):
                self._zone = 'igm'
            else:
                self._zone = 'both'
                # Can't remember why I do this...
                #raise ValueError("Populations should only affect one zone!")
                
        return self._zone    
        
    @property
    def affects_cgm(self):
        if not hasattr(self, '_affects_cgm'):
            self._affects_cgm = self.is_ion_src_cgm 
        return self._affects_cgm
    
    @property
    def affects_igm(self):
        if not hasattr(self, '_affects_igm'):
            self._affects_igm = self.is_ion_src_igm or self.is_heat_src_igm
        return self._affects_igm    
    
    @property
    def is_ion_src_cgm(self):
        if not hasattr(self, '_is_ion_src_cgm'):
            if not self.pf['radiative_transfer']:
                self._is_ion_src_cgm = False
            elif self.pf['pop_ion_src_cgm']:
                self._is_ion_src_cgm = True
            else:
                self._is_ion_src_cgm = False

        return self._is_ion_src_cgm
    
    @property
    def is_ion_src_igm(self):
        if not hasattr(self, '_is_ion_src_igm'):
            if not self.pf['radiative_transfer']:
                self._is_ion_src_igm = False
            elif self.pf['pop_ion_src_igm']:
                self._is_ion_src_igm = True
            else:
                self._is_ion_src_igm = False

        return self._is_ion_src_igm
    
    @property
    def is_heat_src_igm(self):
        if not hasattr(self, '_is_heat_src_igm'):
            if not self.pf['radiative_transfer']:
                self._is_heat_src_igm = False
            elif self.pf['pop_heat_src_igm']:
                self._is_heat_src_igm = True
            else:
                self._is_heat_src_igm = False

        return self._is_heat_src_igm
    
    @property
    def is_lya_src(self):
        if not hasattr(self, '_is_lya_src'):
            if not self.pf['radiative_transfer']:
                self._is_lya_src = False
            elif self.pf['pop_sed_model']:
                self._is_lya_src = \
                    (self.pf['pop_Emin'] <= 10.2 <= self.pf['pop_Emax']) \
                    and self.pf['pop_lya_src']
            else:
                return self.pf['pop_lya_src']
    
        return self._is_lya_src
    
    @property
    def is_lw_src(self):
        if not hasattr(self, '_is_lw_src'):
            if not self.pf['radiative_transfer']:
                self._is_lw_src = False
            elif not self.pf['pop_lw_src']:
                self._is_lw_src = False
            elif self.pf['pop_sed_model']:
                self._is_lw_src = \
                    (self.pf['pop_Emin'] <= 11.2 <= self.pf['pop_Emax']) and \
                    (self.pf['pop_Emin'] <= E_LL <= self.pf['pop_Emax'])
            else:                
                raise NotImplementedError('help')
    
        return self._is_lw_src    

    @property
    def is_uv_src(self):
        if not hasattr(self, '_is_uv_src'):
            if not self.pf['radiative_transfer']:
                self._is_uv_src = False
            elif self.pf['pop_sed_model']:
                self._is_uv_src = \
                    (self.pf['pop_Emax'] > E_LL) \
                    and self.pf['pop_ion_src_cgm']
            else:
                self._is_uv_src = self.pf['pop_ion_src_cgm']        
    
        return self._is_uv_src    
    
    @property
    def is_xray_src(self):
        if not hasattr(self, '_is_xray_src'):
            if not self.pf['radiative_transfer']:
                self._is_xray_src = False
            elif self.pf['pop_sed_model']:
                self._is_xray_src = \
                    (E_LL <= self.pf['pop_Emin']) \
                    and self.pf['pop_heat_src_igm']
            else:
                self._is_xray_src = self.pf['pop_heat_src_igm']        
    
        return self._is_xray_src
        
    @property
    def is_emissivity_separable(self):
        """
        Are the frequency and redshift-dependent components independent?
        """
        return True
    
    @property
    def is_emissivity_scalable(self):
        """
        Can we just determine a luminosity density by scaling the SFRD?
    
        The answer will be "no" for any population with halo-mass-dependent
        values for photon yields (per SFR), escape fractions, or spectra.
        """
        
        if not hasattr(self, '_is_emissivity_scalable'):
            self._is_emissivity_scalable = True
    
            if self.pf.Npqs == 0:
                return self._is_emissivity_scalable
    
            for par in self.pf.pqs:
    
                # Exceptions. Ideally, exotic_heating_func wouldn't make it
                # to the population parameter files...
                if (par == 'pop_fstar') or (not par.startswith('pop_')):
                #if par in ['pop_fstar', 'exotic_heating_func', 'spin_temperature_floor']:
                    continue
                    
                # Could just skip parameters that start with pop_    
    
                if isinstance(self.pf[par], basestring):
                    self._is_emissivity_scalable = False
                    break
    
                for i in range(self.pf.Npqs):
                    pn = '{0!s}[{1}]'.format(par,i)
                    if pn not in self.pf:
                        continue
    
                    if isinstance(self.pf[pn], basestring):
                        self._is_emissivity_scalable = False
                        break
    
        return self._is_emissivity_scalable
    
        
