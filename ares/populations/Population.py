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

class Population(object):
    def __init__(self, grid=None, **kwargs):
        
        # why is this necessary?
        if 'problem_type' in kwargs:
            del kwargs['problem_type']

        self.pf = ParameterFile(**kwargs)
        self.grid = grid

        self.zform = self.pf['pop_zform']
        self.zdead = self.pf['pop_zdead']
        
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
        