"""

Population.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu May 28 13:59:41 MDT 2015

Description: 

"""

from ..physics import Cosmology
from ..util import ParameterFile
from ..physics.Constants import E_LyA

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

        