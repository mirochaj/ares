"""

Population.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu May 28 13:59:41 MDT 2015

Description: 

"""

from ..physics import Cosmology
from ..util import ParameterFile

class Population(object):
    def __init__(self, grid=None, **kwargs):
        
        self.pf = ParameterFile(**kwargs)
        
        self.grid = grid
        
        if grid is None:
            self.cosm = Cosmology(
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
            self.cosm = grid.cosm

    def SpaceDensity(self, z):
        pass
    
    def LuminosityDensity(self, z):
        pass    
            
        
        