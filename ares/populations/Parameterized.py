"""

Parameterized.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Thu Jul 14 14:00:10 PDT 2016

Description: 

"""

from types import FunctionType
from .Population import Population
from ..phenom.HaloProperty import ParameterizedHaloProperty
from ..util.ParameterFile import ParameterFile, par_info, get_php_pars

class ParametricPopulation(Population):
        
    @property
    def is_lya_src(self):
        return True if self.pf['pop_Ja'] is not None else False 
    
    @property
    def is_uv_src(self):
        return True if self.pf['pop_ion_rate'] is not None else False
    
    @property
    def is_xray_src(self):
        return True if self.pf['pop_heat_rate'] is not None else False           
    
    @property
    def Ja(self):
        if not hasattr(self, '_Ja'):
            if self.pf['pop_Ja'][0:3] == 'php':
                pars = get_php_pars(self.pf['pop_Ja'], self.pf)
                self._Ja = ParameterizedHaloProperty(**pars)
            elif type(self.pf['pop_Ja']) is FunctionType:
                self._Ja = self.pf['pop_Ja']
            else:
                raise ValueError('Unrecognized data type for pop_Ja!')    
        
        return self._Ja
        
    def LymanAlphaFlux(self, z):
        return self.Ja(z, None)
        
    @property
    def Gamma(self):
        if not hasattr(self, '_k_ion'):
            if self.pf['pop_ion_rate'][0:3] == 'php':
                pars = get_php_pars(self.pf['pop_ion_rate'], self.pf)
                self._k_ion = ParameterizedHaloProperty(**pars)
            elif type(self.pf['pop_ion_rate']) is FunctionType:
                self._k_ion = self.pf['pop_ion_rate']
            else:
                raise ValueError('Unrecognized data type for pop_ion_rate!')    
    
        return self._k_ion    
        
    def IonizationRateCGM(self, z):
        return self.Gamma(z, None)
        
    @property
    def epsilon_X(self):
        if not hasattr(self, '_k_heat'):
            if self.pf['pop_heat_rate'][0:3] == 'php':
                pars = get_php_pars(self.pf['pop_heat_rate'], self.pf)
                self._k_heat = ParameterizedHaloProperty(**pars)
            elif type(self.pf['pop_heat_rate']) is FunctionType:
                self._k_heat = self.pf['pop_heat_rate']
            else:
                raise ValueError('Unrecognized data type for pop_heat_rate!')    
    
        return self._k_heat    
    
    def HeatingRate(self, z):
        return self.epsilon_X(z, None)
        
        
    
    
        