"""

Parameterized.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Thu Jul 14 14:00:10 PDT 2016

Description: 

"""

import numpy as np
from types import FunctionType
from .Population import Population
from ..phenom.ParameterizedQuantity import ParameterizedQuantity
from ..util.ParameterFile import ParameterFile, par_info, get_pq_pars
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

parametric_options = ['pop_Ja', 'pop_ion_rate_cgm', 'pop_ion_rate_igm',
    'pop_heat_rate']

class ParametricPopulation(Population):

    def __getattr__(self, name):
        if (name[0] == '_'):
            raise AttributeError('This will get caught. Don\'t worry!')
            
        # This is the name of the thing as it appears in the parameter file.
        full_name = 'pop_' + name
        
        # Now, possibly make an attribute
        if not hasattr(self, name):
            try:
                is_pq = self.pf[full_name][0:2] == 'pq'
            except (IndexError, TypeError):
                is_pq = False
                
            if type(self.pf[full_name]) in [float, np.float64]:
                result = lambda z: self.pf[full_name]
            elif type(self.pf[full_name]) is FunctionType:
                result = self.pf[full_name]
            elif is_pq:
                pars = get_pq_pars(self.pf[full_name], self.pf)            
                result = ParameterizedQuantity(**pars)
            elif isinstance(self.pf[full_name], basestring):
                x, y = np.loadtxt(self.pf[full_name], unpack=True)
                result = interp1d(x, y, kind=self.pf['interp_hist'])
            else:
                raise NotImplementedError('Problem with: {!s}'.format(name))
                
            self.__setattr__(name, result)
            
        return getattr(self, name)    

    def LymanAlphaFlux(self, z):
        return self.Ja(z=z)
                
    def IonizationRateCGM(self, z):
        return self.ion_rate_cgm(z=z)
        
    def IonizationRateIGM(self, z):
        return self.ion_rate_igm(z=z)    
            
    def HeatingRate(self, z):
        return self.heat_rate(z=z)
    
    
        
        
    
    
        
