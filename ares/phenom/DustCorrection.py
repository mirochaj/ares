"""

DustCorrection.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Tue Jan 19 17:55:27 PST 2016

Description: 

"""

import numpy as np
from ..util import ParameterFile

class DustCorrection(object):
    def __init__(self, **kwargs):
        self.pf = ParameterFile(**kwargs)
        
    @property
    def method(self):
        return self.pf['dustcorr_Afun']
        
    def AUV(self, z, mag):
        if self.pf['dustcorr_Afun'] is None:
            return 0.0
        elif self.pf['dustcorr_Afun'].lower() == 'meurer1999':
            return self.MeurerDC(z, mag)
        else:
            raise NotImplemented('sorry!')
            
    def MeurerDC(self, z, mag):        
        return 4.43 + 1.99 * self.Beta(z, mag)
        
    def Beta(self, z, mag):
        if self.pf['dustcorr_Bfun'] == 'constant':
            return self.pf['dustcorr_Bfun_par0']
        else:
            raise NotImplemented('sorry!')
        

        
        
        