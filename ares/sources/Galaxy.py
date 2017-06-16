"""

Galaxy.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Thu Jun  8 13:01:27 PDT 2017

Description: 

"""

import numpy as np
from ..util import ParameterFile
from SyntesisModel import SynthesisModel

class Galaxy(object):
    def __init__(self, **kwargs):
        self.pf = ParameterFile(**kwargs)

    def generate_history(self, t, wave, band=None, new_inst=False):
        """
        Given a star-formation history and potentially metallicity, create
        fully time-dependent SED etc. for this source population.
        """
        
        k = np.argmin(np.abs(wave - self.wavelengths))
    
    
    
