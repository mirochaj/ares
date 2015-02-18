"""

MetaGalacticBackground.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Feb 16 12:43:06 MST 2015

Description: 

"""

import numpy as np
from ..util import ParameterFile

class MetaGalacticBackground:
    def __init__(self, **kwargs):
        self.pf = ParameterFile(**kwargs)
        
    
    def evolve(self, t, dt):
        pass
    
    def step(self, t, dt):
        pass    
    
    def tau(self):
        pass
    

