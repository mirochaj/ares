"""

Collapse.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Feb 16 12:48:16 MST 2015

Description: 

"""

import numpy as np
from ..util.ReadData import _sort_data
from ..util import ProgressBar, ParameterFile

class Collapse:
    def __init__(self, **kwargs):
        self.pf = ParameterFile(**kwargs)
        
    def run(self):
        pass    
        
    def step(self):
        pass    
        
    def update_density(self):
        pass