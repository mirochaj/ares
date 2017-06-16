"""

Galaxy.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Thu Jun  8 13:01:27 PDT 2017

Description: 

"""

import numpy as np
from ..util import ParameterFile
from .SynthesisModel import SynthesisModel

class Galaxy(object):
    def __init__(self, **kwargs):
        self.pf = ParameterFile(**kwargs)

    @property
    def src(self):
        if not hasattr(self, '_src'):
            self._src = SynthesisModel(**self.pf)

        return self._src

    def generate_history(self):
        pass


