"""

ModelSelection.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Mon Feb 15 18:53:21 PST 2016

Description: 

"""

import numpy as np
import matplotlib.pyplot as pl
from .ModelSet import ModelSet

class ModelSelection(object):
    def __init__(self, msets):
        """
        Initialize object for comparing results of different fits.
        
        ..note:: Not really model selection by any rigorous definition at 
            this point.
            
        """
        self.msets = msets
        
        assert type(self.msets) is list
        assert len(self.msets) > 1
        
        self._ms = {}
        
    @property
    def Nsets(self):
        return len(self.msets)    
        
    def ms(self, i):

        if i in self._ms:
            return self._ms[i]
            
        self._ms[i] = ModelSet(self.msets[i])
        return self._ms[i]
        
    def PosteriorPDF(self, pars, ax=None, fig=1, **kwargs):
        """
        PDFs from two different calculations.
        """
        
        for i in range(self.Nsets):
            ax = self.ms(i).PosteriorPDF(pars, ax=ax, fig=fig, **kwargs)

        return ax
        
    def TrianglePlot(self, pars, ax=None, fig=1, **kwargs):
        """
        PDFs from two different calculations.
        """
    
        for i in range(self.Nsets):
            ax = self.ms(i).TrianglePlot(pars, ax=ax, fig=fig, **kwargs)
    
        return ax  
    
    def ReconstructedFunction(self, name, ivar=None, ax=None, fig=1, **kwargs):
        """
        PDFs from two different calculations.
        """
    
        for i in range(self.Nsets):
            ax = self.ms(i).ReconstructedFunction(name, ivar, ax=ax, fig=fig, 
                **kwargs)
    
        return ax      
        
    def DeriveBlob(self, expr, varmap, save=True, name=None, clobber=False):
        results = []
        for i in range(self.Nsets):
            result = \
                self.ms(i).DeriveBlob(expr, varmap, save, name, clobber)
            results.append(result)
            
        return results
            
                
