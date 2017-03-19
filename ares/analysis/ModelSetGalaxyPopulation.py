"""

ModelSetGalaxyPopulation.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sat Oct 24 10:42:45 PDT 2015

Description: 

"""

import re
import ares
import numpy as np
from ..util import read_lit
import matplotlib.pyplot as pl
from .ModelSet import ModelSet
from .GalaxyPopulation import GalaxyPopulation
from ..phenom.DustCorrection import DustCorrection
from ..util.SetDefaultParameterValues import SetAllDefaults

ln10 = np.log(10.)

phi_of_M = lambda M, pstar, Mstar, alpha: 0.4 * ln10 * pstar \
    * (10**(0.4 * (Mstar - M)*(1. + alpha))) \
    * np.exp(-10**(0.4 * (Mstar - M)))

class ModelSetGalaxyPopulation(ModelSet):
    """
    Basically a ModelSet instance with routines specific to the high-z
    galaxy luminosity function.
    """
    
    @property
    def obsdata(self):
        if not hasattr(self, '_obsdata'):
            self._obsdata = GalaxyPopulation()
        return self._obsdata
    
    @property
    def dc(self):
        if not hasattr(self, '_dc'):
            self._dc = DustCorrection(**self.base_kwargs)
        return self._dc
    
    def _Recovered_Function(self, z, quantity, ax=None, fig=1, 
        compare_to=None, percentile=0.685, 
        Mlim=(-24, -10), samples=1, take_log=False, un_log=False,
        multiplier=1, skip=0, stop=None, use_best=False, best='median',
        show_all=False, **kwargs):
        """
        Basically a wrapper around ModelSet.ReconstructedFunction but a lil
        better.
        """

        if quantity in ['lf', 'smf']:
            return self.ReconstructedFunction(quantity, ivar=[z, None], ax=ax, 
                percentile=percentile, apply_dc=quantity=='lf', 
                use_best=use_best, best=best, skip=skip, stop=stop, **kwargs)
        elif type(quantity) in [tuple, list]:
            return self.ReconstructedRelation(quantity, ivar=[z, None], ax=ax,
                percentile=percentile, **kwargs)
        else:
            raise NotImplementedError('help me!')

    def RecoveredModel(self, z, mp=None, fig=1, quantity='lf', compare_to=None, 
        percentile=0.685,
        samples=1, skip=0, stop=None, use_best=False, best='median',
        show_all=False, data_kwargs={}, **kwargs):
        """
        Plot the luminosity function used to train the SFE.
        """
        
        if type(z) in [int, float]:
            z = [z]
    
        # First, plot observational data [optional]
        if mp is None:
            if compare_to is not None:
                if len(z) > 1:
                    mp = self.obsdata.MultiPlot(z, fig=fig, 
                        quantity=quantity, sources=compare_to, **data_kwargs)
                else:
                    mp = self.obsdata.Plot(z[0], fig=fig, 
                        quantity=quantity, sources=compare_to, **data_kwargs)
                
                self.obsdata.add_master_legend(mp, scatterpoints=1, numpoints=1, 
                    ncol=4, fontsize=14, handletextpad=0.5, columnspacing=0.5)
                        
            else:
                mp = None   
        
        ##
        # Now, loop over redshift and plot reconstructed model
        ##
        for i, redshift in enumerate(z):
            if len(z) == 1:
                ax = mp
            else:
                k = self.obsdata.redshifts_in_mp[i]
                ax = mp.grid[k]

            # Need to apply DC if quantity == 'lf'
            ax = self._Recovered_Function(redshift, quantity=quantity, ax=ax,
                percentile=percentile, use_best=use_best, best=best,
                samples=samples, skip=skip, stop=stop, 
                show_all=show_all, **kwargs)    
        
        pl.draw()

        if len(z) == 1:
            return ax
        else:
            return mp
    
       
            
        
        