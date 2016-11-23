"""

Animation.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Wed Nov 23 09:32:17 PST 2016

Description: 

"""

import pickle
import numpy as np
import matplotlib.pyplot as pl
from .ModelSet import ModelSet

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1
            
class Animation(object):
    def __init__(self, prefix=None):
        self._prefix = prefix
        
    @property
    def model_set(self):
        if not hasattr(self, '_model_set'):
            if isinstance(self._prefix, ModelSet):
                self._model_set = self._prefix
            elif type(self._prefix) is str:
                self._model_set = ModelSet(self._prefix)
            elif type(self._prefix) in [list, tuple]:
                raise NotImplementedError('help!')
            
        return self._model_set
            
    def Plot1D(self, plane, par=None, pivots=None, prefix='test', plotter=None, 
        ivar=None, take_log=False, un_log=False, multiplier=1., 
        ax=None, fig=1, **kwargs):
        """
        Animate variations of a single parameter.
        
        Parameters
        ----------
        par : str
            Parameter to vary.
        pivots : list, tuple
            
        
        
        ..note:: should implement override for kwargs, like change color of
            line/symbol if some condition violated (e.g., tau_e).
            
            
        """
        
        if par is None:
            assert len(self.model_set.parameters) == 1
            par = self.model_set.parameters[0]
        else:    
            assert par in self.model_set.parameters, \
                "Supplied parameter '%s' is unavailable!" % par
            
        _pars = [par]
        for _p in plane:
            if _p in self.model_set.all_blob_names:
                _pars.append(_p)
                
        data = self.model_set.ExtractData(_pars, ivar=ivar, 
            take_log=take_log, un_log=un_log, multiplier=multiplier)

        N = data[par].shape[0]                
                        
        limits = data[par].min(), data[par].max()
        
        # Re-order the data.
        order = np.argsort(data[par])
        
        data_sorted = {par:data[par][order]}
        for p in plane:
            # How does this work if 1-D blob?
            data_sorted[p] = data[p][order]
    
        if pivots is None:
            pivots = [data_sorted[par][N / 2], limits[0], limits[1], 
                      data_sorted[par][N / 2]]
        
        for element in pivots:
            assert limits[0] <= element <= limits[1], \
                "Pivot point lies outside range of data!"
                                                                                              
        data_assembled = {p:[] for p in _pars}
        i = 0
        for pivot in pivots:
            j = np.argmin(np.abs(pivot - data_sorted[par]))
            
            if j < i: 
                step = -1
            else:
                step = 1
                                    
            for p in _pars:    
                data_assembled[p].extend(list(data_sorted[p][i:j:step]))

            i = 1 * j

        if ax  is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True    

        for i, val in enumerate(data_assembled[par]):
        
            x, y = data_assembled[plane[0]][i], data_assembled[plane[1]][i]          
                                            
            ax.scatter(x, y)          
            
            pl.savefig('%s_%s.png' % (prefix, str(i).zfill(4)))
            ax.clear()
            
        
            
    def add_residue(self):
        pass
        
    def add_marker(self):
        pass
    
    def add_slider(self):
        pass
        
        
"""
How to do this?

Need to start:
- ModelSet object or filename pointing to one.
- Or, series of individual calculations? Implement this one later.

Fancy options:
- sweep out polygon?
- sliders
- leave residue (from past frames)
- leave tracks (i.e., all models up to N) or "trail"


Try: 
- change anspect ratio of GS plots?


"""    
        
        
        
        
        
    

#parvals_lo = np.logspace(-1, 0, 11)
#parvals_hi = np.logspace(0, 1., 11)
#
#base_pars = {'verbose': False, 'progress_bar': False, 'final_redshift': 6.}
#refvals = {'Nlw': 1e4, 'Nion': 4e3, 'fX': 0.2, 'fstar': 0.1}

# Loop over parameters of interest
#frame = 0
#for par in ['Nlw', 'fX', 'Nion', 'fstar']:
#
#    allvals = np.concatenate((parvals_lo[-1::-1], parvals_lo,
#                              parvals_hi, parvals_hi[-1::-1]))
#
#    for val in allvals:
#
#        if frame % size != rank:
#            frame += 1
#            continue
#
#        kwargs = refvals.copy()
#        kwargs.update(base_pars)
#        kwargs[par] = refvals[par] * val
#
#        sim = ares.simulations.Global21cm(**kwargs)
#        sim.run()
#
#        anl = ares.analysis.Global21cm(sim)
#        ax = anl.GlobalSignature(ymax=60, ymin=-200)
#        ax.set_ylim(-180, 60)
#
#        ax.annotate(r'$f_{\alpha} = %.2g$' % (sim.pf['Nlw'] / refvals['Nlw']), 
#            (15, 40), ha='left')
#        ax.annotate(r'$f_{X} = %.2g$' % (sim.pf['fX'] / refvals['fX']), 
#            (55, 40), ha='left')        
#        ax.annotate(r'$f_{\mathrm{ion}} = %.2g$' % (sim.pf['Nion'] / refvals['Nion']),
#            (95, 40), ha='left')    
#        ax.annotate(r'$f_{\ast} = %.2g$' % sim.pf['fstar'],
#            (135, 40), ha='left')        
#
#        pl.draw()
#        pl.savefig('frame_%s.png' % (str(frame).zfill(5)), dpi=120)
#        
#        ax.clear()
#
#        frame += 1
#
#