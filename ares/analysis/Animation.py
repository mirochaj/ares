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
from ..util.Aesthetics import Labeler
from ares.physics.Constants import nu_0_mhz
from mpl_toolkits.axes_grid import inset_locator

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
        
    def _limits_w_padding(self, limits, take_log=False, un_log=False, 
        padding=0.1):
        mi, ma = limits
                    
        if (mi <= 0) or self.model_set.is_log[0]:
            mi -= padding
        elif (take_log) and (not self.model_set.is_log[0]):
            mi -= padding
        else:
            mi *= (1. - padding)
        
        if (ma >= 0) or self.model_set.is_log[0]:
            ma += padding    
        elif (take_log) and (not self.model_set.is_log[0]):
            mi += padding    
        else:
            ma *= (1. + padding)
        
        return mi, ma        
        
    def Plot1D(self, plane, par=None, pivots=None, prefix='test', plotter=None, 
        ivar=None, take_log=False, un_log=False, multiplier=1., 
        ax=None, fig=1, clear=True, z_to_freq=True, slider_kwargs={}, 
        backdrop=None, backdrop_kwargs={}, squeeze_main=True, **kwargs):
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
        _x = None
        for _p in plane:
            if _p in self.model_set.all_blob_names:
                _pars.append(_p)
            else:
                _x = _p
                
        data = self.model_set.ExtractData(_pars, ivar=ivar, 
            take_log=take_log, un_log=un_log, multiplier=multiplier)

        N = data[par].shape[0]                
                        
        limits = data[par].min(), data[par].max()
        
        # Re-order the data.
        order = np.argsort(data[par])
        
        data_sorted = {par:data[par][order]}
        for p in plane:
            # How does this work if 1-D blob?
            if p in _pars:
                data_sorted[p] = data[p][order]
            else:
                ii, jj, nd, dims = self.model_set.blob_info(p)
                data_sorted[p] = self.model_set.blob_ivars[ii][jj]

        if pivots is None:
            pivots = [data_sorted[par][N / 2], limits[0], limits[1], 
                      data_sorted[par][N / 2 - 1]]
        
        for element in pivots:
            assert limits[0] <= element <= limits[1], \
                "Pivot point lies outside range of data!"
                                                                                              
        data_assembled = {p:[] for p in _pars}
        i = np.argmin(np.abs(pivots[0] - data_sorted[par]))
        for k, pivot in enumerate(pivots):
            if k == 0:
                continue
                
            j = np.argmin(np.abs(pivot - data_sorted[par]))
            
            if j < i: 
                step = -1
            else:
                step = 1
                                    
            for p in _pars: 
                data_assembled[p].extend(list(data_sorted[p][i:j:step]))

            i = 1 * j

        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            fig.subplots_adjust(right=0.7)
            ax = fig.add_subplot(111)
            
            sax = self.add_slider(ax, limits=limits, take_log=take_log,
                un_log=un_log, **slider_kwargs)
                
        else:
            gotax = True    
            
        labeler = Labeler(_pars, **self.model_set.base_kwargs)            
        
        for i, val in enumerate(data_assembled[par]):
        
            if _x is None:
                x = data_assembled[plane[0]][i]
            else:
                x = _x

            y = data_assembled[plane[1]][i]

            if type(x) in [int, float]:
                ax.scatter(x, y, **kwargs)
            else:
                if ('z' in _pars) and z_to_freq:
                    ax.plot(nu_0_mhz / (1.+ x), y, **kwargs)
                else:
                    ax.plot(x, y, **kwargs)
            
            # Need to be careful with axes limits not changing...
            if ('z' in _pars) and z_to_freq:
                xarr = nu_0_mhz / (1. + data[plane[0]])
            else:
                xarr = data[plane[0]]
            
            if _x is None:    
                _xmi, _xma = xarr.min(), xarr.max()
                xmi, xma = self._limits_w_padding((_xmi, _xma))
                ax.set_xlim(xmi, xma)
                
            _ymi, _yma = data[plane[1]].min(), data[plane[1]].max()
            ymi, yma = self._limits_w_padding((_ymi, _yma))
            ax.set_ylim(ymi, yma)
            
            sax.plot([val]*2, [0, 1], **kwargs)
            sax = self._reset_slider(sax, limits, take_log, un_log, 
                **slider_kwargs)
            
            if ('z' in _pars) and z_to_freq:
                ax.set_xlabel(labeler.label('nu'))
            else:
                ax.set_xlabel(labeler.label(plane[0]))
                
            ax.set_ylabel(labeler.label(plane[1]))
        
            pl.savefig('%s_%s.png' % (prefix, str(i).zfill(4)))
            
            #yield ax
            
            if clear:
                ax.clear()
                sax.clear()
                
                if backdrop is not None:
                    ax.plot(backdrop[0], backdrop[1], **backdrop_kwargs)
                
        if not clear:
            return ax
        else:
            pl.close()    
            
    def add_residue(self):
        pass
        
    def add_marker(self):
        pass
    
    def _reset_slider(self, ax, limits, take_log=False, un_log=False, **kwargs):
        ax.set_yticks([])
        ax.set_yticklabels([])
        
        if take_log:
            lim = np.log10(limits)
        elif (un_log) and (self.model_set.is_log[0]):
            lim = 10**limits
        else:
            lim = limits
        
        lo, hi = self._limits_w_padding(limits, take_log=take_log, un_log=un_log)
                
        ax.set_xlim(lo, hi)
        ax.tick_params(axis='x', labelsize=8, length=3, width=1, which='major')
        
        if 'label' in kwargs:
            ax.set_xlabel(kwargs['label'], fontsize=10)
                
        return ax
        
    def add_slider(self, ax, limits, label=None,
        take_log=False, un_log=False, rect=[0.75, 0.7, 0.2, 0.05], **kwargs):
        """
        Add inset 'slider' thing.
        """

        inset = pl.axes(rect)
        inset = self._reset_slider(inset, limits, take_log, un_log, **kwargs) 
        pl.draw()
    
        return inset
        
        
class AnimationSet(object):
    def __init__(self, prefix):
        self._prefix = prefix

    @property
    def animations(self):
        if not hasattr(self, '_animations'):
            self._animations = []
            for prefix in self._prefix:
                self._animations.append(Animation(prefix))
        
        return self._animations        
                
    def PlotIndependentND(self, pars):

        assert type(pars) in [list, tuple]

        N = len(pars)

        for k in range(3):    
            self.add_slider(ax, limits=limits, take_log=take_log,
                un_log=un_log, rect=[0.75, 0.55-0.15*k, 0.2, 0.05],
                **slider_kwargs)


    

#