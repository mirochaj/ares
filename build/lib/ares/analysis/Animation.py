"""

Animation.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Wed Nov 23 09:32:17 PST 2016

Description: 

"""
import numpy as np
import matplotlib.pyplot as pl
from .ModelSet import ModelSet
from ..physics import Hydrogen
from ..util.Aesthetics import Labeler
from ..physics.Constants import nu_0_mhz
from .MultiPhaseMedium import add_redshift_axis
from mpl_toolkits.axes_grid1 import inset_locator
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1

class Animation(object): # pragma: no cover
    def __init__(self, prefix=None):
        self._prefix = prefix

    @property
    def model_set(self):
        if not hasattr(self, '_model_set'):
            if isinstance(self._prefix, ModelSet):
                self._model_set = self._prefix
            elif isinstance(self._prefix, basestring):
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
                
    def build_tracks(self, plane, _pars, pivots=None, ivar=None, take_log=False, 
        un_log=False, multiplier=1, origin=None):
        """
        Construct array of models in the order in which we'll plot them.
        """
        
        data = self.model_set.ExtractData(_pars, ivar=ivar, 
            take_log=take_log, un_log=un_log, multiplier=multiplier)

        par = _pars[0]
        
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
             
        if origin is None:
            start = end = data_sorted[par][N // 2]
        else:
            start = end = origin

        # By default, scan to lower values, then all the way up, then return
        # to start point
        if pivots is None:
            pivots = [round(v, 4) for v in [start, limits[0], limits[1], end]]
                
        for element in pivots:
            assert limits[0] <= element <= limits[1], \
                "Pivot point lies outside range of data!"
                                                                                                        
        data_assembled = {p:[] for p in _pars}
        i = np.argmin(np.abs(pivots[0] - data_sorted[par]))
        for k, pivot in enumerate(pivots):
            if k == 0:
                continue

            j = np.argmin(np.abs(pivot - data_sorted[par]))
            
            #if par == 'pop_logN{1}':
            #    print i, j, k
            
            if j < i: 
                step = -1
            else:
                step = 1
                
            for p in _pars:
                data_assembled[p].extend(list(data_sorted[p][i:j:step]))

            i = 1 * j
        
        # Add start point!
        data_assembled[p].append(start)
        
        data_assembled[par] = np.array(data_assembled[par])
                
        self.data = {'raw': data, 'sorted': data_sorted, 
            'assembled': data_assembled, 'limits':limits}    
            
    def prepare_axis(self, ax=None, fig=1, squeeze_main=True, 
        take_log=False, un_log=False, **kwargs):
        
        if ax is None:
            fig = pl.figure(fig)
            fig.subplots_adjust(right=0.7)
            ax = fig.add_subplot(111)

        sax = self.add_slider(ax, limits=self.data['limits'], 
            take_log=take_log, un_log=un_log, **kwargs)

        return ax, sax

    def Plot1D(self, plane, par=None, pivots=None, prefix='test', twin_ax=None,
        ivar=None, take_log=False, un_log=False, multiplier=1., 
        ax=None, sax=None, fig=1, clear=True, z_to_freq=True, 
        slider_kwargs={}, backdrop=None, backdrop_kwargs={}, squeeze_main=True, 
        close=False, xlim=None, ylim=None, xticks=None, yticks=None, 
        z_ax=True, origin=None, sticks=None, slims=None, inits=None,
        **kwargs):
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
                "Supplied parameter '{!s}' is unavailable!".format(par)

        _pars = [par]
        _x = None
        for _p in plane:
            if _p in self.model_set.all_blob_names:
                _pars.append(_p)
            else:
                _x = _p
                
        if type(sticks) is dict:
            sticks = sticks[par]
        if type(slims) is dict:
            slims = slims[par]            
                                        
        # This sets up all the data
        self.build_tracks(plane, _pars, pivots=pivots, ivar=ivar, 
            take_log=[take_log, False, False], un_log=[un_log, False, False], 
            multiplier=multiplier, origin=origin)

        if ax is None:
            ax, sax = self.prepare_axis(ax, fig, **slider_kwargs)
            
        if z_ax and 'z' in _pars:
            twin_ax = add_redshift_axis(ax, twin_ax)

        labeler = Labeler(_pars, **self.model_set.base_kwargs)
        
        # What do we need to make plots?
        # data_assembled, plane, ax, sax, take_log etc.

        data = self.data['raw']
        limits = self.data['limits']
        data_assembled = self.data['assembled']
        
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
                    
            if inits is not None:
                if z_to_freq:
                    ax.plot(nu_0_mhz / (1. + inits['z']), inits['dTb'], 
                        **kwargs)
                else:
                    ax.plot(inits['z'], inits['dTb'], **kwargs)
            
            # Need to be careful with axes limits not changing...
            if ('z' in _pars) and z_to_freq:
                xarr = nu_0_mhz / (1. + data[plane[0]])
            else:
                xarr = data[plane[0]]
            
            if xlim is not None:
                xmi, xma = xlim
            elif _x is None:    
                _xmi, _xma = xarr.min(), xarr.max()
                xmi, xma = self._limits_w_padding((_xmi, _xma))
            
            ax.set_xlim(xmi, xma)                
            if twin_ax is not None:
                twin_ax.set_xlim(xmi, xma)    
                
            if ylim is not None:
                ax.set_ylim(ylim)
            else:
                _ymi, _yma = data[plane[1]].min(), data[plane[1]].max()
                ymi, yma = self._limits_w_padding((_ymi, _yma))
                ax.set_ylim(ymi, yma)
                            
            sax.plot([val]*2, [0, 1], **kwargs)
            sax = self._reset_slider(sax, limits, take_log, un_log, 
                sticks=sticks, slims=slims, **slider_kwargs)      
            
            if ('z' in _pars) and z_to_freq:
                ax.set_xlabel(labeler.label('nu'))
            else:
                ax.set_xlabel(labeler.label(plane[0]))
                
            ax.set_ylabel(labeler.label(plane[1]))
            
            if xticks is not None:
                ax.set_xticks(xticks, minor=True)
            if yticks is not None:
                ax.set_yticks(yticks, minor=True)    
                
            if ('z' in _pars) and z_to_freq:
                if z_ax:
                    twin_ax = add_redshift_axis(ax, twin_ax)    
                         
            pl.draw()
        
            pl.savefig('{0!s}_{1!s}.png'.format(prefix, str(i).zfill(4)))
            
            if clear:
                ax.clear()
                sax.clear()
                if twin_ax is not None:
                    twin_ax.clear()  

        return ax, twin_ax

    def add_residue(self):
        pass

    def add_marker(self):
        pass

    def _reset_slider(self, ax, limits, take_log=False, un_log=False, 
        sticks=None, slims=None, **kwargs):
        ax.set_yticks([])
        ax.set_yticklabels([])

        if slims is None:
            lo, hi = self._limits_w_padding(limits, take_log=take_log, un_log=un_log)
        else:
            lo, hi = slims
            
        ax.set_xlim(lo, hi)
        ax.tick_params(axis='x', labelsize=10, length=3, width=1, which='major')

        if 'label' in kwargs:
            ax.set_xlabel(kwargs['label'], fontsize=14)
        
        if sticks is not None:
            ax.set_xticks(sticks)    

        return ax
        
    def add_slider(self, ax, limits, take_log=False, un_log=False, 
        rect=[0.75, 0.7, 0.2, 0.05], **kwargs):
        """
        Add inset 'slider' thing.
        """
                
        inset = pl.axes(rect)
        inset = self._reset_slider(inset, limits, take_log, un_log, **kwargs) 
        pl.draw()
    
        return inset
        
        
class AnimationSet(object): # pragma: no cover
    def __init__(self, prefix):
        self._prefix = prefix

    @property
    def animations(self):
        if not hasattr(self, '_animations'):
            self._animations = []
            for prefix in self._prefix:
                self._animations.append(Animation(prefix))
        
        return self._animations
        
    @property
    def parameters(self):
        if not hasattr(self, '_parameters'):
            self._parameters = []
            for animation in self.animations:
                if len(animation.model_set.parameters) == 1:
                    self._parameters.append(animation.model_set.parameters[0])
                else:
                    self._parameters.append('unknown')
                    
        return self._parameters
        
    @property
    def labels(self):
        if not hasattr(self, '_labels'):
            self._labels = []
            for animation in self.animations:
                if len(animation.model_set.parameters) == 1:
                    self._labels.append(animation.model_set.parameters[0])
                else:
                    self._labels.append('unknown')

        return self._labels                  
    
    @labels.setter
    def labels(self, value):
        if type(value) is dict:
            self._labels = []
            for par in self.parameters:
                self._labels.append(value[par])
        elif type(value) in [list, tuple]:
            assert len(value) == len(self.parameters)
            self._labels = value
        
        
    @property
    def origin(self):
        if not hasattr(self, '_origin'):
            self._origin = [None] * len(self.animations)
        return self._origin    
        
    @origin.setter
    def origin(self, value):
        if type(value) is dict:
            self._origin = []
            for par in self.parameters:
                self._origin.append(value[par])
        elif type(value) in [list, tuple]:
            assert len(value) == len(self.parameters)
            self._origin = value
    
    @labels.setter
    def labels(self, value):
        if type(value) is dict:
            self._labels = []
            for par in self.parameters:
                self._labels.append(value[par])
        elif type(value) in [list, tuple]:
            assert len(value) == len(self.parameters)
            self._labels = value    
            
    @property
    def take_log(self):
        if not hasattr(self, '_take_log'):
            self._take_log = [False] * len(self.parameters)
        return self._take_log
        
    @take_log.setter
    def take_log(self, value):
        if type(value) is dict:
            self._take_log = []
            for par in self.parameters:
                self._take_log.append(value[par])
        elif type(value) in [list, tuple]:
            assert len(value) == len(self.parameters)
            self._take_log = value
        
    @property
    def un_log(self):
        if not hasattr(self, '_un_log'):
            self._un_log = [False] * len(self.parameters)
        return self._un_log
    
    @un_log.setter
    def un_log(self, value):
        if type(value) is dict:
            self._un_log = []
            for par in self.parameters:
                self._un_log.append(value[par])
        elif type(value) in [list, tuple]:
            assert len(value) == len(self.parameters)    
            self._un_log = [False] * len(self.parameters)
            
    @property
    def inits(self):
        if not hasattr(self, '_inits'):
            hydr = Hydrogen()
            inits = hydr.inits    
        
            anim = self.animations[0]
            gr, i, nd, dims = anim.model_set.blob_info('z')
            _z = anim.model_set.blob_ivars[gr][i]
    
            z = np.arange(max(_z), 1100, 1)
    
            dTb = hydr.dTb_no_astrophysics(z)
    
            self._inits = {'z': z, 'dTb': dTb}
    
        return self._inits        
        
    def Plot1D(self, plane, pars=None, ax=None, fig=1, prefix='test', 
        xlim=None, ylim=None, xticks=None, yticks=None, sticks=None,
        slims=None, top_sax=0.75, include_inits=True, **kwargs):
        """
        Basically run a series of Plot1D.
        """
        
        if pars is None:
            pars = self.parameters
            
        assert type(pars) in [list, tuple]

        N = len(pars)
        
        if sticks is None:
            sticks = {par:None for par in pars}
        if slims is None:
            slims = {par:None for par in pars}
        
        ## 
        # First: setup axes
        ##

        ax = None
        sax = []
        for k in range(N):
            
            assert len(self.animations[k].model_set.parameters) == 1
            par = self.animations[k].model_set.parameters[0]
            
            _pars = [par]
            _x = None
            for _p in plane:
                if _p in self.animations[k].model_set.all_blob_names:
                    _pars.append(_p)
                else:
                    _x = _p
            
            self.animations[k].build_tracks(plane, _pars, 
                take_log=self.take_log[k], un_log=False, multiplier=1,
                origin=self.origin[k])
            
            ax, _sax = self.animations[k].prepare_axis(ax=ax, fig=fig, 
                squeeze_main=True, rect=[0.75, top_sax-0.15*k, 0.2, 0.05],
                label=self.labels[k])
                
            sax.append(_sax)
            
        
        ##
        # Now do all the plotting
        ##
        twin_ax = None
        
        for k in range(N):
            par = self.animations[k].model_set.parameters[0]
            _pars = [par]
            _x = None
            for _p in plane:
                if _p in self.animations[k].model_set.all_blob_names:
                    _pars.append(_p)
                else:
                    _x = _p

            kw = {'label': self.labels[k]}
            
            # Add slider bar for all currently static parameters
            # (i.e., grab default value)
            for l in range(N):
                if l == k:
                    continue
                    
                _p = self.parameters[l]
                limits = self.animations[l].data['limits']
                sax[l].plot([self.origin[l]]*2, [0, 1], **kwargs)
                self.animations[l]._reset_slider(sax[l], limits, 
                    take_log=self.take_log[l], un_log=self.un_log[l],
                    label=self.labels[l], sticks=sticks[_p], slims=slims[_p])
                        
            # Plot variable parameter
            ax, twin_ax = \
                self.animations[k].Plot1D(plane, par, ax=ax, sax=sax[k],
                take_log=self.take_log[k], un_log=self.un_log[k],
                prefix='{0!s}.{1!s}'.format(prefix, par), close=False,
                slider_kwargs=kw, xlim=xlim, ylim=ylim, origin=self.origin[k],
                xticks=xticks, yticks=yticks, twin_ax=twin_ax, 
                sticks=sticks, slims=slims, inits=self.inits, **kwargs)
                
            
            
            

        
