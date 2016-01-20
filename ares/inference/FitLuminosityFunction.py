"""

FitGLF.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Oct 23 14:34:01 PDT 2015

Description: 

"""

import numpy as np
from ..util import read_lit
from emcee.utils import sample_ball
from .ModelFit import LogLikelihood
from ..util.PrintInfo import print_fit
from .FitGlobal21cm import FitGlobal21cm
import gc, os, sys, copy, types, time, re
from ..util.ParameterFile import par_info
from ..simulations import Global21cm as simG21
from ..simulations import MultiPhaseMedium as simMPM
from ..analysis.InlineAnalysis import InlineAnalysis
from ..util.SetDefaultParameterValues import _blob_names, _blob_redshifts, \
    SetAllDefaults

try:
    import cPickle as pickle
except:
    import pickle    

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1
    
twopi = 2. * np.pi

defaults = SetAllDefaults()

class loglikelihood(LogLikelihood):

    @property
    def runsim(self):
        if not hasattr(self, '_runsim'):
            self._runsim = True
        return self._runsim
    @runsim.setter
    def runsim(self, value):
        self._runsim = value

    @property
    def redshifts(self):
        return self._redshifts
    @redshifts.setter
    def redshifts(self, value):
        self._redshifts = value

    @property
    def sim_class(self):
        if not hasattr(self, '_sim_class'):
            if 'include_igm' in self.base_kwargs:
                if self.base_kwargs['include_igm']:
                    self._sim_class = simG21
                else:
                    self._sim_class = simMPM
            elif defaults['include_igm']:
                self._sim_class = simG21
            else:
                self._sim_class = simMPM                
                
        return self._sim_class
        
    @property
    def const_term(self):
        if not hasattr(self, '_const_term'):
            self._const_term = -np.log(np.sqrt(twopi)) \
                             -  np.sum(np.log(self.error))
        return self._const_term                     
        
    def __call__(self, pars, blobs=None):
        """
        Compute log-likelihood for model generated via input parameters.
    
        Returns
        -------
        Tuple: (log likelihood, blobs)
    
        """

        kwargs = {}
        for i, par in enumerate(self.parameters):
        
            if self.is_log[i]:
                kwargs[par] = 10**pars[i]
            else:
                kwargs[par] = pars[i]

        # Apply prior on model parameters first (dont need to generate signal)
        lp = self.logprior_P(pars)
        if not np.isfinite(lp):
            return -np.inf, self.blank_blob
    
        # Run a model and retrieve turning points
        kw = self.base_kwargs.copy()
        kw.update(kwargs)
        
        self.checkpoint(**kw)

        sim = self.sim = self.sim_class(**kw)

        if isinstance(sim, simG21):
            medium = sim.medium
        else:
            medium = sim

        # If we're only fitting the LF, no need to run simulation
        if self.runsim:
                        
            try:
                sim.run()                     
            except (ValueError, IndexError):
                # Seems to happen in some weird cases when the 
                # HAM fit fails
                # Also, if Tmin goes crazy big (IndexError in integration)
                
                f = open('%s.fail.%s.pkl' % (self.prefix, str(rank).zfill(3)), 
                    'ab')
                pickle.dump(kwargs, f)
                f.close()
                
                del sim, kw, f
                gc.collect()
                
                return -np.inf, self.blank_blob
        
        if self.logprior_B.pars != []:
            lp += self._compute_blob_prior(sim)

        # emcee will crash if this returns NaN. OK if it's inf though.
        if np.isnan(lp):
            return -np.inf, self.blank_blob

        # Figre out which population is the one with the LF
        for popid, pop in enumerate(medium.field.pops):
            if pop.is_fcoll_model:
                continue
            break

        self.pop = pop

        # Compute the luminosity function, goodness of fit, return
        phi = []
        for i, z in enumerate(self.redshifts):
            p = pop.LuminosityFunction(z=z, x=np.array(self.xdata[i]), 
                mags=True, dc=False)
            phi.extend(p)

        PofD = self.const_term - \
            0.5 * np.sum((np.array(phi) - self.ydata)**2 / self.error**2)

        if np.isnan(PofD):
            return -np.inf, self.blank_blob

        try:
            blobs = sim.blobs
        except:
            blobs = self.blank_blob

        del sim, kw
        gc.collect()
                        
        return lp + PofD, blobs
    
class FitLuminosityFunction(FitGlobal21cm):
    """
    Basically a Global21cm fit except we might not actually press "run" on
    any of the simulations. By default, we don't.
    """
    
    def __init__(self, **kwargs):
        FitGlobal21cm.turning_points = False
        FitGlobal21cm.__init__(self, **kwargs)
            
    @property
    def runsim(self):
        if not hasattr(self, '_runsim'):
            self._runsim = False
        return self._runsim
    
    @runsim.setter
    def runsim(self, value):
        self._runsim = value
        
    @property
    def loglikelihood(self):
        if not hasattr(self, '_loglikelihood'):
            self._loglikelihood = loglikelihood(self.xdata, 
                self.ydata_flat, self.error_flat, 
                self.parameters, self.is_log, self.base_kwargs, self.priors, 
                self.prefix, self.blob_info, self.checkpoint_by_proc)   
            
            self._loglikelihood.runsim = self.runsim
            self._loglikelihood.redshifts = self.redshifts

        return self._loglikelihood

    @property
    def redshifts(self):
        if not hasattr(self, '_redshifts'):
            raise ValueError('Set by hand or include in litdata.')
            
        return self._redshifts
                    
    @redshifts.setter
    def redshifts(self, value):
        # This can be used to override the redshifts in the dataset and only
        # use some subset of them
        if not hasattr(self, '_redshifts'):
            raise NotImplemented('you should have already set the redshifts')
            
        if type(value) in [int, float]:
            value = [value]
            
        tmp1 = copy.deepcopy(self._redshifts)
        tmp2 = []
        for redshift in value:
            if redshift not in tmp1:
                raise ValueError('Redshift %g not in this dataset!')        
            tmp2.append(redshift)

        self._redshifts = tmp2    

    @property
    def data(self):
        if not hasattr(self, '_data'):
            raise AttributeError('Must set data by hand!')
        return self._data    
                
    @data.setter
    def data(self, value):
        if type(value) == str:
            litdata = read_lit(value)
            self._data = litdata.data['lf']
            self._redshifts = litdata.redshifts
            
        else:
            raise NotImplemented('help!')
                                        
    @property
    def xdata_flat(self):
        if not hasattr(self, '_xdata_flat'):
            self._xdata_flat = []; self._ydata_flat = []
            self._error_flat = []; self._redshifts_flat = []
            for i, redshift in enumerate(self.redshifts):
                self._xdata_flat.extend(self.data[redshift]['M'])
                self._ydata_flat.extend(self.data[redshift]['phi'])
                self._error_flat.extend(self.data[redshift]['err'])
                
                zlist = [redshift] * len(self.data[redshift]['M'])
                self._redshifts_flat.extend(zlist)

        return self._xdata_flat
    
    @property
    def ydata_flat(self):
        if not hasattr(self, '_ydata_flat'):
            xdata_flat = self.xdata_flat
    
        return self._ydata_flat      
    
    @property
    def error_flat(self):
        if not hasattr(self, '_error_flat'):
            xdata_flat = self.xdata_flat
    
        return self._error_flat   
    
    @property
    def redshifts_flat(self):
        if not hasattr(self, '_redshifts_flat'):
            xdata_flat = self.xdata_flat
    
        return self._redshifts_flat      
    
    @property
    def xdata(self):
        if not hasattr(self, '_xdata'):
            if hasattr(self, '_data'):
                self._xdata = []; self._ydata = []; self._error = []
                for i, redshift in enumerate(self.redshifts):
                    self._xdata.append(self.data[redshift]['M'])
                    self._ydata.append(self.data[redshift]['phi'])
                    self._error.append(self.data[redshift]['err'])
                    
        return self._xdata
        
    @xdata.setter
    def xdata(self, value):
        self._xdata = value
        
    @property
    def ydata(self):
        if not hasattr(self, '_ydata'):
            if hasattr(self, '_data'):
                xdata = self.xdata
                
        return self._ydata    
    
    @ydata.setter
    def ydata(self, value):
        self._ydata = value
        
    @property
    def error(self):
        if not hasattr(self, '_error'):
            if hasattr(self, '_data'):
                xdata = self.xdata
        return self._error
    
    @error.setter
    def error(self, value):
        self._error = value    
    
    @property
    def guess_override(self):
        if not hasattr(self, '_guess_override_'):
            self._guess_override_ = {}
        
        return self._guess_override_
    
    @guess_override.setter
    def guess_override(self, kwargs):
        if not hasattr(self, '_guess_override_'):
            self._guess_override_ = {}
            
        self._guess_override_.update(kwargs)
                        
    def save_data(self, prefix, clobber=False):
        if rank > 0:
            return
            
        fn = '%s.data.pkl' % prefix
        
        if os.path.exists(fn) and (not clobber):
            print "%s exists! Set clobber=True to overwrite." % fn
            return
                
        f = open(fn, 'wb')
        pickle.dump((self.xdata, self.ydata, self.redshifts, self.error), f)
        f.close()
     
    
