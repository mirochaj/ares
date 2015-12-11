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
from ..util.PrintInfo import print_fit
from .FitGlobal21cm import FitGlobal21cm
import gc, os, sys, copy, types, time, re
from ..util.ParameterFile import par_info
from ..simulations import Global21cm as simG21
from .ModelFit import LogPrior, update_blob_names
from ..simulations import MultiPhaseMedium as simMPM
from ..analysis.InlineAnalysis import InlineAnalysis
from ..populations.Galaxy import param_redshift, GalaxyPopulation
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

defaults = SetAllDefaults()

class loglikelihood:
    def __init__(self, xdata, ydata, error, redshifts, parameters, is_log,
        base_kwargs, priors={}, prefix=None, blob_names=None, 
        blob_redshifts=None, run_21cm=False):
        """
        Computes log-likelihood at given step in MCMC chain.

        Parameters
        ----------

        """

        self.parameters = parameters # important that they are in order?
        self.is_log = is_log
        self.run_21cm = run_21cm

        self.base_kwargs = base_kwargs

        # Not flat
        self.xdata = xdata
        self.redshifts = redshifts
        
        # Flat
        self.ydata = np.array(ydata)
        self.error = np.array(error)

        self.prefix = prefix   

        self.blob_names = update_blob_names(blob_names)
        self.blob_redshifts = blob_redshifts
        
        tmp = (self.blob_names, self.blob_redshifts)
        self.base_kwargs['inline_analysis'] = tmp
        
        #self._prep_binfo()
        
        # Sort through priors        
        priors_P = {}   # parameters
        priors_B = {}   # blobs

        p_pars = []
        b_pars = []
        for key in priors:
            # Priors on model parameters
            if len(priors[key]) == 3:
                p_pars.append(key)
                priors_P[key] = priors[key]

            elif len(priors[key]) == 4:
                b_pars.append(key)
                priors_B[key] = priors[key]
            
            # Should set up a proper Warnings module for this sort of thing
            if key == 'tau_e' and len(priors[key]) != 4:
                if rank == 0:
                    print 'Must supply redshift for prior on %s!' % key
                MPI.COMM_WORLD.Abort()

        self.logprior_P = LogPrior(priors_P, self.parameters, self.is_log)
        self.logprior_B = LogPrior(priors_B, b_pars)
        
    @property
    def blank_blob(self):
        if not hasattr(self, '_blank_blob'):
            
            if self.blob_names is None:
                self._blank_blob = {}
                return {}

            tup = tuple(np.ones(len(self.blob_names)) * np.inf)
            self._blank_blob = []
            for i in range(len(self.blob_redshifts)):
                self._blank_blob.append(tup)
    
        return np.array(self._blank_blob)
        
    @property
    def sim_class(self):
        if not hasattr(self, '_sim_class'):
            if (not self.run_21cm):
                self._sim_class = GalaxyPopulation
            elif 'include_igm' in self.base_kwargs:
                if self.base_kwargs['include_igm']:
                    self._sim_class = simG21
                else:
                    self._sim_class = simMPM
            elif defaults['include_igm']:
                self._sim_class = simG21
            else:
                self._sim_class = simMPM                
                
        return self._sim_class
        
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
    
        sim = self.sim = self.sim_class(**kw)
        
        if isinstance(sim, simG21):
            medium = sim.medium
        elif isinstance(sim, GalaxyPopulation):
            medium = None       
        else:
            medium = sim
                
        # If we're only fitting the LF, no need to run simulation
        if self.run_21cm:
                        
            try:
                sim.run()                      
                sim.run_inline_analysis()
            except ValueError:
                # Seems to happen in some weird cases when the 
                # HAM fit fails
                
                f = open('%s.fail.%s.pkl' % (self.prefix, str(rank).zfill(3)), 
                    'ab')
                pickle.dump(kwargs, f)
                f.close()
                
                del sim, kw, f
                gc.collect()
                
                return -np.inf, self.blank_blob
                                                                                                      
            #tfn = '%s.timing_%s.pkl' % (self.prefix, str(rank).zfill(4))
            #with open(tfn, 'ab') as f:
            #    pickle.dump((t2 - t1, kwargs), f)

        # Timestep weird (happens when xi ~ 1)
        #except SystemExit:
        #    pass

        ## most likely: no (or too few) turning pts
        #except ValueError:                     
        #    # Write to "fail" file
        #    if not self.burn:
        #        f = open('%s.fail.%s.pkl' % (self.prefix, str(rank).zfill(3)), 'ab')
        #        pickle.dump(kwargs, f)
        #        f.close()
        #
        #    del sim, kw, f
        #    gc.collect()
        #
        #    return -np.inf, self.blank_blob

        # Apply priors to blobs
        blob_vals = []
        for key in self.logprior_B.priors:
            
            if not hasattr(sim, 'blobs'):
                break

            z = self.logprior_B.priors[key][3]
    
            i = self.blob_names.index(key)
            j = self.blob_redshifts.index(z)

            val = sim.blobs[j,i]

            blob_vals.append(val)

        if blob_vals:
            lp -= self.logprior_B(blob_vals)
            # emcee will crash if this returns NaN
            if np.isnan(lp):
                return -np.inf, self.blank_blob

        if hasattr(sim, 'blobs'):
            blobs = sim.blobs
        else:
            blobs = self.blank_blob

        # Figre out which population is the one with the LF
        if medium is not None:
            for popid, pop in enumerate(medium.field.pops):
                if not pop.is_ham_model:
                    continue
                break
        else:
            pop = sim
            
        phi = []
        for i, z in enumerate(self.redshifts):
            p = pop.LuminosityFunction(M=np.array(self.xdata[i]), z=z)
            phi.extend(p)
            
        logL = lp - 0.5 * (np.sum((np.array(phi) - self.ydata)**2 \
            / 2. / self.error**2 + np.log(2. * np.pi * self.error**2)))

        #if blobs.shape != self.blank_blob.shape:
        #    raise ValueError('Shape mismatch between requested blobs and actual blobs!')    
    
        del sim, kw
        gc.collect()
    
        return logL, blobs
    
class FitLuminosityFunction(FitGlobal21cm):
    """
    Basically a Global21cm fit except we might not actually press "run" on
    any of the simulations. By default, we don't.
    """
    
    def __init__(self, **kwargs):
        FitGlobal21cm.turning_points = False
        FitGlobal21cm.__init__(self, **kwargs)
        
        self.blob_names = update_blob_names(self.blob_names, **self.base_kwargs)
    
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
                self.ydata_flat, self.error_flat, self.redshifts, 
                self.parameters, self.is_log, self.base_kwargs, self.priors, 
                self.prefix, self.blob_names, self.blob_redshifts, self.runsim)    

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
            
    @property
    def guesses(self):
        if not hasattr(self, '_guesses'):
            raise ValueError('Must set guesses by hand!')
        return self._guesses            
    
    @guesses.setter
    def guesses(self, value):
        """
        Can supply string, e.g., 'bouwens2015', to initialize walkers within
        confidence regions of literature sources (for Schecter parameters).
        """
        
        if rank > 0:
            return
        
        assert type(value) == str, 'Must supply bouwens2015 at the moment.'
        
        data = read_lit(value)
        fits = data.fits['lf']
        z = data.redshifts
        
        jitter = []
        guesses = []
        for i, par in enumerate(self.parameters):
            prefix, popid, popz = par_info(par)
            
            # Will never be log (?) for now, anyways
            if re.search('pop_lf', prefix):
                name = prefix.replace('pop_lf_', '')
                
                if self.is_log[i]:
                    err = fits['err'][name][z.index(popz)]
                    val = fits['pars'][name][z.index(popz)]
                    
                    new_jit = np.log10(abs(val - err) / val)
                    new_guess = np.log10(fits['pars'][name][z.index(popz)])
                else:    
                    new_jit = fits['err'][name][z.index(popz)]
                    new_guess = fits['pars'][name][z.index(popz)]
                    
            # Will be in same units as priors
            elif par in self.guess_override:
                new_jit = self.jitter[i]
                new_guess = self.guess_override[par]
            # Never log
            elif prefix in defaults:
                new_jit = self.jitter[i]
                new_guess = defaults[prefix]
            else:
                raise NotImplemented('help')
                
            jitter.append(new_jit)
            guesses.append(new_guess)
        
        self.jitter = jitter
        guesses = sample_ball(guesses, self.jitter, size=self.nwalkers)
            
        # Fix parameters whose values lie outside prior space
        self._guesses = self._fix_guesses(guesses)
                        
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
     
    
