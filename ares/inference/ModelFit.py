"""

MCMC.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Oct 23 19:02:38 PDT 2015

Description: 

"""

import numpy as np
from ..util.Stats import get_nu
from ..util.PrintInfo import print_fit
from ..physics.Constants import nu_0_mhz
from ..util.AbundanceMatching import HAM
from ..util.ParameterFile import par_info
import gc, os, sys, copy, types, time, re
from ..analysis import Global21cm as anlG21
from ..util.Stats import Gauss1D, GaussND, rebin
from ..analysis.TurningPoints import TurningPoints
from ..analysis.InlineAnalysis import InlineAnalysis
from ..util.BlobFactory import BlobFactory
from ..util.SetDefaultParameterValues import _blob_names, _blob_redshifts
from ..util.ReadData import flatten_chain, flatten_logL, flatten_blobs, \
    read_pickled_chain

try:
    import cPickle as pickle
except:
    import pickle    

try:
    import emcee
    from emcee.utils import sample_ball
except ImportError:
    pass

emcee_mpipool = False
try:
    from mpi_pool import MPIPool
except ImportError:
    try:
        from emcee.utils import MPIPool
        emcee_mpipool = True    
    except ImportError:
        pass
     
try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1
    
twopi = np.sqrt(2 * np.pi)
    
guesses_shape_err = "If you supply guesses as 2-D array, it must have" 
guesses_shape_err += " shape (nwalkers, nparameters)!"

jitter_shape_err = "If you supply jitter as an array, it must have"
jitter_shape_err += " shape (nparameters)"

def uninformative_lin(x, mi, ma):
    if (mi <= x <= ma):
        return 1.0 / (ma - mi)
    else:
        return 0.0

def uninformative_log(x, mi, ma):
    if (mi <= x <= ma):
        return 1.0 / ((ma - mi) * x)
    else:
        return 0.0

def gaussian_prior(x, mu, sigma):
    return np.exp(-0.5 * (x - mu)**2 / sigma**2) / twopi / sigma

def_kwargs = {'verbose': False, 'progress_bar': False}

def _str_to_val(p, par, pvals, pars):
    """
    Convert string to parameter value.
    
    Parameters
    ----------
    p : str
        Name of parameter that the prior for this paramemeter is linked to.
    par : str
        Name of parameter who's prior is linked.
    pars : list
        List of values for each parameter on this step.
        
    Returns
    -------
    Numerical value corresponding to this linker-linkee relationship.    
    
    """
    
    # Look for populations
    m = re.search(r"\{([0-9])\}", p)

    # Single-pop model? I guess.
    if m is None:
        raise NotImplemented('This should never happen.')

    # Population ID number
    num = int(m.group(1))

    # Pop ID including curly braces
    prefix = p.split(m.group(0))[0]

    return pvals[pars.index('%s{%i}' % (prefix, num))]
    
class LogPrior:
    def __init__(self, priors, parameters, is_log=None):
        self.pars = parameters  # just names *in order*
        self.priors = priors
        
        if is_log is None:
            self.is_log = [False] * len(parameters)
        else:
            self.is_log = is_log

        if priors:
            self.prior_len = [len(self.priors[par]) for par in self.pars]

    def __call__(self, pars):
        """
        Compute prior of given model.
        """

        if not self.priors:
            return -np.inf

        logL = 0.0
        for i, par in enumerate(self.pars):
            val = pars[i]

            ptype = self.priors[self.pars[i]][0]
            p1, p2 = self.priors[self.pars[i]][1:]
            
            # Figure out if this prior is linked to others
            if type(p1) is str:
                p1 = _str_to_val(p1, par, pars, self.pars)
            if type(p2) is str:
                p2 = _str_to_val(p2, par, pars, self.pars)

            # Uninformative priors
            if ptype == 'uniform':
                if self.is_log[i]:
                    logL_i = uninformative_log(val, p1, p2)
                else:
                    logL_i = np.log(uninformative_lin(val, p1, p2))
            # Gaussian priors
            elif ptype == 'gaussian':
                logL_i = np.log(gaussian_prior(val, p1, p2))
            else:
                raise ValueError('Unrecognized prior type: %s' % ptype)

            logL -= logL_i

        return logL
        
# For backward compatibility
logprior = LogPrior        
        
class LogLikelihood:
    def __init__(self, steps, parameters, is_log, mu, errors,
        base_kwargs, nwalkers, priors={}, errmap=None, errunits=None, 
        prefix=None, 
        burn=False, blob_names=None, blob_redshifts=None):
        """
        Computes log-likelihood at given step in MCMC chain.

        Parameters
        ----------

        """

        self.parameters = parameters # important that they are in order?
        self.is_log = is_log

        self.base_kwargs = base_kwargs
        self.nwalkers = nwalkers

        self.burn = burn
        self.prefix = prefix   
        self.fit_signal = fit_signal
        self.fit_turning_points = fit_turning_points     

        self.blob_names = blob_names
        self.blob_redshifts = blob_redshifts
        
        # Sort through priors        
        priors_P = {}   # parameters
        priors_B = {}   # blobs

        p_pars = []
        b_pars = []
        for key in priors:
            # Priors on model parameters
            if key in self.parameters:
                p_pars.append(key)
                priors_P[key] = priors[key]
            else:
                b_pars.append(key)
                priors_B[key] = priors[key]
            
        self.logprior_P = logprior(priors_P, self.parameters)
        self.logprior_B = logprior(priors_B, b_pars)

        self.mu = mu
        self.errors = errors

        self.is_cov = False        
        if len(self.errors.shape) > 1:
            self.is_cov = True
            self.Dcov = np.linalg.det(self.errors)
            self.icov = np.linalg.inv(self.errors)
            
        self.errmap = errmap
        self.errunits = errunits
        
    def compute_blob_prior(self, pars):
        lp = 0.0
        blob_vals = []
        for key in self.logprior_B.priors:

            if not hasattr(sim, 'blobs'):
                break
            
            i = self.blob_names.index(key) 
            j = self.blob_redshifts.index(z)

            val = sim.blobs[j,i]
            
            blob_vals.append(val)    

        if blob_vals:
            lp -= self.logprior_B(blob_vals)

        return lp

    @property
    def blank_blob(self):
        if not hasattr(self, '_blank_blob'):
    
            self._blank_blob = []
            for i, group in enumerate(self.blob_names):
                if self.blob_ivars[i] is None:
                    self._blank_blob.append([np.inf] * len(group))
                else:
                    if self.blob_nd[i] == 0:
                        self._blank_blob.append([np.inf] * len(group))
                    else:
                        arr = np.ones([len(group), self.blob_ivars[i].size])
                        self._blank_blob.append(arr * np.inf)
    
        return self._blank_blob
       
    def __call__(self, pars, blobs=None):
        """
        Compute log-likelihood for model generated via input parameters.

        Returns
        -------
        Tuple: (log likelihood, blobs)

        """
        
        # Apply prior on model parameters first (dont need to generate signal)
        lp = self.logprior_P(pars)
        if not np.isfinite(lp):
            return -np.inf, self.blank_blob

        # Run a model and retrieve turning points
        kw = self.base_kwargs.copy()
        kw.update(kwargs)

        try:
            sim = simG21(**kw)
            sim.run()
            
            sim.run_inline_analysis()
            
            tps = sim.turning_points
                    
        # Timestep weird (happens when xi ~ 1)
        except SystemExit:
            
            sim.run_inline_analysis()
            tps = sim.turning_points
                 
        # most likely: no (or too few) turning pts
        except ValueError:                     
            # Write to "fail" file
            if not self.burn:
                f = open('%s.fail.%s.pkl' % (self.prefix, str(rank).zfill(3)), 'ab')
                pickle.dump(kwargs, f)
                f.close()
                            
            del sim, kw, f
            gc.collect()
        
            return -np.inf, self.blank_blob
                
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

        if (not self.fit_turning_points) and (not self.fit_signal):
            del sim, kw
            gc.collect()
            return lp, blobs

        # Compute the likelihood if we've made it this far
        
        if self.fit_signal:
            
            xarr = np.interp(self.signal_z, sim.history['z'][-1::-1],
                sim.history['dTb'][-1::-1])
                
        elif self.fit_turning_points: 
            # Fit turning points    
            xarr = []
            
            # Convert frequencies to redshift, temperatures to K
            for element in self.errmap:
                tp, i = element            
            
                # Models without turning point B, C, or D get thrown out.
                if tp not in tps:
                    del sim, kw
                    gc.collect()
            
                    return -np.inf, self.blank_blob
            
                if i == 0 and self.errunits[0] == 'MHz':
                    xarr.append(nu_0_mhz / (1. + tps[tp][i]))
                else:
                    xarr.append(tps[tp][i])
                       
            # Values of current model that correspond to mu vector
            xarr = np.array(xarr)
                    
        if np.any(np.isnan(xarr)):
            return -np.inf, self.blank_blob
        
        # Compute log-likelihood, including prior knowledge
        if self.is_cov:
            a = (xarr - self.mu).T
            b = np.dot(self.icov, xarr - self.mu)
            
            logL = lp - 0.5 * np.dot(a, b)

        else:
            logL = lp \
                - np.sum((xarr - self.mu)**2 / 2. / self.errors**2)
                        
        if blobs.shape != self.blank_blob.shape:
            raise ValueError('Shape mismatch between requested blobs and actual blobs!')    
            
        del sim, kw
        gc.collect()
        
        return logL, blobs

class ModelFit(BlobFactory):
    def __init__(self, **kwargs):
        """
        Initialize a wrapper class for MCMC simulations.
        
        Mostly just handles setup, file I/O, parallelization.

        Optional Keyword Arguments
        --------------------------
        Anything you want based to each ares.simulations.Global21cm call.
        
        """

        self.base_kwargs = def_kwargs.copy()
        self.base_kwargs.update(kwargs)            
        
    @property
    def pf(self):
        if not hasattr(self, '_pf'):    
            self._pf = self.base_kwargs
        return self._pf
    
    @pf.setter
    def pf(self, value):
        self._pf = value
        
    @property
    def blob_info(self):
        if not hasattr(self, '_blob_info'):
            self._blob_info = \
                {'blob_names': self.blob_names, 
                 'blob_ivars': self.blob_ivars,
                 'blob_funcs': self.blob_funcs,
                 'blob_nd': self.blob_nd,
                 'blob_dims': self.blob_dims}
        return self._blob_info
                                                                                  
    @property
    def loglikelihood(self):
        if not hasattr(self, '_loglikelihood'):
            raise AttributeError("Must set loglikelihood by hand!")
    
        return self._loglikelihood
            
    @loglikelihood.setter
    def loglikelihood(self, value):
        """
        Supply log-likelihood function.
        """        
        
        self._loglikelihood = value
            
    @property
    def error_independent(self):
        if not hasattr(self, '_err_indep'):
            self._err_indep = self.error.ndim == 1
        return self._err_indep
            
    @property 
    def xdata(self):
        if not hasattr(self, '_xdata'):
            raise AttributeError("Must set xdata by hand!")
        return self._xdata
        
    @xdata.setter
    def xdata(self, value):
        self._xdata = value
        
    @property
    def ydata(self):
        if not hasattr(self, '_ydata'):
            raise AttributeError("Must set ydata by hand!")
        return self._ydata
        
    @xdata.setter
    def ydata(self, value):
        self._ydata = value    
    
    @property
    def error(self):
        if not hasattr(self, '_error'):    
            raise AttributeError("Must set error by hand!")
        return self._error

    @error.setter
    def error(self, value):
        """
        Can be 1-D or 2-D.
        """
        self._error = np.array(value)

    @property
    def priors(self):
        if not hasattr(self, '_priors'):
            raise ValueError('Must set priors by hand!')

        return self._priors

    @priors.setter
    def priors(self, value):
        self._priors = value
        
    @property
    def nwalkers(self):
        if not hasattr(self, '_nw'):
            self._nw = self.Nd * 2
            
            if rank == 0:
                print "Defaulting to nwalkers=2*Nd=%i." % self._nw
            
        return self._nw
        
    @nwalkers.setter
    def nwalkers(self, value):
        self._nw = value
        
    @property
    def Nd(self):
        if not hasattr(self, '_Nd'):
            self._Nd = len(self.parameters)
        return self._Nd
        
    @property
    def guesses(self):
        """
        Generate initial position vectors for all walkers.
        """
        
        if rank > 0:
            return
        
        # Set using priors
        if not hasattr(self, '_guesses') and hasattr(self, 'priors'):
            
            if rank == 0:
                self._guesses = []
                for i in range(self.nwalkers):
                    
                    p0 = []
                    to_fix = []
                    for j, par in enumerate(self.parameters):
                
                        if par in self.priors:
                            
                            dist, lo, hi = self.priors[par]
                            
                            # Fix if tied to other parameter
                            if (type(lo) is str) or (type(hi) is str):                            
                                to_fix.append(par)
                                p0.append(None)
                                continue
                                
                            if dist == 'uniform':
                                val = np.random.rand() * (hi - lo) + lo
                            else:
                                val = np.random.normal(lo, scale=hi)
                        else:
                            raise ValueError('No prior for %s' % par)
                
                        # Save
                        p0.append(val)
                
                    # If some priors are linked, correct for that
                    for par in to_fix:
                
                        dist, lo, hi = self.priors[par]
                
                        if type(lo) is str:
                            lo = p0[self.parameters.index(lo)]
                        else:    
                            hi = p0[self.parameters.index(hi)]
                
                        if dist == 'uniform':
                            val = np.random.rand() * (hi - lo) + lo
                        else:
                            val = np.random.normal(lo, scale=hi)
                        
                        k = self.parameters.index(par)
                        p0[k] = val
                    
                    self._guesses.append(p0)
                
                self._guesses = np.array(self._guesses)
                    
        return self._guesses

    @guesses.setter
    def guesses(self, value):
        """
        Initial guesses for walkers. 
        
        .. note :: You can either supply a 1-D array, representing best guess
            for each parameter AND set the ``jitter`` attribute, which is a
            fractional offset in each dimension about this best guess point. 
            OR you can supply
            
        """
        
        if rank > 0:
            return
        
        guesses_tmp = np.array(value)
        
        if guesses_tmp.ndim == 1:
            self._guesses = sample_ball(guesses_tmp, self.jitter, 
                size=self.nwalkers)
        elif guesses_tmp.ndim == 2:
            assert (guesses_tmp.shape == (self.nwalkers, len(self.parameters))), \
                guesses_shape_err
            
            self._guesses = guesses_tmp
        else:
            raise ValueError('Dunno about this shape')
            
    def _fix_guesses(self, pos):
        
        if rank > 0:
            return
        
        guesses = pos.copy()
        
        # Fix parameters whose values lie outside prior space
        for i, par in enumerate(self.parameters):
            if par not in self.priors:
                continue
                
            if self.priors[par][0] != 'uniform':
                continue
            
            mi, ma = self.priors[par][1:]
            
            ok_lo = guesses[:,i] >= mi
            ok_hi = guesses[:,i] <= ma
            
            if np.all(ok_lo) and np.all(ok_hi):
                continue
                
            # Draw from uniform distribution for failed cases
            
            not_ok_lo = np.logical_not(ok_lo)
            not_ok_hi = np.logical_not(ok_hi)
            not_ok = np.logical_or(not_ok_hi, not_ok_lo)
            
            bad_mask = np.argwhere(not_ok)
            
            for j in bad_mask:
                print "Fixing guess for walker %i parameter %s" % (j[0], par)
                guesses[j[0],i] = np.random.uniform(mi, ma)
                
        return guesses
        
    @property 
    def jitter(self):
        if not hasattr(self, '_jitter'):
            if not hasattr(self, '_jitter'):    
                raise AttributeError("Must set jitter by hand!")
        return self._jitter
            
    @jitter.setter
    def jitter(self, value):
        
        if type(value) in [int, float]:
            self._jitter = np.ones(len(self.parameters)) * value
        else:
            assert (len(value) == len(self.parameters)), jitter_shape_error 
                
            self._jitter = np.array(value)
            
    @property
    def parameters(self):
        if not hasattr(self, '_parameters'):
            if not hasattr(self, '_parameters'):    
                raise AttributeError("Must set parameters by hand!")
        return self._parameters
        
    @parameters.setter
    def parameters(self, value):
        self._parameters = value
    
    @property
    def is_log(self):
        if not hasattr(self, '_is_log'):
            self._is_log = [False] * self.Nd
        return self._is_log
          
    @is_log.setter         
    def is_log(self, value):
        if type(value) is bool:
            self._is_log = [value] * self.Nd
        else:
            self._is_log = value
            
    def prep_output_files(self, restart, clobber):
        if restart:
            pos = self._prep_from_restart()
        else:
            pos = None
            self._prep_from_scratch(clobber)    
    
        return pos
    
    def _prep_from_restart(self):

        prefix = self.prefix
        
        f = open('%s.pinfo.pkl' % prefix, 'rb')
        pars, is_log = pickle.load(f)
        f.close()
                                            
        if pars != self.parameters:
            if size > 1:
                if rank == 0:
                    print 'parameters from file dont match those supplied!'
                MPI.COMM_WORLD.Abort()
            raise ValueError('parameters from file dont match those supplied!')
        if is_log != self.is_log:
            if size > 1:
                if rank == 0:
                    print 'is_log from file dont match those supplied!'
                MPI.COMM_WORLD.Abort()
            raise ValueError('is_log from file dont match those supplied!')
                    
        f = open('%s.setup.pkl' % prefix, 'rb')
        base_kwargs = pickle.load(f)
        f.close()  
        
        if base_kwargs != self.base_kwargs:
            if size > 1:
                if rank == 0:
                    print 'base_kwargs from file dont match those supplied!'
                MPI.COMM_WORLD.Abort()
            raise ValueError('base_kwargs from file dont match those supplied!')   
                    
        # Start from last step in pre-restart calculation
        chain = read_pickled_chain('%s.chain.pkl' % prefix)
        
        pos = chain[-self.nwalkers:,:]
        
        return pos

    def _prep_from_scratch(self, clobber):
        
        prefix = self.prefix
        
        if clobber:
            # Delete only the files made by this routine. Don't want to risk
            # deleting other files the user may have created with similar
            # naming convention!
            
            for suffix in ['chain', 'logL', 'facc', 'pinfo', 'setup']:
                os.system('rm -f %s.%s.pkl' % (prefix, suffix))
            
            os.system('rm -f %s.fail*.pkl' % prefix)
            os.system('rm -f %s.blob*.pkl' % prefix)
                    
        # Each processor gets its own fail file
        for i in range(size):
            f = open('%s.fail.%s.pkl' % (prefix, str(i).zfill(3)), 'wb')
            f.close()  
        
        # Main output: MCMC chains (flattened)
        f = open('%s.chain.pkl' % prefix, 'wb')
        f.close()
        
        # Main output: log-likelihood
        f = open('%s.logL.pkl' % prefix, 'wb')
        f.close()
        
        # Store acceptance fraction
        f = open('%s.facc.pkl' % prefix, 'wb')
        f.close()
        
        # File for blobs themselves
        if self.blob_names is not None:
            
            for i, group in enumerate(self.blob_names):
                for blob in group:
                    fntup = (prefix, self.blob_nd[i], blob)
                    f = open('%s.blob_%id.%s.pkl' % fntup, 'wb')
                    f.close()
        
        # Parameter names and list saying whether they are log10 or not
        f = open('%s.pinfo.pkl' % prefix, 'wb')
        pickle.dump((self.parameters, self.is_log), f)
        f.close()
        
        # Constant parameters being passed to ares.simulations.Global21cm
        f = open('%s.setup.pkl' % prefix, 'wb')
        tmp = self.base_kwargs.copy()
        to_axe = []
        for key in tmp:
            if re.search(key, 'tau_table'):
                to_axe.append(key)
        for key in to_axe:
            del tmp[key] # this might be big, get rid of it
        pickle.dump(tmp, f)
        del tmp
        f.close()

    def run(self, prefix, steps=1e2, burn=0, clobber=False, restart=False, 
        save_freq=500):
        """
        Run MCMC.

        Parameters
        ----------
        prefix : str
            Prefix for all output files.
        steps : int
            Number of steps to take.
        burn : int
            Number of steps to burn.
        save_freq : int
            Number of steps to take before writing data to disk.
        clobber : bool  
            Overwrite pre-existing files of the same prefix if one exists?
        restart : bool
            Append to pre-existing files of the same prefix if one exists?
            
        """
        
        self.prefix = prefix

        if os.path.exists('%s.chain.pkl' % prefix) and (not clobber):
            if not restart:
                msg = '%s exists! Remove manually, set clobber=True,' % prefix
                msg += ' or set restart=True to append.' 
                raise IOError(msg)

        if not os.path.exists('%s.chain.pkl' % prefix) and restart:
            msg = "This can't be a restart, %s*.pkl not found." % prefix
            raise IOError(msg)
            
        # Initialize Pool
        if size > 1:
            self.pool = MPIPool()
            
            if not emcee_mpipool:
                self.pool.start()
            
            # Non-root processors wait for instructions until job is done,
            # at which point, they don't need to do anything below here.
            if not self.pool.is_master():
                
                if emcee_mpipool:
                    self.pool.wait()
                    
                sys.exit(0)

        else:
            self.pool = None

        # Initialize sampler
        self.sampler = emcee.EnsembleSampler(self.nwalkers,
            self.Nd, self.loglikelihood, pool=self.pool)
                
        pos = self.prep_output_files(restart, clobber)        
                
        # Burn in, prep output files     
        if (burn > 0) and (not restart):
            
            if rank == 0:
                print "Starting burn-in: %s" % (time.ctime())
            
            t1 = time.time()
            pos, prob, state, blobs = \
                self.sampler.run_mcmc(self.guesses, burn)
            self.sampler.reset()
            t2 = time.time()

            if rank == 0:
                print "Burn-in complete in %.3g seconds." % (t2 - t1)

            # Find maximum likelihood point
            mlpt = pos[np.argmax(prob)]

            pos = sample_ball(mlpt, np.std(pos, axis=0), size=self.nwalkers)
            pos = self._fix_guesses(pos)
            
        elif not restart:
            pos = self.guesses
            state = None
        else:
            state = None # should this be saved and restarted?

        #
        ## MAIN CALCULATION BELOW
        #

        if rank == 0:
            print "Starting MCMC: %s" % (time.ctime())
            
        # Take steps, append to pickle file every save_freq steps
        ct = 0
        pos_all = []; prob_all = []; blobs_all = []
        for pos, prob, state, blobs in self.sampler.sample(pos, 
            iterations=steps, rstate0=state, storechain=False):

            # Only the rank 0 processor ever makes it here
            ct += 1

            pos_all.append(pos.copy())
            prob_all.append(prob.copy())
            blobs_all.append(blobs)

            if ct % save_freq != 0:
                continue

            # Remember that pos.shape = (nwalkers, ndim)
            # So, pos_all has shape = (nsteps, nwalkers, ndim)

            data = [flatten_chain(np.array(pos_all)),
                    flatten_logL(np.array(prob_all)),
                    blobs_all]

            for i, suffix in enumerate(['chain', 'logL', 'blobs']):

                # Blobs
                if suffix == 'blobs':
                    if self.blob_names is None:
                        continue
                    self.save_blobs(data[i])
                # Other stuff
                else:
                    fn = '%s.%s.pkl' % (prefix, suffix)
                    f = open(fn, 'ab')
                    pickle.dump(data[i], f)
                    f.close()

            # This is a running total already so just save the end result 
            # for this set of steps
            f = open('%s.facc.pkl' % prefix, 'ab')
            pickle.dump(self.sampler.acceptance_fraction, f)
            f.close()

            print "Checkpoint: %s" % (time.ctime())

            del data, f, pos_all, prob_all, blobs_all
            gc.collect()

            # Delete chain, logL, etc., to be conscious of memory
            self.sampler.reset()

            pos_all = []; prob_all = []; blobs_all = []

        if self.pool is not None and emcee_mpipool:
            self.pool.close()
        elif self.pool is not None:
            self.pool.stop()

        if rank == 0:
            print "Finished on %s" % (time.ctime())
    
    def save_blobs(self, blobs, uncompress=True):
        """
        Write blobs to disk.
        
        Parameters
        ----------
        uncompress : bool
            True for MCMC, False for model grids.
        """
        
        # Number of steps taken between the last checkpoint and this one
        blen = len(blobs)
        # Usually this will just be = save_freq
        
        # Weird shape: must re-organize a bit
        # First, get rid of # walkers dimension and compress
        # the # of steps dimension
        if uncompress:
            blobs_now = []
            for k in range(blen):
                blobs_now.extend(blobs[k])
        else:
            blobs_now = blobs

        # We're saving one file per blob
        # The shape of the array will be just blob_nd

        for j, group in enumerate(self.blob_names):
            for k, blob in enumerate(group):
                to_write = []
                for l in range(self.nwalkers * blen):  
                    # indices: walkers*steps, blob group, blob
                    barr = blobs_now[l][j][k]
                                                    
                    to_write.append(barr)   
                    
                bfn = '%s.blob_%id.%s.pkl' \
                    % (self.prefix, self.blob_nd[j], blob)
                with open(bfn, 'ab') as f:
                    pickle.dump(np.array(to_write), f) 
                    
                       
