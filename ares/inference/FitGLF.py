"""

FitGLF.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Oct 23 14:34:01 PDT 2015

Description: 

"""

import numpy as np
from ..util.Stats import get_nu
from ..util.PrintInfo import print_fit
from ..physics.Constants import nu_0_mhz
import gc, os, sys, copy, types, time, re
from ..analysis import Global21cm as anlG21
from ..simulations import Global21cm as simG21
from ..util.Stats import Gauss1D, GaussND, rebin
from ..analysis.TurningPoints import TurningPoints
from ..analysis.InlineAnalysis import InlineAnalysis
from ..util.SetDefaultParameterValues import _blob_names, _blob_redshifts, \
    SetAllDefaults
from ..util.ReadData import flatten_chain, flatten_logL, flatten_blobs, \
    read_pickled_chain
    
from ..util import read_lit
from ..util.ParameterFile import pop_id_num    
from ..populations.Galaxy import param_redshift
    
from .FitGlobal21cm import logprior

try:
    from scipy.interpolate import interp1d
except ImportError:
    pass
    
try:
    import cPickle as pickle
except:
    import pickle    

try:
    import emcee
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
    import h5py
except ImportError:
    pass

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1
    
def_params = SetAllDefaults()    
def_kwargs = {'track_extrema': 1, 'verbose': False, 'progress_bar': False,
    'one_file_per_blob': True}

class loglikelihood:
    def __init__(self, steps, parameters, is_log, x, z, mu, errors,
        base_kwargs, nwalkers, priors={}, prefix=None,
        burn=False, blob_names=None, blob_redshifts=None, runsim=False):
        """
        Computes log-likelihood at given step in MCMC chain.

        Parameters
        ----------

        """

        self.parameters = parameters # important that they are in order?
        self.is_log = is_log

        self.base_kwargs = base_kwargs
        self.nwalkers = nwalkers

        self.x = x
        self.z = z

        self.burn = burn
        self.prefix = prefix   

        self.blob_names = blob_names
        self.blob_redshifts = blob_redshifts
        
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

        self.logprior_P = logprior(priors_P, self.parameters)
        self.logprior_B = logprior(priors_B, b_pars)

        self.mu = mu
        self.errors = errors
        self.runsim = runsim
        
        # Setup binfo pkl file
        self._prep_binfo()
        
        # Setup timing file
        #self._prep_timing()
        
    def _prep_timing(self):
        fn = '%s.timing_%s.pkl' % (self.prefix, str(rank).zfill(4))
        if os.path.exists(fn):
            return
        f = open(fn, 'wb')
        f.close()
        
    def _prep_binfo(self):
        if rank > 0:
            return
    
        # Outputs for arbitrary meta-data blobs
    
        # Blob names and list of redshifts at which to track them
        f = open('%s.binfo.pkl' % self.prefix, 'wb')
        pickle.dump((self.blob_names, self.blob_redshifts), f)
        f.close()
        
    @property
    def blank_blob(self):
        if not hasattr(self, '_blank_blob'):

            tup = tuple(np.ones(len(self.blob_names)) * np.inf)
            self._blank_blob = []
            for i in range(len(self.blob_redshifts)):
                self._blank_blob.append(tup)
    
        return np.array(self._blank_blob)
    
    def __call__(self, pars, blobs=None):
        """
        Compute log-likelihood for model generated via input parameters.
    
        Returns
        -------
        Tuple: (log likelihood, blobs)
    
        """

        kwargs = {}
        for i, par in enumerate(self.parameters):
    
            save = kwargs
    
            if self.is_log[i]:
                save[par] = 10**pars[i]
            else:
                save[par] = pars[i]
            
        # Apply prior on model parameters first (dont need to generate signal)
        lp = self.logprior_P(pars)
        if not np.isfinite(lp):
            return -np.inf, self.blank_blob
    
        # Run a model and retrieve turning points
        kw = self.base_kwargs.copy()
        kw.update(kwargs)
    
        sim = simG21(**kw)
        
        # If we're only fitting the LF, no need to run simulation
        if self.runsim:
            t1 = time.time()
            sim.run()
            t2 = time.time()
            sim.run_inline_analysis()
                                    
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

        for popid, pop in enumerate(sim.medium.field.pops):
            if not pop.is_ham_model:
                continue
            break

        phi = []
        for i, z in enumerate(self.z):
            pop = sim.medium.field.pops[popid]
            p = pop.LuminosityFunction(M=np.array(self.x[i]), z=z)
            phi.extend(p)

        phi = np.array(phi)

        mu = np.concatenate(self.mu)
        err = np.concatenate(self.errors)

        logL = lp - 0.5 * (np.sum((phi - mu)**2 / 2. / err**2 \
            + np.log(2. * np.pi * err**2)))

        #if blobs.shape != self.blank_blob.shape:
        #    raise ValueError('Shape mismatch between requested blobs and actual blobs!')    
    
        del sim, kw
        gc.collect()
    
        return logL, blobs
    
class FitGLF(object):
    def __init__(self, **kwargs):
        """
        Initialize a class for fitting the turning points in the global
        21-cm signal.
    
        Optional Keyword Arguments
        --------------------------
        Anything you want based to each ares.simulations.Global21cm call.
    
        """
    
        self.base_kwargs = def_kwargs.copy()
        self.base_kwargs.update(kwargs)
    
        if 'auto_generate_blobs' in self.base_kwargs:            
            if self.base_kwargs['auto_generate_blobs']:
                kw = self.base_kwargs.copy()
        
                sim = simG21(**kw)
                sim.run()
                
                anl = InlineAnalysis(sim)
        
                self.blob_names, self.blob_redshifts = \
                    anl.generate_blobs()
        
                del sim, anl
                gc.collect()
        
                self.base_kwargs['inline_analysis'] = \
                    (self.blob_names, self.blob_redshifts)
                self.base_kwargs['auto_generate_blobs'] = False
        else:
        
            if 'inline_analysis' in self.base_kwargs and \
                (not hasattr(self, 'blob_names')):
                self.blob_names, self.blob_redshifts = \
                    self.base_kwargs['inline_analysis']
        
            elif (not hasattr(self, 'blob_names')):
                self.blob_names = _blob_names
                self.blob_redshifts = _blob_redshifts
        
            self.base_kwargs['inline_analysis'] = \
                (self.blob_names, self.blob_redshifts)
        
        self.one_file_per_blob = self.base_kwargs['one_file_per_blob']        
    
    @property
    def error(self):
        if not hasattr(self, '_error'):
            raise AttributeError('Must run `set_error`!')
    
        return self._error

    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, value):
        self._x = value
        
    @property
    def z(self):
        return self._z
    
    @z.setter
    def z(self, value):
        self._z = value
    
    @property
    def mu(self):
        return self._mu    
    
    @mu.setter
    def mu(self, value):
        self._mu = value
    
    def set_error(self, err, nu=0.68):
        """
        Set errors to be used in likelihood calculation.
    
        Parameters
        ----------
        error1d : np.ndarray
            Array of nu-sigma error bars on turnings points
        nu : float
            Confidence contour the errors correspond to.
    
        Sets
        ----
        Attribute `_error`.
    
        """
    
        # Convert to 1-sigma errors
        #err = []
        #for val in error1d:
        #    err.append(get_nu(val, nu_in=nu, nu_out=0.68))

        self._error = err#np.array(err)
    
    @property
    def priors(self):
        if not hasattr(self, '_priors'):
            raise ValueError('Must set priors (set_priors)!')
    
        return self._priors
    
    @priors.setter
    def priors(self, value):
        self._priors = value
    
    @property
    def nwalkers(self):
        if not hasattr(self, '_nw'):
            self._nw = self.Nd * 2
    
            if rank == 0:
                print "Set nwalkers=%i." % self._nw
    
        return self._nw
    
    @nwalkers.setter
    def nwalkers(self, value):
        self._nw = value
    
    @property
    def _guess_override_(self):
        if not hasattr(self, '_guess_override_'):
            self._guess_override_ = {}
        
        return self._guess_override_
        
    def guess_override(self, **kwargs):
        if not hasattr(self, '_guess_override'):
            self._guess_override_ = {}
            
        self._guess_override.update(kwargs)
        return self._guess_override
        
    #@property
    #def jitter(self):
    #    pass
    #
    #@jitter.setter(self):
    #    pass
        
    @property
    def guesses(self):
        if not hasattr(self, '_guesses'):
            jitter = 0.05
            
            b15 = read_lit('bouwens2015')
                        
            self._guesses = np.zeros([self.nwalkers, len(self.parameters)])
            for i, key in enumerate(self.parameters):
                
                scat = np.random.normal(scale=jitter, size=self.nwalkers)
                                  
                if 'pop_lf_' not in key or (not re.search('\[', key)):
                    prefix, popid = pop_id_num(prefix_w_pop)
                    val = self.base_kwargs[key]
                else:
                    prefix_w_pop, z = param_redshift(key)
                    prefix, popid = pop_id_num(prefix_w_pop)
                    par = prefix.replace('pop_lf_', '')
                    k = np.argmin(np.abs(z - np.array(b15.redshifts)))
                    val = b15.lf_pars[par][k]
                                    
                if self.is_log[i]:
                    self._guesses[:,i] = np.log10(val)
                else:
                    self._guesses[:,i] = val
                    
                self._guesses[:,i] *= (1. + scat)    
                    
        return self._guesses

    def set_axes(self, parameters, is_log=True):
        """
        Set axes of parameter space to explore.

        Parameters
        ----------
        parameters : list
            List of parameters to vary in fit.
        is_log : bool, list
            Explore log10 parameter space?

        """

        self.parameters = parameters
        self.Nd = len(self.parameters)
    
        if type(is_log) is bool:
            self.is_log = [is_log] * self.Nd
        else:
            self.is_log = is_log
    
    def tight_ball(self, pos, prob):
        # emcee has something like this already, should probably just use that
    
        # Set new initial position to region of high likelihood
        imaxL = np.argsort(prob)[-1::-1]
    
        pvec = []
        for i in range(self.nwalkers / 4):
            if np.isinf(imaxL[i]):
                break
    
            pvec.append(pos[imaxL[i]])
    
        pvec = np.array(pvec)
    
        ball_prior = {}
        for j, par in enumerate(self.parameters):
            ball_prior[par] = \
                ['gaussian', pvec[:,j].mean(), pvec[:,j].std()]
    
        guesses = []
        for i in range(self.nwalkers):
    
            p0 = []
            for j, par in enumerate(self.parameters):
    
                dist, lo, hi = ball_prior[par]
    
                if dist == 'uniform':
                    val = np.random.rand() * (hi - lo) + lo
                else:
                    val = np.random.normal(lo, scale=hi)
    
                p0.append(val)
    
            guesses.append(p0)
    
        return np.array(guesses) 
        
    def save_data(self, prefix, clobber=False):
        if rank > 0:
            return
            
        fn = '%s.data.pkl' % prefix
        
        if os.path.exists(fn) and (not clobber):
            print "%s exists! Set clobber=True to overwrite." % fn
            return
                
        f = open(fn, 'wb')
        pickle.dump((self.x, self.mu, self.z, self.error), f)
        f.close()
     
    def run(self, prefix, steps=1e2, burn=0, clobber=False, restart=False, 
        runsim=False, save_freq=500, live_dangerously=False):
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
        fit_turning_points : bool
            If False, merely explore prior space.
        frequency_channels : np.ndarray
            Frequencies corresponding to 'mu' values if fit_signal=True.
    
        """
    
        #self._check_for_conflicts()
    
        self.prefix = prefix
        
        assert len(self.z) == len(self.x) == len(self.mu), \
            "Multi redshift data mismatch!"
    
        if os.path.exists('%s.chain.pkl' % prefix) and (not clobber):
            if not restart:
                raise IOError('%s exists! Remove manually, set clobber=True, or set restart=True to append.' 
                    % prefix)
    
        if not os.path.exists('%s.chain.pkl' % prefix) and restart:
            raise IOError("This can't be a restart, %s*.pkl not found." % prefix)
    
        print_fit(self, steps=steps, burn=burn, fit_TP=False)
    
        assert len(self.error) == len(self.mu)
        
        self.save_data(prefix, clobber=clobber)

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

        self.loglikelihood = loglikelihood(steps, self.parameters, self.is_log, 
            self.x, self.z, self.mu, self.error, self.base_kwargs,
            self.nwalkers, self.priors,
            prefix=self.prefix, blob_names=self.blob_names,
            blob_redshifts=self.blob_redshifts, runsim=runsim)
    
        self.sampler = emcee.EnsembleSampler(self.nwalkers,
            self.Nd, self.loglikelihood, pool=self.pool)
    
        if burn > 0 and not restart:
            t1 = time.time()
            pos, prob, state, blobs = self.sampler.run_mcmc(self.guesses, burn)
            self.sampler.reset()
            t2 = time.time()
    
            if rank == 0:
                print "Burn-in complete in %.3g seconds." % (t2 - t1)
    
            pos = self.tight_ball(pos, prob)
        else:
            pos = self.guesses
            state = None
    
        # Check to see if things match
        if restart:
            f = open('%s.pinfo.pkl' % prefix, 'rb')
            pars, is_log = pickle.load(f)
            f.close()
    
            if pars != self.parameters:
                if size > 1:
                    if rank == 0:
                        print 'warning: parameters from file dont match those supplied!'
                    
                    if not live_dangerously:
                        if rank == 0:
                            print "if you dont care, set live_dangerously=True"
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
    
            ct = 0
            for kw in base_kwargs:
                try:
                    if base_kwargs[kw] != self.base_kwargs[kw]:
                        ct += 1
                # Need to handle arrays separately
                except ValueError:
                    if not np.allclose(base_kwargs[kw], self.base_kwargs[kw]):
                        ct += 1
    
            if ct > 0:
                if size > 1:
                    if rank == 0:
                        print 'warning: base_kwargs from file dont match those supplied!'
                    if not live_dangerously:
                        if rank == 0:
                            print "if you dont care, set live_dangerously=True"
                        MPI.COMM_WORLD.Abort()
                        raise ValueError('base_kwargs from file dont match those supplied!')            
    
            # Start from last step in pre-restart calculation
            chain = read_pickled_chain('%s.chain.pkl' % prefix)
    
            pos = chain[-self.nwalkers:,:]
    
        # Setup output file
        else:
    
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
            if self.one_file_per_blob:
                for blob in self.blob_names:
                    f = open('%s.subset.%s.pkl' % (prefix, blob), 'wb')
                    f.close()
            else:
                f = open('%s.blobs.pkl' % prefix, 'wb')
                f.close()
    
            # Blob-info "binfo" file will be written by likelihood
    
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
                    flatten_blobs(np.array(blobs_all))]

            for i, suffix in enumerate(['chain', 'logL', 'blobs']):
                fn = '%s.%s.pkl' % (prefix, suffix)

                # Skip blobs if there are none being tracked
                if blobs_all == [{}] * len(blobs_all):
                    continue

                if suffix == 'blobs':
                    if self.one_file_per_blob:
                        for j, blob in enumerate(self.blob_names):
                            barr = np.array(data[i])[:,:,j]
                            bfn = '%s.subset.%s.pkl' % (self.prefix, blob)
                            with open(bfn, 'ab') as f:
                                pickle.dump(barr, f)                        
                    else:
                        with open('%s.blobs.pkl' % self.prefix, 'ab') as f:
                            pickle.dump(data[i], f)

                    continue

                f = open(fn, 'ab')
                pickle.dump(data[i], f)
                f.close()

            # This is a running total already so just save the end result 
            # for this set of steps
            f = open('%s.facc.pkl' % prefix, 'ab')
            pickle.dump(self.sampler.acceptance_fraction, f)
            f.close()
    
            print "Checkpoint #%i: %s" % (ct / save_freq, time.ctime())
    
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
    
    
    
    
    
    
    
    
    
    
