"""

ModelFit.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon May 12 14:01:29 MDT 2014

Description: 

"""

import numpy as np
from ..util.Stats import get_nu
from ..analysis import Global21cm
from ..util.PrintInfo import print_fit
from ..physics.Constants import nu_0_mhz
import gc, os, sys, copy, types, time, re
from ..analysis import Global21cm as anlG21
from ..analysis.GalacticForeground import GSM
from ..simulations import Global21cm as simG21
from ..util.Stats import Gauss1D, GaussND, rebin
from ..analysis.TurningPoints import TurningPoints
from ..analysis.InlineAnalysis import InlineAnalysis
from ..util.ReadData import flatten_chain, flatten_logL, flatten_blobs  
from ..util.SetDefaultParameterValues import _blob_names, _blob_redshifts

try:
    from scipy.spatial import KDTree
    from scipy.interpolate import interp1d
except ImportError:
    pass
    
#try:
#    import cPickle as pickle
#except:
import pickle    

try:
    import emcee
except ImportError:
    pass

emcee_mpipool = False
try:
    from mpi_pool import MPIPool
except ImportError:
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

nearest = lambda x, points: KDTree(points, leafsize=1e7).query(x)
    
# Uninformative prior
uninformative = lambda x, mi, ma: 1.0 if (mi <= x <= ma) else 0.0

# Gaussian prior
gauss1d = lambda x, mu, sigma: np.exp(-0.5 * (x - mu)**2 / sigma**2)

def_kwargs = {'track_extrema': 1, 'verbose': False, 'progress_bar': False}

default_errors = \
{
 'B': np.diag([0.1, 1.])**2,
 'C': np.diag([0.1, 1.])**2,
 'D': np.diag([0.1, 1.])**2,
}

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

class logprior:
    def __init__(self, priors, parameters):
        self.pars = parameters  # just names *in order*
        self.priors = priors

        if priors:
            self.prior_len = [len(self.priors[par]) for par in self.pars]

    def __call__(self, pars):
        """
        Compute log-likelihood of given model.
        """

        if not self.priors:
            return -np.inf

        logL = 0.0
        for i, par in enumerate(self.pars):
            val = pars[i]

            ptype = self.priors[self.pars[i]][0]
            if self.prior_len[i] == 3:
                p1, p2 = self.priors[self.pars[i]][1:]
            else:
                p1, p2, red = self.priors[self.pars[i]][1:]                

            # Figure out if this prior is linked to others
            if type(p1) is str:
                p1 = _str_to_val(p1, par, pars, self.pars)
            elif type(p2) is str:
                p2 = _str_to_val(p2, par, pars, self.pars)
                
            # Uninformative priors
            if ptype == 'uniform':
                logL -= np.log(uninformative(val, p1, p2))
            # Gaussian priors
            elif ptype == 'gaussian':
                logL -= np.log(gauss1d(val, p1, p2))
            else:
                raise ValueError('Unrecognized prior type: %s' % ptype)

        return logL
        
class loglikelihood:
    def __init__(self, steps, parameters, is_log, mu, errors,
        base_kwargs, nwalkers, priors={}, 
        errmap=None, errunits=None, prefix=None, fit_turning_points=True,
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
        self.fit_turning_points = fit_turning_points     

        self.blob_names = blob_names
        self.blob_redshifts = blob_redshifts
        
        print blob_names

        # Setup binfo pkl file
        self._prep_binfo()

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

        self.is_cov = False        
        if len(self.errors.shape) > 1:
            self.is_cov = True
            self.Dcov = np.linalg.det(self.errors)
            self.icov = np.linalg.inv(self.errors)
            
        self.errmap = errmap
        self.errunits = errunits

    @property
    def blank_blob(self):
        if not hasattr(self, '_blank_blob'):

            tup = tuple(np.ones(len(self.blob_names)) * np.inf)
            self._blank_blob = []
            for i in range(len(self.blob_redshifts)):
                self._blank_blob.append(tup)

        return np.array(self._blank_blob)
        
    def _prep_binfo(self):
        if rank > 0:
            return
        
        # Outputs for arbitrary meta-data blobs
        
        # Blob names and list of redshifts at which to track them
        f = open('%s.binfo.pkl' % self.prefix, 'wb')
        pickle.dump((self.blob_names, self.blob_redshifts), f)
        f.close()    
        
    def __call__(self, pars, blobs=None):
        """
        Compute log-likelihood for model generated via input parameters.

        Returns
        -------
        Tuple: (log likelihood, blobs)

        """

        kwargs = {}
        fg_kwargs = {}
        for i, par in enumerate(self.parameters):
            
            if par[0:2] == 'fg':
                save = fg_kwargs
            else:
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

        try:
            sim = simG21(**kw)
            sim.run()
            
            tps = sim.turning_points
        
        # Timestep weird (happens when xi ~ 1)
        except SystemExit:
            
            sim.run_inline_analysis()
            tps = sim.turning_points
                 
        # most likely: no (or too few) turning pts
        except:                     
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

        if not self.fit_turning_points:
            del sim, kw
            gc.collect()
            return lp, blobs

        # Compute the likelihood if we've made it this far

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

class ModelFit(object):
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
                
        TPs = 0
        for z in self.blob_redshifts:
            if z in list('BCD'):
                TPs += 1
                
        assert TPs > 0, 'Fitting won\'t work if no turning points are provided.'
            
    @property
    def error(self):
        if not hasattr(self, '_error'):
            self._error = []
            for key in list('BCD'):
                self._error.extend(np.sqrt(np.diag(default_errors[key])))
            self._error = np.array(self._error)

        return self._error
        
    @property
    def measurement_map(self):
        """
        A list containing pairs of the form (<Turning Point>, <0 or 1>), for
        each element of the mu vector. The <Turning Point> element should be
        'B', 'C', or 'D', while the <0 or 1> refer to redshift and brightness
        temperature, respectively.
        
        If mu is not supplied, these represent a map for the second
        dimension of a flattened MCMC chain.
        """
        if not hasattr(self, '_measurement_map'):
            self._measurement_map = \
                [('B', 0), ('C', 0), ('D', 0),
                 ('B', 1), ('C', 1), ('D', 1)]
            
            if hasattr(self, '_mu'):
                del self._mu

        return self._measurement_map

    @measurement_map.setter
    def measurement_map(self, value):
         self._measurement_map = value 

    @property
    def measurement_units(self):
        """
        A two-element tuple containing units assumed for measurements.
        
        By measurements, we mean the units of the mean and 1-D errors 
        (or chain) supplied. ares.simulations.Global21cm will always return
        turning points in redshift and mK units.
        
        First element is MHz or redshift, second is mK or K.
        """
        if not hasattr(self, '_measurement_units'):
            self._measurement_units = ('redshift', 'mK')
    
        return self._measurement_units
    
    @measurement_units.setter
    def measurement_units(self, value):
        self._measurement_units = value
                 
    @property
    def mu(self):
        if not hasattr(self, '_mu'):
            if hasattr(self, '_turning_points'):
                self._mu = []
                for tp, i in self.measurement_map:
                    self._mu.append(self._turning_points[tp][i])
            else:
                self._mu = None
            
        return self._mu    
            
    @mu.setter
    def mu(self, value):
        """
        Set turning point positions.
        
        Parameters
        ----------
        value : np.ndarray, ares.analysis.Global21cm instance
            Array corresponding to measurement_map or analysis class instance.
            
        """
                
        if type(value) is dict:
            self._turning_points = value
        else:    
            self._mu = value        
            
    def set_error(self, error1d=None, cov=None, nu=0.68):
        """
        Set errors to be used in likelihood calculation.

        Parameters
        ----------
        error1d : np.ndarray
            Array of nu-sigma error bars on turnings points
        cov : np.ndarray
            Covariance matrix
        nu : float
            Confidence contour the errors correspond to.

        Sets
        ----
        Attribute "errors".

        """

        if error1d is not None:
            
            # Convert to 1-sigma errors
            err = []
            for val in error1d:
                err.append(get_nu(val, nu_in=nu, nu_out=0.68))
            
            self._error = np.array(err)
        
        elif cov is not None:    
            self._error = cov
        
    @property
    def priors(self):
        if not hasattr(self, '_priors'):
            raise ValueError('Must set priors (set_priors)!')

        return self._priors

    @priors.setter
    def priors(self, value):
        self._priors = value
    
    def _check_for_conflicts(self):
        """
        Hacky at the moment. Preventative measure against is_log=True for
        spectrum_logN. Could generalize.
        """
        for i, element in enumerate(self.parameters):
            if re.search('spectrum_logN', element):
                if self.is_log[i]:
                    raise ValueError('spectrum_logN is already logarithmic!')
    
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
    def guesses(self):
        """
        Generate initial position vectors for all walkers.
        """
        
        if not hasattr(self, '_guesses'):
            
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
        
    def _fix_guesses(self):
        pass    
        
    @guesses.setter
    def guesses(self, value):
        self._guesses = value
                        
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
        
    def set_cov(self, cov):
        self.cov = np.diag(cov)
        
    @property
    def data(self):
        if not hasattr(self, '_data'):
            self._data = None
    
        return self._data
    
    @data.setter
    def data(self, value):
        self._data = value    

    def run(self, prefix, steps=1e2, burn=0, clobber=False, restart=False, 
        save_freq=500, fit_turning_points=True):
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

        """
        
        self._check_for_conflicts()
        
        self.prefix = prefix
        
        if os.path.exists('%s.chain.pkl' % prefix) and (not clobber):
            if not restart:
                raise IOError('%s exists! Remove manually, set clobber=True, or set restart=True to append.' 
                    % prefix)
        
        if not os.path.exists('%s.chain.pkl' % prefix) and restart:
            raise IOError("This can't be a restart, %s*.pkl not found." % prefix)

        print_fit(self, steps=steps, burn=burn, fit_TP=fit_turning_points)

        if len(self.error.shape) == 1:
            assert len(self.error) == len(self.measurement_map)
        else:
            assert len(np.diag(self.error)) == len(self.measurement_map)

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
            self.mu, self.error, self.base_kwargs,
            self.nwalkers, self.priors,
            self.measurement_map, self.measurement_units, prefix=self.prefix,
            fit_turning_points=fit_turning_points,
            blob_names=self.blob_names,
            blob_redshifts=self.blob_redshifts)
            
        self.sampler = emcee.EnsembleSampler(self.nwalkers,
            self.Nd, self.loglikelihood, pool=self.pool)
                
        if burn > 0:
            t1 = time.time()
            pos, prob, state, blobs = self.sampler.run_mcmc(self.guesses, burn)
            self.sampler.reset()
            t2 = time.time()

            if rank == 0:
                print "Burn-in complete in %.3g seconds." % (t2 - t1)

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

            pos = np.array(guesses)

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
                if not os.path.exists(fn):
                    continue

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
    
