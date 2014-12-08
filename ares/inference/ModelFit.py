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
from ..analysis import Global21cm as anlG21
from ..simulations import Global21cm as simG21
import gc, os, sys, copy, types, pickle, time, re
from ..analysis.TurningPoints import TurningPoints
from ..util.Stats import Gauss1D, GaussND, error_1D, rebin
from ..util.ReadData import read_pickled_chain, flatten_chain, flatten_logL, \
    flatten_blobs  
    
try:
    from scipy.spatial import KDTree
    from scipy.interpolate import interp1d
except ImportError:
    pass

try:
    import emcee
    from emcee.utils import MPIPool
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
        
def parse_turning_points(turning_points_model, turning_points_input, use):
    """
    Convert dictionary of turning points to array.

    Parameters
    ----------
    turning_points_input : list
        Turning points that appeared in the reference model.
    turning_points_model : dict
        Turning points for some model we just calculated in our parameter
        space exploration. Same format as turning_points_input.
    use : dict
        For example, 
            use={'B': [1, 0]} 
        means use the redshift of turning point B in the fit, but not the 
        brightness temperature.

    """

    arr_TPs = []
    for tp in turning_points_input:

        if tp not in turning_points_model:
            arr_TPs.extend([0.0] * 2)
            continue

        if tp not in use:
            continue    

        if use[tp][0]:    
            arr_TPs.append(float(turning_points_model[tp][0]))
        if use[tp][1]:    
            arr_TPs.append(float(turning_points_model[tp][1]))

    return np.array(arr_TPs)

nearest = lambda x, points: KDTree(points, leafsize=1e7).query(x)
    
turning_points_all = list('BCD')

# Prior for parameters that cannot have values exceeding 1
lt1gt0 = lambda x: 1.0 if x <= 1.0 else 0.0

# Priors that must be positive
gt0 = lambda x: 1.0 if x > 0 else 0.0

# Uninformative prior
uninformative = lambda x, mi, ma: 1.0 if (mi <= x <= ma) else 0.0

# Gaussian prior
gauss1d = lambda x, mu, sigma: np.exp(-0.5 * (x - mu)**2 / sigma**2)

# Other priors
other_priors = {'Tmin': lambda x: 1.0 if x > 300. else 0.0}

lt1gt0_pars = ['fstar', 'fesc', 'eta']
gt0_pars = ['Nlw', 'Tmin']

def_kwargs = {'track_extrema': 1, 'verbose': False, 'progress_bar': False}

default_errors = \
{
 'B': np.diag([0.1, 1.])**2,
 'C': np.diag([0.1, 1.])**2,
 'D': np.diag([0.1, 1.])**2,
}

_z_blob = list('BCD')
_z_blob.extend(list(np.arange(10., 36.)))

default_blobs = \
    (['dTb', 'z', 'curvature', 'igm_Tk', 'igm_heat', 'cgm_h_2', 'cgm_Gamma', 
    'Ts', 'Ja'], _z_blob)

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
        base_kwargs, nwalkers, priors={}, chain=None, logL=None, 
        errmap=None, errunits=None, prefix=None, fit_turning_points=True,
        burn=False):
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
        
        if 'inline_analysis' in self.base_kwargs:
            self.blob_names, self.blob_redshifts = \
                self.base_kwargs['inline_analysis']
                        
        # Sort through priors        
        priors_P = {}   # parameters
        priors_B = {}   # blobs
        
        p_pars = []
        b_pars = []
        for key in priors:
            # Priors on model parameters
            if len(priors[key]) == 3:
            #if key in self.parameters:
                p_pars.append(key)
                priors_P[key] = priors[key]
                continue
            
            if len(priors[key]) == 4:
            #if key in ['tau_e', 'cgm_h_2']:
                b_pars.append(key)
                priors_B[key] = priors[key]
                continue

        self.logprior_P = logprior(priors_P, self.parameters)
        self.logprior_B = logprior(priors_B, b_pars)
        
        self.errors = errors
        self.errmap = errmap
        self.errunits = errunits
        self.chain = chain
        self.logL = logL
        
        if self.logL is not None:
            self.logL_size = logL.size
        
        if self.chain is not None:
            self.mu = self.chain[np.argmax(self.logL)]
            self.sigma = np.std(self.chain, axis=0)
        else:
            self.mu = mu

        self.error_type = 'non-ideal' if self.chain is not None \
            else 'idealized'

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
        
        try:
            sim = simG21(**kw)
            sim.run()  

            tps = sim.turning_points      

        # most likely: no (or too few) turning pts
        except:         
            #self.warning(None, kwargs)
            
            # Write to "fail" file - this might cause problems in parallel
            if not self.burn:
                f = open('%s.fail.pkl' % self.prefix, 'ab')
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
                
            z = self.logprior_B.priors[key][-1]
            
            i = self.blob_names.index(key) 
            j = self.blob_redshifts.index(z)
            
            val = sim.blobs[j,i]
                                
            blob_vals.append(val)    
                
        if blob_vals:
            lp -= self.logprior_B(blob_vals)
            
        if hasattr(sim, 'blobs'):
            blobs = sim.blobs
        else:
            blobs = self.blank_blob    
            
        if not self.fit_turning_points:
            return lp, blobs

        # Compute the likelihood if we've made it this far
        xarr = []
                    
        # Convert frequencies to redshift, temperatures to K
        for element in self.errmap:
            tp, i = element
                        
            # Models without turning point B, C, or D get thrown out.
            if tp not in tps:
                del sim, kw
                gc.collect()
                
                return -np.inf, self.blank_blob
        
            xarr.append(tps[tp][i])
                   
        # Values of current model that correspond to mu vector
        xarr = np.array(xarr)           
                   
        # Compute log-likelihood, including prior knowledge

        if self.error_type == 'idealized':
            logL = lp \
                - np.sum((xarr - self.mu)**2 / 2. / self.errors**2)

        else:
            
            try:
                dist, loc = nearest(xarr, self.chain)
                
                if loc >= self.logL_size:
                    logL = -np.inf
                else:
                    logL = lp - self.logL[loc]
            except ValueError:
                del sim, kw
                gc.collect()
                
                return -np.inf, self.blank_blob

        # Blobs!
        if hasattr(sim, 'blobs'):
            blobs = sim.blobs
        else:
            blobs = self.blank_blob
                                                
        if blobs.shape != self.blank_blob.shape:
            raise ValueError('help')    
            
        del sim, kw
        gc.collect()    

        return logL, blobs
        
    def warning(self, tps, kwargs):
        print "\n----------------------------------"
        print "WARNING: Error w/ following model:"
        print "----------------------------------"
        print "kw = \\"
        print "{"
        for kw in kwargs:
            print ' \'%s\': %.4g,' % (kw, kwargs[kw])
        print "}"
        
        print "----------------------------------"
        
        if tps is not None:
            print "# of turning points: %i" % len(tps)    
            print "----------------------------------\n"    
        else:
            print ""
        
class DummySampler:
    def __init__(self):
        pass        

class ModelFit(object):
    def __init__(self, fn=None, **kwargs):
        """
        Initialize an extracted signal.
        
        Parameters
        ----------
        fn : str
            Output filename of previous MCMC run.
        
        Optional Keyword Arguments
        --------------------------
        Anything you want based to each ares.simulations.Global21cm call.
        
        """
                
        self.base_kwargs = def_kwargs.copy()
        self.base_kwargs.update(kwargs)
        
        # Prepare for blobs (optional)
        if 'inline_analysis' in self.base_kwargs:
            self.blob_names, self.blob_redshifts = \
                self.base_kwargs['inline_analysis']
        
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
         
    def _check_for_conflicts(self):
        
        for i, element in enumerate(self.parameters):
            if re.search('spectrum_logN', element):
                if self.is_log[i]:
                    raise ValueError('spectrum_logN is already logarithmic!')
                    
            
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
            
    def set_error(self, error1d=None, nu=0.68, chain=None, logL=None,
        force_idealized=False):
        """
        Set errors to be used in likelihood calculation.

        Parameters
        ----------
        value : 2 options here

        Sets
        ----
        Attribute "errors"

        """

        if error1d is not None:
            
            err = []
            for val in error1d:
                err.append(get_nu(val, nu_in=nu, nu_out=0.68))
            
            self._error = np.array(err)
            
            # Convert units
            #for tp, i in self.measurement_map:
            #    
            #    # Convert frequencies to redshifts
            #    if i == 0 and self.measurement_units[i] == 'MHz':
            #        self._error[i] = 
            #    
            #    self._mu.append(self._turning_points[tp][i])
            
        else:            
            self.chain, self.logL = chain, logL
            
            # Convert units
            for j, (tp, i) in enumerate(self.measurement_map):
                
                # Convert frequencies to redshifts
                if i == 0 and self.measurement_units[i] == 'MHz':
                    self.chain[:,j] = (nu_0_mhz / self.chain[:,j]) - 1.
                elif i == 1 and self.measurement_units[i] == 'K':
                    self.chain[:,j] *= 1e3
                    
            if force_idealized:
                self._mu = np.mean(self.chain, axis=0)
                self._error = np.std(self.chain, axis=0)   
                del self.chain, self.logL     
         
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
    def guesses(self):
        if not hasattr(self, '_guesses'):
            
            self._guesses = []
            for i in range(self.nwalkers):
                
                p0 = []
                for j, par in enumerate(self.parameters):

                    if par in self.priors:
                        dist, lo, hi = self.priors[par]
                        
                        if dist == 'uniform':
                            val = np.random.rand() * (hi - lo) + lo
                        else:
                            val = np.random.normal(lo, scale=hi)
                    else:
                        raise ValueError('No prior for %s' % par)

                    p0.append(val)
                
                self._guesses.append(p0)
        
            self._guesses = np.array(self._guesses)
        
        return self._guesses
        
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
    def priors(self):
        if not hasattr(self, '_priors'):
            self._priors = {}
        
        return self._priors    

    @priors.setter
    def priors(self, value):
        self._priors = value

    def run(self, prefix, steps=1e2, burn=0, clobber=False, restart=False, 
        save_freq=10, fit_turning_points=True):
        """
        Run MCMC.
        
        Parameters
        ----------
        prefix : str
            Prefix for all output files
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

        if not hasattr(self, 'chain'):
            self.chain = None
            self.logL = None
        
            assert len(self.error) == len(self.measurement_map)
            
        else:
            assert self.chain.shape[1] == len(self.measurement_map)
            
        if size > 1:
            self.pool = MPIPool()
            if not self.pool.is_master():
                self.pool.wait()
                sys.exit(0)
        else:
            self.pool = None    
            
        # Burn-in using stdev from raw chains as error
        if self.chain is not None:
        
            self.loglikelihood_BURN = loglikelihood(steps, self.parameters, self.is_log, 
                np.mean(self.chain), np.std(self.chain), self.base_kwargs,
                self.nwalkers, self.priors, None, None, 
                self.measurement_map, self.measurement_units,
                fit_turning_points=fit_turning_points, burn=True)

            self.sampler_BURN = emcee.EnsembleSampler(self.nwalkers,
                self.Nd, self.loglikelihood_BURN, pool=self.pool)
                
        self.loglikelihood = loglikelihood(steps, self.parameters, self.is_log, 
            self.mu, self.error, self.base_kwargs,
            self.nwalkers, self.priors, self.chain, self.logL, 
            self.measurement_map, self.measurement_units, prefix=self.prefix,
            fit_turning_points=fit_turning_points)
            
        self.sampler = emcee.EnsembleSampler(self.nwalkers,
            self.Nd, self.loglikelihood, pool=self.pool)
                
        if burn > 0:
            t1 = time.time()
            if self.chain is not None:
                pos, prob, state, blobs = self.sampler_BURN.run_mcmc(self.guesses, burn)
            else:
                pos, prob, state, blobs = self.sampler.run_mcmc(self.guesses, burn)
                self.sampler.reset()
            t2 = time.time()
            
            if rank == 0:
                print "Burn-in complete in %.3g seconds." % (t2 - t1)
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
        if rank == 0 and (not restart):

            # Main output: MCMC chains (flattened)
            f = open('%s.chain.pkl' % prefix, 'wb')
            f.close()
            
            # Main output: log-likelihood
            f = open('%s.logL.pkl' % prefix, 'wb')
            f.close()
            
            f = open('%s.fail.pkl' % prefix, 'wb')
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
            
            # Outputs for arbitrary meta-data blos
            if hasattr(self, 'blob_names'):
                
                # File for blobs themselves
                f = open('%s.blobs.pkl' % prefix, 'wb')
                f.close()
                
                # Blob names and list of redshifts at which to track them
                f = open('%s.binfo.pkl' % prefix, 'wb')
                pickle.dump((self.blob_names, self.blob_redshifts), f)
                f.close()
                        
        # Take steps, append to pickle file every save_freq steps
        ct = 0
        pos_all = []; prob_all = []; blobs_all = []
        for pos, prob, state, blobs in self.sampler.sample(pos, 
            iterations=steps, rstate0=state, storechain=False):
                        
            # Only the rank 0 processor ever makes it here
            
            ct += 1
            
            pos_all.append(pos)
            prob_all.append(prob)
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
                                
            print "Checkpoint: %s" % (time.ctime())
             
            del data, f, pos_all, prob_all, blobs_all
            gc.collect()
            self.sampler.reset()

            pos_all = []; prob_all = []; blobs_all = []

        if self.pool is not None:
            self.pool.close()
    
