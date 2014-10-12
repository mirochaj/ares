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
import os, sys, copy, types, pickle, time
from ..simulations import Global21cm as simG21
from ..analysis.TurningPoints import TurningPoints
from ..util.ReadData import read_pickled_chain, flatten_chain, flatten_logL, \
    flatten_blobs  

try:
    from scipy.spatial import KDTree
    from scipy.interpolate import interp1d, NearestNDInterpolator
except ImportError:
    pass

try:
    import emcee
    from emcee.utils import MPIPool
except ImportError:
    pass
    
try:
    have_mathutils = True
    from mathutils.stats import Gauss1D, GaussND, error_1D, rebin
except ImportError:
    have_mathutils = False    
    
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
            p1, p2 = self.priors[self.pars[i]][1:]
            
            if self.prior_len[i] == 3:
            
                # Uninformative priors
                if ptype == 'uniform':
                    logL -= np.log(uninformative(val, p1, p2))
                elif ptype == 'gaussian':
                    logL -= np.log(gauss1d(val, p1, p2))
                else:
                    raise ValueError('Unrecognized prior type: %s' % ptype)
            
            elif self.prior_len[i] == 4:
                z = self.priors[self.pars[i]][-1]
                                
            else:
                raise ValueError('Unsupported prior information.')
                

        return logL
        
class loglikelihood:
    def __init__(self, steps, parameters, is_log, mu, errors,
        base_kwargs, nwalkers, priors={}, chain=None, logL=None, 
        errmap=None, errunits=None, blobs=None):
        """
        Computes log-likelihood at given step in MCMC chain.
        
        Parameters
        ----------
        
        """
        
        self.parameters = parameters # important that they are in order?
        self.is_log = is_log

        self.base_kwargs = base_kwargs
        self.nwalkers = nwalkers
        
        self.blobs = blobs
        
        if 'inline_analysis' in self.base_kwargs:
            self.blob_names, self.blob_redshifts = \
                self.base_kwargs['inline_analysis']
                        
        # Sort through priors
        
        priors_P = {}   # parameters
        priors_M = {}   # measurements - same as blobs
        priors_DQ = {}  # derived quantities - maybe same as blobs
        priors_B = {}   # blobs
        
        for key in priors:
            # Priors on model parameters
            if key in self.parameters:
                priors_P[key] = priors[key]
                continue

            # Otherwise, split by _
            i = key.rfind('_')
            key_pre = key[0:i]
            pt = key[i+1:]

            if key_pre in ['z', 'dTb']:
                priors_M[key] = priors[key]
                continue

            # Must be a prior on a derived quantity
            priors_DQ[key] = priors[key]

        self.logprior_P = logprior(priors_P, parameters)
        self.logprior_M = logprior(priors_M, parameters)
        self.logprior_DQ = logprior(priors_DQ, parameters)
        self.logprior_B = logprior(priors_B, parameters)

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
                                        
    def __call__(self, pars):
        """
        Compute log-likelihood for model generated via input parameters.
        """
        
        kwargs = {}
        for i, par in enumerate(self.parameters):
            if self.is_log[i]:
                kwargs[par] = 10**pars[i]
            else:
                kwargs[par] = pars[i]
        
        # Apply prior
        lp = self.logprior_P(pars)
        if not np.isfinite(lp):
            return -np.inf, []

        # Run a model and retrieve turning points
        kw = self.base_kwargs.copy()
        kw.update(kwargs)
        
        try:
            sim = simG21(**kw)
            sim.run()     
                
            tps = sim.turning_points            
            
        except:         # most likely: no (or too few) turning pts
            #self.warning(None, kwargs)
            return -np.inf, self.blank_blob
            
        # Apply measurement priors now that we have the turning points
        for key in self.logprior_M.priors:
        
            mi, ma = self.logprior_M.priors[key]
        
            i = key.rfind('_')
            key_pre = key[0:i]
            pt = key[i+1:]
            
            j = 0 if key_pre == 'z' else 1
            
            if pt not in tps:
                return -np.inf, self.blank_blob

            if mi <= tps[pt][j] <= ma:
                continue
            else:
                return -np.inf, self.blank_blob
                        
        # Apply priors on derived fields (e.g., CMB optical depth)
        # (not yet implemented)
        #for key in self.logprior_DQ.priors:
        #
        #    mi, ma = self.logprior_DQ.priors[key]
        #
        #    i = key.rfind('_')
        #    key_pre = key[0:i]
        #    pt = key[i+1:]
        #    
        #    j = 0 if key_pre == 'z' else 1
        #    
        #    if mi <= tps[pt][j] <= ma:
        #        continue
        #    else:
        #        return -np.inf, self.blank_blob
                        
        # Compute the likelihood if we've made it this far
        xarr = []
                    
        # Convert frequencies to redshift, temperatures to K
        for element in self.errmap:
            tp = element[0]
            
            # Models without turning point B, C, or D get thrown out.
            if tp not in tps:
                return -np.inf, self.blank_blob
        
            # Convert positions to redshift
            if element[1] == 0:
                if self.errunits[0] == 'redshift':
                    xarr.append(tps[tp][0])
                elif self.errunits[0] in ['mhz', 'MHz']:
                    xarr.append(nu_0_mhz / (1. + tps[tp][0]))
                else:
                    raise ValueError('Unrecognized redshift/time/freq unit %s.' \
                        % self.errunits[0])
            # Convert temperatures to milli-Kelvin        
            elif element[1] == 1:
                if self.errunits[1] in ['mk', 'mK']:
                    xarr.append(tps[tp][1])
                elif self.errunits[1] == 'K':
                    xarr.append(tps[tp][1] / 1e3)     
                else:
                    raise ValueError('Unrecognized temperature unit %s.' \
                        % self.errunits[1])
                   
        # Values of current model that correspond to mu vector
        xarr = np.array(xarr)           
                   
        # Compute log-likelihood, including prior knowledge

        if self.error_type == 'idealized':
            logL = lp \
                - np.sum((xarr - self.mu)**2 / 2. / self.errors**2)

        else:
            
            try:
                dist, loc = nearest(xarr, self.chain)
                
                Lact = np.sum((xarr - np.array([30.,20.,12.,-5,-100,25]))**2 \
                    / 2. / np.array([0.5,0.5,0.5,5.,5.,5.])**2)
                    
                L1D = np.sum((xarr - self.mu)**2 \
                    / 2. / self.sigma**2)
                                
                if loc >= self.logL_size:
                    logL = -np.inf
                else:
                    logL = lp -L1D#+ self.logL[loc]
            except ValueError:
                return -np.inf, self.blank_blob

        # Blobs!
        if hasattr(sim, 'blobs'):
            blobs = sim.blobs
        else:
            blobs = self.blank_blob
                        
        if blobs.shape != self.blank_blob.shape:
            raise ValueError('help')    
            
        return logL, blobs
        
    def warning(self, tps, kwargs):
        print "\n----------------------------------"
        print "WARNING: Error w/ following model:"
        print "----------------------------------"
        for kw in kwargs:
            print '%s = %.4g' % (kw, kwargs[kw])
        print "----------------------------------"
        
        if tps is not None:
            print "# of turning points: %i" % len(tps)    
            print "----------------------------------\n"    
        else:
            print ""
        
class DummySampler:
    def __init__(self):
        pass        

class ModelFit:
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
            
        return self._measurement_map

    @measurement_map.setter
    def measurement_map(self, value):
         self._measurement_map = value     
            
    @property
    def measurement_units(self):
        """
        A two-element tuple containing units assumed for measurements.
        
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
            if hasattr(self, 'sim'):
                self._mu = []
                for tp, i in self.measurement_map:
                    self._mu.append(self.sim.turning_points[tp][i])
            else:        
                raise ValueError('Must supply mu or reference model!')
            
        return self._mu    
            
    @mu.setter
    def mu(self, value):
        self._mu = value        
            
    def set_error(self, error1d=None, nu=0.68, chain=None, logL=None, 
        cols=None):
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
   
        else:            
            self.chain, self.logL = chain, logL
         
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
            
    def set_input_realization(self, **kwargs):
        """
        Set realization of the global 21-cm signal to be fitted.
        
        Parameters
        ----------

        """
            
        kw = self.base_kwargs.copy()
        kw.update(kwargs)
                
        sim = simG21(**kw)
        sim.run()
        
        self.sim = Global21cm(sim)
                        
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

    def run(self, prefix, steps=1e2, burn=0, clobber=False, save_freq=10):
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
    
        """
        
        assert len(self.error) == len(self.measurement_map)
        
        self.prefix = prefix
        
        if os.path.exists('%s.chain.pkl' % prefix) and (not clobber):
            raise IOError('%s exists! Remove manually or set clobber=True.' 
                % prefix)

        print_fit(self, steps=steps, burn=burn)

        if not hasattr(self, 'chain'):
            self.chain = None
            self.logL = None
            
        self.loglikelihood = loglikelihood(steps, self.parameters, self.is_log, 
            self.mu, self.error, self.base_kwargs,
            self.nwalkers, self.priors, self.chain, self.logL, 
            self.measurement_map, self.measurement_units)
            
        if size > 1:
            self.pool = MPIPool()
            if not self.pool.is_master():
                self.pool.wait()
                sys.exit(0)
        else:
            self.pool = None
            
        self.sampler = emcee.EnsembleSampler(self.nwalkers,
            self.Nd, self.loglikelihood, pool=self.pool)
        
        if burn > 0:
            pos, prob, state, blobs = self.sampler.run_mcmc(self.guesses, burn)
            self.sampler.reset()
            if rank == 0:
                print "Burn-in complete."
        else:
            pos = self.guesses
            state = None
            
        # Setup output file
        if rank == 0:

            # Main output: MCMC chains (flattened)
            f = open('%s.chain.pkl' % prefix, 'wb')
            f.close()
            
            # Main output: log-likelihood
            f = open('%s.logL.pkl' % prefix, 'wb')
            f.close()
            
            # Parameter names and list saying whether they are log10 or not
            f = open('%s.pinfo.pkl' % prefix, 'wb')
            pickle.dump((self.parameters, self.is_log), f)
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
            iterations=steps, rstate0=state):
            
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
                    
                pickle.dump(data[i], open(fn, 'ab'))
                f.close()
                        
            pos_all = []; prob_all = []; blobs_all = []
                        
        if self.pool is not None:
            self.pool.close()
    
