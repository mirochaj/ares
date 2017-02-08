"""

MCMC.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Oct 23 19:02:38 PDT 2015

Description: 

"""

import pickle
import numpy as np
from .PriorSet import PriorSet
from ..util.Stats import get_nu
from ..util.MPIPool import MPIPool
from ..util.PrintInfo import print_fit
from ..physics.Constants import nu_0_mhz
from ..util.ParameterFile import par_info
import gc, os, sys, copy, types, time, re
from ..analysis import Global21cm as anlG21
from types import FunctionType, InstanceType
from ..analysis.BlobFactory import BlobFactory
from ..analysis.TurningPoints import TurningPoints
from ..analysis.InlineAnalysis import InlineAnalysis
from ..util.Stats import Gauss1D, GaussND, rebin, get_nu
from ..util.SetDefaultParameterValues import _blob_names, _blob_redshifts
from ..util.ReadData import flatten_chain, flatten_logL, flatten_blobs, \
    read_pickled_chain

try:
    import emcee
    from emcee.utils import sample_ball
except ImportError:
    pass

emcee_mpipool = False
     
try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1
    
sqrt_twopi = np.sqrt(2 * np.pi)
    
guesses_shape_err = "If you supply guesses as 2-D array, it must have" 
guesses_shape_err += " shape (nwalkers, nparameters)!"

jitter_shape_err = "If you supply jitter as an array, it must have"
jitter_shape_err += " shape (nparameters)"

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
        
def guesses_from_priors(pars, prior_set, nwalkers):
    """
    Generate initial position vectors for nwalkers.

    Parameters
    ----------
    pars : list 
        Names of parameters
    prior_set : PriorSet object

    nwalkers : int
        Number of walkers

    """

    guesses = []
    for i in range(nwalkers):
        draw = prior_set.draw()
        guesses.append([draw[pars[ipar]] for ipar in range(len(pars))])

    return np.array(guesses)    
        
class LogLikelihood(object):
    def __init__(self, xdata, ydata, error, parameters, is_log,
        base_kwargs, param_prior_set=None, blob_prior_set=None,
        prefix=None, blob_info=None, checkpoint_by_proc=True, timeout=None):
        """
        This is only to be inherited by another log-likelihood class.

        Parameters
        ----------

        """

        self.parameters = parameters # important that they are in order?
        self.is_log = is_log
        self.checkpoint_by_proc = checkpoint_by_proc

        self.base_kwargs = base_kwargs
        self.timeout = timeout

        if blob_info is not None:
            self.blob_names = blob_info['blob_names']
            self.blob_ivars = blob_info['blob_ivars']
            self.blob_funcs = blob_info['blob_funcs']
            self.blob_nd = blob_info['blob_nd']
            self.blob_dims = blob_info['blob_dims']
        else:
            self.blob_names = self.blob_ivars = self.blob_funcs = \
                self.blob_nd = self.blob_dims = None

        ##
        # Note: you might think using np.array is innocuous, but we have to be
        # a little careful below since sometimes xdata, ydata, etc. are
        # masked arrays, and casting them to regular arrays screws things up.
        ##
        
        # Not flat
        if type(xdata) is list:
            self.xdata = np.array(xdata)
        else:
            self.xdata = xdata
        
        # Flat
        if type(ydata) is list:
            self.ydata = np.array(ydata)
        else:
            self.ydata = ydata
        
        if type(error) is list:
            self.error = np.array(error)
        else:
            self.error = error
        
        self.prefix = prefix

        self.priors_P = param_prior_set
        if len(self.priors_P.params) != len(self.parameters):
            raise ValueError("The number of parameters of the priors given " +\
                             "to a loglikelihood object is not equal to " +\
                             "the number of parameters given to the object.")
        
        if blob_info is None:
            self.priors_B = PriorSet()
        elif isinstance(blob_prior_set, PriorSet):
            self.priors_B = blob_prior_set
        else:
            try:
                # perhaps the prior tuples
                # (prior, params, transformations) were given
                self.priors_B = PriorSet(prior_tuples=blob_prior_set)
            except:
                raise ValueError("The value given as the blob_prior_set " +\
                                 "argument to the initializer of a " +\
                                 "Loglikelihood could not be cast " +\
                                 "into a PriorSet.")


    def _compute_blob_prior(self, sim):
        blob_vals = {}
        for key in self.priors_B.params:

            grp, i, nd, dims = sim.blob_info(key)
            
            #if nd == 0:
            #    blob_vals[key] = sim.get_blob(key)
            #elif nd == 1:    
            blob_vals[key] = sim.get_blob(key)
            #else:
            #    raise NotImplementedError('help')

        try:
            # will return 0 if there are no blobs
            return self.priors_B.log_prior(blob_vals)
        except:
            # some of the blobs were not retrieved (then they are Nones)!
            return -np.inf
    
    @property
    def blank_blob(self):
        if not hasattr(self, '_blank_blob'):
            
            if self.blob_names is None:
                self._blank_blob = []
                return []
    
            self._blank_blob = []
            for i, group in enumerate(self.blob_names):
                if self.blob_ivars[i] is None:
                    self._blank_blob.append([np.inf] * len(group))
                else:
                    if self.blob_nd[i] == 0:
                        self._blank_blob.append([np.inf] * len(group))
                    elif self.blob_nd[i] == 1:
                        arr = np.ones([len(group), self.blob_dims[i][0]])
                        self._blank_blob.append(arr * np.inf)
                    elif self.blob_nd[i] == 2:
                        dims = len(group), self.blob_dims[i][0], \
                            self.blob_dims[i][1]
                        arr = np.ones(dims)
                        self._blank_blob.append(arr * np.inf)
    
        return self._blank_blob
        
    def checkpoint(self, **kwargs):
        if self.checkpoint_by_proc:
            procid = str(rank).zfill(3)
            fn = '%s.%s.checkpt.pkl' % (self.prefix, procid)
            with open(fn, 'wb') as f:
                pickle.dump(kwargs, f)
            
            fn = '%s.%s.checkpt.txt' % (self.prefix, procid)
            with open(fn, 'w') as f:
                print >> f, "Simulation began: %s" % time.ctime()
            
    def checkpoint_on_completion(self, **kwargs):
        if self.checkpoint_by_proc:
            fn = '%s.%s.checkpt.txt' % (self.prefix, procid)
            with open(fn, 'a') as f:
                print >> f, "Simulation finished: %s" % time.ctime() 
            
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
    def info(self):
        print_fit(self)      
    
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
    def seed(self):
        if not hasattr(self, '_seed'):
            self._seed = None

        return self._seed
        
    @seed.setter
    def seed(self, value):
        if rank > 0:
            return
        
        self._seed = value

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
        
    @ydata.setter
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
    def prior_set_P(self):
        if not hasattr(self, '_prior_set_P'):
            ps = self.prior_set
            subset = PriorSet()
            for (prior, params, transforms) in ps._data:
                are_pars_P = [(par in self.parameters) for par in params]
                are_pars_P = np.array(are_pars_P)

                if np.all(are_pars_P):
                    subset.add_prior(prior, params, transforms)
                elif not np.all(are_pars_P == False):
                    raise AttributeError("Blob priors and parameter " +\
                                         "priors are coupled!")
            self._prior_set_P = subset
        return self._prior_set_P
    
    @property
    def prior_set_B(self):
        if not hasattr(self, '_prior_set_B'):
            ps = self.prior_set
            subset = PriorSet()
            for (prior, params, transforms) in ps._data:
                are_pars_B = [(par in self.all_blob_names) for par in params]
                are_pars_B = np.array(are_pars_B)
                
                if np.all(are_pars_B):
                    subset.add_prior(prior, params, transforms)
                elif not np.all(are_pars_B == False):
                    raise AttributeError("Blob priors and parameter " +\
                                         "priors are coupled!")
            self._prior_set_B = subset
        return self._prior_set_B

    @property
    def prior_set(self):
        if not hasattr(self, '_prior_set'):
            raise ValueError('Must set prior_set by hand!')

        return self._prior_set

    @prior_set.setter
    def prior_set(self, value):
        if isinstance(value, PriorSet):
            # could do more error catching but it would enforce certain
            # attributes being set before others which could complicate things
            self._prior_set = value

            # Warn user if a prior has no match in parameters or blobs
            for param in self._prior_set.params:
                if param in self.parameters:
                    continue
                if param in self.all_blob_names:
                    continue
                
                warn = ("Setting prior on %s but %s " % (param, param,)) +\
                        "not in parameters or blobs!"
            
                if size == 1:
                    raise KeyError(warn)
                else:
                    print warn
                    MPI.COMM_WORLD.Abort()
        else:
            try:
                self._prior_set = PriorSet(prior_tuples=value)
            except:
                raise ValueError("The prior_set property was set to " +\
                                 "something which was not a PriorSet " +\
                                 "and could not be cast as a PriorSet.")

    @property
    def nwalkers(self):
        if not hasattr(self, '_nw'):
            self._nw = self.Nd * 2
            
            if rank == 0:
                print "Defaulting to nwalkers=2*Nd=%i." % self._nw
            
        return self._nw
        
    @nwalkers.setter
    def nwalkers(self, value):
        self._nw = int(value)
        
    def _handler(self, signum, frame):
        raise RuntimeError('timeout!')
            
    @property
    def timeout(self):
        if not hasattr(self, '_timeout'):
            self._timeout = None
        return self._timeout
    
    @timeout.setter
    def timeout(self, value):
        self._timeout = int(value)
        
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

        if hasattr(self, '_guesses'):
            return self._guesses

        # Set using priors
        if (not hasattr(self, '_guesses')) and hasattr(self, '_prior_set'):            
            self._guesses = guesses_from_priors(self.parameters, 
                self.prior_set, self.nwalkers)
        else:
            raise AttributeError('Must set guesses or prior_set by hand!')
                     
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
        
        if type(value) is dict:
            guesses_tmp = np.array([value[par] for par in self.parameters])
        else:    
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
                        
    # I don't know how to integrate this using the new prior system
    # Can you help, Jordan?
    #
    #def _fix_guesses(self, pos):
    #    
    #    if rank > 0:
    #        return
    #    
    #    guesses = pos.copy()
    #    
    #    # Fix parameters whose values lie outside prior space
    #    for i, par in enumerate(self.parameters):
    #        if par not in self.priors:
    #            continue
    #            
    #        if self.priors[par][0] != 'uniform':
    #            continue
    #        
    #        mi, ma = self.priors[par][1:]
    #        
    #        ok_lo = guesses[:,i] >= mi
    #        ok_hi = guesses[:,i] <= ma
    #        
    #        if np.all(ok_lo) and np.all(ok_hi):
    #            continue
    #            
    #        # Draw from uniform distribution for failed cases
    #        
    #        not_ok_lo = np.logical_not(ok_lo)
    #        not_ok_hi = np.logical_not(ok_hi)
    #        not_ok = np.logical_or(not_ok_hi, not_ok_lo)
    #        
    #        bad_mask = np.argwhere(not_ok)
    #        
    #        for j in bad_mask:
    #            #print "Fixing guess for walker %i parameter %s" % (j[0], par)
    #            guesses[j[0],i] = np.random.uniform(mi, ma)
    #            
    #    return guesses
        
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
            pos = self._prep_from_restart(restart)
        else:
            pos = None
            self._prep_from_scratch(clobber)    
    
        return pos
    
    def _prep_from_restart(self, restart):

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
          
        # Identical to setup, just easier for scp'ing *info.pkl files.
        if os.path.exists('%s.binfo.pkl' % prefix):
            f = open('%s.binfo.pkl' % prefix, 'rb')
            base_kwargs = pickle.load(f)
            f.close()
        else:
            # Deprecate this eventually          
            f = open('%s.setup.pkl' % prefix, 'rb')
            base_kwargs = pickle.load(f)
            f.close()
        
        f = open('%s.rinfo.pkl' % self.prefix, 'r')
        nwalkers, save_freq, steps = pickle.load(f)
        f.close()
        
        # These things CANNOT change on restart
        assert nwalkers == self.nwalkers
        assert save_freq == self.save_freq
        
        #for kwarg in base_kwargs:
        #    if type(base_kwargs[kwarg])
        #if base_kwargs != self.base_kwargs:
        #    if size > 1:
        #        if rank == 0:
        #            print 'base_kwargs from file dont match those supplied!'
        #        MPI.COMM_WORLD.Abort()
        #    raise ValueError('base_kwargs from file dont match those supplied!')   
                    
        # Start from last step in pre-restart calculation
        if (not self.checkpoint_append):
            if type(restart) is bool:
            
                ct = 0 
                while True:
                    dd = 'dd' + str(ct).zfill(4)
                    fn = '%s.%s.chain.pkl' % (prefix, dd)
                                        
                    if os.path.exists(fn):
                        ct += 1
                        continue
                    else:
                        break
                        
                # Back up a step to file that exists
                self.ct = ct - 1    

                dd = 'dd' + str(self.ct).zfill(4)
                fn = '%s.%s.chain.pkl' % (prefix, dd)
            else:
                assert type(restart) is str, \
                    "If not True or False, restart must be filename!"
                fn = restart
                self.ct = int(fn.split('.')[-3][2:])
                 
        else:
            fn = '%s.chain.pkl' % prefix
            
        print "Restarting from %s." % fn
        chain = read_pickled_chain(fn)    
           
        (nw, sf) = (self.nwalkers, self.save_freq)
        pos = chain[-(nw-1)*sf-1::sf,:]    
        
        return pos

    @property
    def checkpoint_by_proc(self):
        if not hasattr(self, '_checkpoint_by_proc'):
            self._checkpoint_by_proc = True
        return self._checkpoint_by_proc
        
    @checkpoint_by_proc.setter
    def checkpoint_by_proc(self, value):
        self._checkpoint_by_proc = value
    
    @property
    def checkpoint_append(self):
        if not hasattr(self, '_checkpoint_append'):
            self._checkpoint_append = True
        return self._checkpoint_append
    
    @checkpoint_append.setter
    def checkpoint_append(self, value):
        self._checkpoint_append = value    

    def _prep_from_scratch(self, clobber, by_proc=False):
        if (rank > 0) and (not by_proc):
            return
        
        if by_proc:
            prefix_by_proc = self.prefix + '.%s' % (str(rank).zfill(3))
        else:
            prefix_by_proc = self.prefix
                               
        if clobber:
            # Delete only the files made by this routine. Don't want to risk
            # deleting other files the user may have created with similar
            # naming convention!
            
            for suffix in ['logL', 'facc', 'pinfo', 'rinfo', 'binfo', 'setup', 'prior_set']:
                os.system('rm -f %s.%s.pkl' % (self.prefix, suffix))
            
            os.system('rm -f %s.*.fail.pkl' % self.prefix)
            os.system('rm -f %s.*.chain.*pkl' % self.prefix)
            os.system('rm -f %s.*.blob*.*pkl' % self.prefix)
            
            # Need to potentially axe a product file
            os.system('rm -f %s.fails.pkl' % self.prefix)
            os.system('rm -f %s.chain.pkl' % self.prefix)
                    
        # Each processor gets its own fail file
        f = open('%s.fail.pkl' % prefix_by_proc, 'wb')
        f.close()  
        
        # Main output: MCMC chains (flattened)
        if self.checkpoint_append:
            f = open('%s.chain.pkl' % prefix_by_proc, 'wb')
            f.close()
        
            # Main output: log-likelihood
            f = open('%s.logL.pkl' % self.prefix, 'wb')
            f.close()
        
        # Store acceptance fraction
        f = open('%s.facc.pkl' % self.prefix, 'wb')
        f.close()
        
        # File for blobs themselves
        if self.blob_names is not None and self.checkpoint_append:
            
            for i, group in enumerate(self.blob_names):
                for blob in group:
                    fntup = (prefix_by_proc, self.blob_nd[i], blob)
                    f = open('%s.blob_%id.%s.pkl' % fntup, 'wb')
                    f.close()
        
        # Parameter names and list saying whether they are log10 or not
        f = open('%s.pinfo.pkl' % self.prefix, 'wb')
        pickle.dump((self.parameters, self.is_log), f)
        f.close()
        
        # "Run" info (MCMC only)
        if hasattr(self, 'steps'):
            f = open('%s.rinfo.pkl' % self.prefix, 'wb')
            pickle.dump((self.nwalkers, self.save_freq, self.steps), f)
            f.close()
        
        # Priors!
        if hasattr(self, '_prior_set'):
            f = open('%s.prior_set.pkl' % self.prefix, 'wb')
            pickle.dump(self.prior_set, f)
            f.close()
        
        # Constant parameters being passed to ares.simulations.Global21cm
        f = open('%s.binfo.pkl' % self.prefix, 'wb')
        tmp = self.base_kwargs.copy()
        to_axe = []
        for key in tmp:
            # this might be big, get rid of it
            if re.search('tau_instance', key):
                to_axe.append(key)
            if re.search('tau_table', key):
                to_axe.append(key)
            if re.search('hmf_instance', key):
                to_axe.append(key)
            if re.search('pop_psm_instance', key):
                to_axe.append(key)        
            if re.search('pop_sed_by_Z', key):
                to_axe.append(key)
        
        for key in to_axe:
            tmp[key] = None
            
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
        restart : bool, str
            Append to pre-existing files of the same prefix if one exists?
            Can also supply the filename of the checkpoint from which to 
            restart.
            
        """
                
        self.prefix = prefix

        if os.path.exists('%s.chain.pkl' % prefix) and (not clobber):
            if not restart:
                msg = '%s exists! Remove manually, set clobber=True,' % prefix
                msg += ' or set restart=True to append.' 
                raise IOError(msg)

        if self.checkpoint_append:
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

        self.steps = steps
        self.save_freq = save_freq

        # Initialize sampler
        self.sampler = emcee.EnsembleSampler(self.nwalkers,
            self.Nd, self.loglikelihood, pool=self.pool)
                
        pos = self.prep_output_files(restart, clobber)    
        
        state = None#np.random.RandomState(self.seed)
                        
        # Burn in, prep output files     
        if (burn > 0) and (not restart):
            
            if rank == 0:
                print "Starting burn-in: %s" % (time.ctime())
            
            t1 = time.time()
            pos, prob, state, blobs = \
                self.sampler.run_mcmc(self.guesses, burn, rstate0=state)
            self.sampler.reset()
            t2 = time.time()

            if rank == 0:
                print "Burn-in complete in %.3g seconds." % (t2 - t1)

            # Find maximum likelihood point
            mlpt = pos[np.argmax(prob)]

            pos = sample_ball(mlpt, np.std(pos, axis=0), size=self.nwalkers)
            #pos = self._fix_guesses(pos)
            
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
        
        # Need to make sure we don't overwrite previous outputs in this case    
        if restart and (not self.checkpoint_append):
            ct = (self.ct + 1) * save_freq
        else:
            ct = 0
                        
        # Take steps, append to pickle file every save_freq steps
        pos_all = []; prob_all = []; blobs_all = []
        for pos, prob, state, blobs in self.sampler.sample(pos, 
            iterations=steps, rstate0=state, storechain=False):
            
            # Only the rank 0 processor ever makes it here
            
            # If we're saving each checkpoint to its own file, this is the
            # identifier to use in the filename
            dd = 'dd' + str(ct / save_freq).zfill(4)
            
            # Increment counter
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

            # The flattened version of pos_all has 
            # shape = (save_freq * nwalkers, ndim)

            for i, suffix in enumerate(['chain', 'logL', 'blobs']):

                if self.checkpoint_append:
                    mode = 'ab'
                else:
                    mode = 'wb'
                    
                # Blobs
                if suffix == 'blobs':
                    if self.blob_names is None:
                        continue
                    self.save_blobs(data[i], dd=dd)
                # Other stuff
                else:
                    if self.checkpoint_append:
                        fn = '%s.%s.pkl' % (prefix, suffix)
                    else:
                        fn = '%s.%s.%s.pkl' % (prefix, dd, suffix)
                    with open(fn, mode) as f:
                        pickle.dump(data[i], f)
                    
            # This is a running total already so just save the end result 
            # for this set of steps
            f = open('%s.facc.pkl' % prefix, 'ab')
            pickle.dump(self.sampler.acceptance_fraction, f)
            f.close()

            if self.checkpoint_append:
                print "Checkpoint #%i: %s" % (ct / save_freq, time.ctime())
            else:
                print "Wrote %s.%s.*.pkl: %s" % (prefix, dd, time.ctime())

            del data, pos_all, prob_all, blobs_all
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
    
    def save_blobs(self, blobs, uncompress=True, prefix=None, dd=None):
        """
        Write blobs to disk.
        
        Parameters
        ----------
        blobs : list

        uncompress : bool
            True for MCMC, False for model grids.
        """
        
        if prefix is None:
            prefix = self.prefix
                
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
        
        if self.blob_names is None:
            return

        for j, group in enumerate(self.blob_names):
            for k, blob in enumerate(group):
                to_write = []
                for l in range(self.nwalkers * blen):  
                    # indices: walkers*steps, blob group, blob
                    barr = blobs_now[l][j][k]
                    to_write.append(barr)   

                if self.checkpoint_append:
                    mode = 'ab'
                    bfn = '%s.blob_%id.%s.pkl' \
                        % (prefix, self.blob_nd[j], blob)
                else:
                    mode = 'wb'
                    bfn = '%s.%s.blob_%id.%s.pkl' \
                        % (prefix, dd, self.blob_nd[j], blob)        
                    
                    assert dd is not None, "checkpoint_append=False but no DDID!"        
                            
                with open(bfn, mode) as f:
                    pickle.dump(np.array(to_write), f) 
                    
                       
