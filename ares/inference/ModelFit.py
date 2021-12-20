"""

MCMC.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Oct 23 19:02:38 PDT 2015

Description:

"""

from __future__ import print_function

import glob
import numpy as np
from ..util import get_hash
from ..util.MPIPool import MPIPool
from ..physics.Constants import nu_0_mhz
from ..util.Warnings import not_a_restart
from ..util.ParameterFile import par_info
import gc, os, sys, copy, types, time, re, glob
from ..analysis import Global21cm as anlG21
from types import FunctionType#, InstanceType # InstanceType not in Python3
from ..analysis import ModelSet
from ..analysis.BlobFactory import BlobFactory
from ..sources import BlackHole, SynthesisModel
from ..analysis.TurningPoints import TurningPoints
from ..util.Stats import Gauss1D, get_nu, bin_e2c
from ..util.Pickling import read_pickle_file, write_pickle_file
from ..util.SetDefaultParameterValues import _blob_names, _blob_redshifts
from ..util.ReadData import flatten_chain, flatten_logL, flatten_blobs, \
    read_pickled_chain, read_pickled_logL

#import psutil
#
#ps = psutil.Process(os.getpid())
#
#def write_memory(checkpt):
#    t = time.time()
#    #mem = psutil.virtual_memory().active / 1e9
#    mem = ps.memory_info().rss / 1e6
#
#    with open('memory.txt', 'a') as f:
#        f.write("{} {} {} {}\n".format(t, mem, rank, checkpt))

try:
    from distpy.distribution import DistributionSet
except ImportError:
    pass

try:
    import emcee
    from emcee.utils import sample_ball
    emcee_v = int(emcee.__version__[0])

    if emcee_v >= 3:
        print("# WARNING: ARES not fully emcee version >= 3 compatible!")
        print("#          emcee 2.2.1 is a safe bet for now.")

except ImportError:
    emcee_v = None


emcee_mpipool = False

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1

try:
    import multiprocessing
except ImportError:
    pass

sqrt_twopi = np.sqrt(2 * np.pi)

guesses_shape_err = "If you supply guesses as 2-D array, it must have"
guesses_shape_err += " shape (nwalkers, nparameters)!"

jitter_shape_err = "If you supply jitter as an array, it must have"
jitter_shape_err += " shape (nparameters)"

def_kwargs = {'verbose': False, 'progress_bar': False}

def checkpoint(prefix, is_blobs=False, checkpoint_by_proc=True,
    **kwargs):
    if checkpoint_by_proc:
        procid = str(rank).zfill(3)

        # Save parameters
        if not is_blobs:
            fn = '{0!s}.{1!s}.checkpt.pkl'.format(prefix, procid)
            write_pickle_file(kwargs, fn, ndumps=1, open_mode='w',
                safe_mode=False, verbose=False)

        # Timing info
        fn = '{0!s}.{1!s}.checkpt.txt'.format(prefix, procid)
        with open(fn, 'a' if is_blobs else 'w') as f:
            if is_blobs:
                print("Generating blobs: {!s}".format(time.ctime()), file=f)
            else:
                print("Simulation began: {!s}".format(time.ctime()), file=f)

def checkpoint_on_completion(prefix, is_blobs=False, checkpoint_by_proc=True,
    **kwargs):
    if checkpoint_by_proc:
        procid = str(rank).zfill(3)
        fn = '{0!s}.{1!s}.checkpt.txt'.format(prefix, procid)
        with open(fn, 'a') as f:
            if is_blobs:
                print("Blobs generated    : {!s}".format(time.ctime()), file=f)
            else:
                print("Simulation finished: {!s}".format(time.ctime()), file=f)


def _compute_blob_prior(sim, priors_B):

    like = 0.0

    blob_vals = {}
    for k, key in enumerate(priors_B.params):

        func, names, trans = priors_B._data[k]

        group_num, element, nd, dims = sim.blob_info(key)

        blob = sim.get_blob(key)

        # Need to figure out independent variable
        if nd == 0:
            val = float(blob)
        elif nd == 1:
            # Should be ('ivar', val) pair
            md = func.metadata

            ivar = sim.get_ivars(key)[0]

            # Interpolate to find value of blob at point where prior is set
            val = np.interp(md[1], ivar, blob)
            #zclose = ivar[np.argmin(np.abs(ivar - md[1]))]

            # Should check ivarn too just to be safe

        else:
            raise NotImplemented('sorry')

        like += np.exp(func(val))

    return np.log(like)

def loglikelihood(pars, prefix, parameters, is_log, prior_set_P, prior_set_B,
    blank_blob, base_kwargs, checkpoint_by_proc, simulator, fitters, debug):

    #write_memory('1')

    kwargs = {}
    for i, par in enumerate(parameters):

        if is_log[i]:
            kwargs[par] = 10**pars[i]
        else:
            kwargs[par] = pars[i]

    # Apply prior on model parameters first (dont need to generate model)
    point = {}
    for i, par in enumerate(parameters):
        # This is in user-supplied units, i.e., don't correct for log10-ness
        # because the distribution set and supplied values are
        # consistent by construction
        point[par] = pars[i]

    lp = prior_set_P.log_value(point)

    if not np.isfinite(lp):
        return -np.inf, blank_blob

    # Update kwargs
    kw = base_kwargs.copy()
    kw.update(kwargs)

    # Don't save base_kwargs for each proc! Needlessly expensive I/O-wise.
    checkpoint(prefix, False, checkpoint_by_proc, **kwargs)

    #for i, par in enumerate(self.parameters):
    #    print(rank, par, pars[i], kwargs[par])

    t1 = time.time()
    sim = simulator(**kw)

    if debug:
        print("# Processor {} simulation starting: {}".format(rank, time.ctime()))
        sim.run()
        print("# Processor {} simulation complete: {}".format(rank, time.ctime()))
        blobs = sim.blobs
        print("# Processor {} generated blobs    : {}".format(rank, time.ctime()))

    try:
        if not debug:
            sim.run()
            blobs = copy.deepcopy(sim.blobs)
    except ValueError:
        print('FAILURE: ', kwargs)
        del sim, kw, kwargs
        gc.collect()
        return -np.inf, blank_blob

    t2 = time.time()

    #write_memory('2')

    checkpoint_on_completion(prefix, False, checkpoint_by_proc, **kwargs)

    lnL = 0.0
    for fitter in fitters:
        lnL += fitter.loglikelihood(sim)

    # Blob prior: only compute if log-likelihood is finite
    if np.isfinite(lnL):

        ##
        # Blobs
        checkpoint(prefix, True, checkpoint_by_proc, **kwargs)

        try:
            blobs = sim.blobs
        except:
            print("WARNING: Failure to generate blobs.")
            blobs = blank_blob

        checkpoint_on_completion(prefix, True, checkpoint_by_proc, **kwargs)
        ##
        #

        if (prior_set_B.params != []):
            lp += _compute_blob_prior(sim, prior_set_B)

    else:
        blobs = blank_blob

    # Final posterior calculation
    PofD = lp + lnL

    # emcee doesn't like nans, but -inf is OK (see below)
    if np.isnan(PofD) or isinstance(PofD, np.ma.core.MaskedConstant):
        del sim, kw, kwargs
        gc.collect()
        return -np.inf, blank_blob

    # Remember, -np.inf is OK (means proposal was bad, probably).
    # +inf is NOT OK! Something is horribly wrong. Helpful in debugging.
    if PofD == np.inf:
        raise ValueError('+inf obtained in likelihood. Should not happen!')

    #write_memory('3')

    del sim, kw, kwargs
    gc.collect()

    #write_memory('4')

    return PofD, blobs

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

    return pvals[pars.index('{0!s}{{{1}}}'.format(prefix, num))]

def guesses_from_priors(pars, prior_set, nwalkers):
    """
    Generate initial position vectors for nwalkers.

    Parameters
    ----------
    pars : list
        Names of parameters
    prior_set : DistributionSet object

    nwalkers : int
        Number of walkers

    """
    print("# Making guesses from prior_set...")
    guesses = []
    for i in range(nwalkers):
        draw = prior_set.draw()
        guesses.append([draw[pars[ipar]] for ipar in range(len(pars))])

    return np.array(guesses)

class LogLikelihood(object):
    def __init__(self, xdata, ydata, error):
        """
        This is only to be inherited by another log-likelihood class.

        Parameters
        ----------

        """

        #self.parameters = parameters # important that they are in order?
        #self.is_log = is_log
        #self.checkpoint_by_proc = checkpoint_by_proc
        #
        #self.base_kwargs = base_kwargs
        #self.timeout = timeout
        #
        #if blob_info is not None:
        #    self.blob_names = blob_info['blob_names']
        #    self.blob_ivars = blob_info['blob_ivars']
        #    self.blob_funcs = blob_info['blob_funcs']
        #    self.blob_nd = blob_info['blob_nd']
        #    self.blob_dims = blob_info['blob_dims']
        #else:
        #    self.blob_names = self.blob_ivars = self.blob_funcs = \
        #        self.blob_nd = self.blob_dims = None

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

        #self.prefix = prefix

        #self.priors_P = param_prior_set
        #if len(self.priors_P.params) != len(self.parameters):
        #    raise ValueError(("The number of parameters of the priors " +\
        #        "given to a loglikelihood object ({0}) is not equal to the " +\
        #        "number of parameters given to the object ({1}).").format(\
        #        len(self.priors_P.params), len(self.parameters)))
        #
        #if blob_info is None:
        #    self.priors_B = DistributionSet()
        #elif isinstance(blob_prior_set, DistributionSet):
        #    self.priors_B = blob_prior_set
        #else:
        #    try:
        #        # perhaps the prior tuples
        #        # (prior, params, transformations) were given
        #        self.priors_B =\
        #            DistributionSet(distribution_tuples=blob_prior_set)
        #    except:
        #        raise ValueError("The value given as the blob_prior_set " +\
        #                         "argument to the initializer of a " +\
        #                         "Loglikelihood could not be cast " +\
        #                         "into a DistributionSet.")


    @property
    def const_term(self):
        if not hasattr(self, '_const_term'):
            self._const_term = -np.log(np.sqrt(2. * np.pi)) \
                             -  np.sum(np.log(self.error))
        return self._const_term

class FitBase(BlobFactory):
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
        pass
        #print("No information at this time.")

    @property
    def pf(self):
        if not hasattr(self, '_pf'):
            self._pf = self.base_kwargs
        return self._pf

    @pf.setter
    def pf(self, value):
        self._pf = value

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
            assert len(value) == self.Nd
            self._is_log = value

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


class ModelFit(FitBase):

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
    def save_hmf(self):
        if not hasattr(self, '_save_hmf'):
            self._save_hmf = True
        return self._save_hmf

    @save_hmf.setter
    def save_hmf(self, value):
        self._save_hmf = value

    @property
    def save_hist(self):
        if not hasattr(self, '_save_hist'):
            self._save_hist = False
        return self._save_hist

    @save_hist.setter
    def save_hist(self, value):
        self._save_hist = value

    @property
    def save_src(self):
        if not hasattr(self, '_save_src'):
            # False by default since this is a little riskier in that we
            # *can* vary SED parameters, but haven't yet implemented HMF
            # variations.
            self._save_src = False
        return self._save_src

    @save_src.setter
    def save_src(self, value):
        self._save_src = value

    @property
    def prior_set_P(self):
        if not hasattr(self, '_prior_set_P'):
            ps = self.prior_set
            subset = DistributionSet()
            for (prior, params, transforms) in ps._data:
                are_pars_P = [(par in self.parameters) for par in params]
                are_pars_P = np.array(are_pars_P)
                if np.all(are_pars_P):
                    subset.add_distribution(prior, params, transforms)
                elif not np.all(are_pars_P == False):
                    raise AttributeError("Blob priors and parameter " +\
                                         "priors are coupled!")
            self._prior_set_P = subset
        return self._prior_set_P

    @property
    def prior_set_B(self):
        if not hasattr(self, '_prior_set_B'):
            ps = self.prior_set
            subset = DistributionSet()
            for (prior, params, transforms) in ps._data:
                are_pars_B = [(par in self.all_blob_names) for par in params]
                are_pars_B = np.array(are_pars_B)

                if np.all(are_pars_B):
                    subset.add_distribution(prior, params, transforms)
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
        if isinstance(value, DistributionSet):
            # could do more error catching but it would enforce certain
            # attributes being set before others which could complicate things
            self._prior_set = value

            # Warn user if a prior has no match in parameters or blobs
            for param in self._prior_set.params:
                if param in self.parameters:
                    continue
                if param in self.all_blob_names:
                    continue

                warn = ("Setting prior on {0!s} but {1!s} not in " +\
                    "parameters or blobs!").format(param, param)

                if size == 1:
                    raise KeyError(warn)
                else:
                    print(warn)
                    MPI.COMM_WORLD.Abort()
        else:
            try:
                self._prior_set = DistributionSet(distribution_tuples=value)
            except:
                err_msg = "The prior_set property was set to something " +\
                          "which was not a DistributionSet and could not " +\
                          "be cast as a DistributionSet."
                if size == 1:
                    raise ValueError(err_msg)
                else:
                    print(err_msg)
                    MPI.COMM_WORLD.Abort()

    @property
    def guesses_prior_set(self):
        """
        The DistributionSet object which is used to initialize the walkers. If
        self.guesses_prior_set is set, it will be used here. Otherwise, the
        same DistributionSet which will be used in the likelihood calculation
        will be used to initialize the walkers.
        """
        if hasattr(self, '_guesses_prior_set'):
            return self._guesses_prior_set
        else:
            return self.prior_set_P

    @guesses_prior_set.setter
    def guesses_prior_set(self, value):
        """
        A setter for the PriorSet which will be used to initialize the walkers.
        This attribute is optional. If it is not set, the DistributionSet which
        is used in the likelihood calculation is used to initialize the
        walkers.

        value a DistributionSet object
        """
        if isinstance(value, DistributionSet):
            self._guesses_prior_set = value

            for param in self._guesses_prior_set.params:
                if param in self.parameters:
                    continue
                warn = ("Setting prior on {0!s} but {1!s} not in " +\
                    "parameters!").format(param, param)
                if size == 1:
                    raise KeyError(warn)
                else:
                    print(warn)
                    MPI.COMM_WORLD.Abort()
        else:
            try:
                self._guesses_prior_set =\
                    DistributionSet(distribution_tuples=value)
            except:
                err_msg = "The guesses_prior_set property was set to " +\
                          "something which was not a DistributionSet and " +\
                          "could not be cast as a DistributionSet."
                if size == 1:
                    raise ValueError(err_msg)
                else:
                    print(err_msg)
                    MPI.COMM_WORLD.Abort()

    @property
    def nwalkers(self):
        if not hasattr(self, '_nw'):
            self._nw = self.Nd * 2

            if rank == 0:
                print("Defaulting to nwalkers=2*Nd={}.".format(self._nw))

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
        # Set using priors
        if (not hasattr(self, '_guesses')):
            if hasattr(self, '_prior_set'):
                self._guesses = guesses_from_priors(self.parameters,
                    self.guesses_prior_set, self.nwalkers)
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

        # Make sure we haven't jittered our way outside the prior.
        self._guesses = self._fix_guesses(self._guesses)

    def _adjust_guesses(self, pos, mlpt):
        if rank > 0:
            return
        std = np.std(pos, axis=0)
        pos = sample_ball(mlpt, std, size=self.nwalkers)
        newpos = pos.copy()
        def is_outside_prior(guess_list):
            this_guess_dict = {}
            for iparam in range(len(self.parameters)):
                this_guess_dict[self.parameters[iparam]] = guess_list[iparam]
            return not np.isfinite(self.prior_set_P.log_value(this_guess_dict))
        for iguess in range(newpos.shape[0]):
            while is_outside_prior(newpos[iguess]):
                newpos[iguess] = sample_ball(mlpt, std, size=1)[0]
        return newpos

    def _fix_guesses(self, pos):
        """
        If any walkers have been re-initialized to locations outside the
        prior range, push them back into the acceptable range.
        """

        if rank > 0:
            return

        guesses = pos.copy()

        # Fix parameters whose values lie outside prior space
        for i, par in enumerate(self.parameters):
            if par not in self.prior_set_P.params:
                continue

            mi, ma = self.prior_set_P.bounds[par]

            ok_lo = guesses[:,i] >= mi
            ok_hi = guesses[:,i] <= ma

            if np.all(ok_lo) and np.all(ok_hi):
                continue

            # Draw from uniform distribution for failed cases
            not_ok_lo = np.logical_not(ok_lo)
            not_ok_hi = np.logical_not(ok_hi)
            not_ok = np.logical_or(not_ok_hi, not_ok_lo)

            bad_mask = np.argwhere(not_ok)
            ok = np.logical_not(not_ok)

            for j in bad_mask:
                print("# Fixing position of walker {0} (parameter {1!s})".format(
                    j[0], par))
                new = np.random.choice(guesses[ok==1,i])
                print("# Moved from {} to {}".format(guesses[j,i][0], new))
                guesses[j[0],i] = new

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
            assert (len(value) == len(self.parameters)), 'jitter has the wrong shape!'

            self._jitter = np.array(value)

    def prep_output_files(self, restart, clobber, prefix_burn=None):
        if restart:
            pos = self._prep_from_restart(prefix_burn)
        else:
            pos = None
            self._prep_from_scratch(clobber)

        return pos

    def _prep_from_restart(self, prefix_restart=None):

        #if prefix is None:
        prefix = self.prefix

        if prefix_restart is None:
            prefix_restart = prefix

        (pars, is_log) = read_pickle_file('{!s}.pinfo.pkl'.format(prefix),\
            nloads=1, verbose=False)

        if pars != self.parameters:
            if size > 1:
                if rank == 0:
                    print('parameters from file dont match those supplied!')
                MPI.COMM_WORLD.Abort()
            raise ValueError('parameters from file dont match those supplied!')
        if is_log != self.is_log:
            if size > 1:
                if rank == 0:
                    print('is_log from file dont match those supplied!')
                MPI.COMM_WORLD.Abort()
            raise ValueError('is_log from file dont match those supplied!')

        # Identical to setup, just easier for scp'ing *info.pkl files.
        if os.path.exists('{!s}.binfo.pkl'.format(prefix)):
            base_kwargs = read_pickle_file('{!s}.binfo.pkl'.format(prefix),\
                nloads=1, verbose=False)
        else:
            # Deprecate this eventually
            base_kwargs = read_pickle_file('{!s}.setup.pkl'.format(prefix),\
                nloads=1, verbose=False)

        (nwalkers, save_freq, steps) =\
            read_pickle_file('{!s}.rinfo.pkl'.format(self.prefix), nloads=1,\
            verbose=False)

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

        try:
            if self.checkpoint_append:
                chain = read_pickled_chain('{!s}.chain.pkl'.format(prefix))
            else:
                # lec = largest existing checkpoint
                chain = \
                    read_pickled_chain(self._latest_checkpoint_chain_file(prefix_restart))

            pos = chain[-((self.nwalkers-1)*self.save_freq)-1::self.save_freq,:]

        except ValueError:
            print("WARNING: chain empty! Starting from last point in burn-in")

            chain = read_pickled_chain('{!s}.burn.chain.pkl'.format(prefix))
            prob = read_pickled_logL('{!s}.burn.logL.pkl'.format(prefix))
            mlpt = chain[np.argmax(prob)]
            pos = sample_ball(mlpt, np.std(chain, axis=0), size=self.nwalkers)

        return pos

    def _saved_checkpoint_chain_files(self, prefix):
        return glob.glob(prefix + ".dd*.chain.pkl")


    def _saved_checkpoints(self, prefix):
        ans = [int(fn[-14:-10])\
            for fn in self._saved_checkpoint_chain_files(prefix)]
        return ans


    def _latest_checkpoint_chain_file(self, prefix):
        all_chain_fns = self._saved_checkpoint_chain_files(prefix)
        ckpt_numbers = [int(fn[-14:-10]) for fn in all_chain_fns]

        return all_chain_fns[np.argmax(ckpt_numbers)]


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

    @property
    def counter(self):
        if not hasattr(self, '_counter'):
            self._counter = 0

        return self._counter

    @counter.setter
    def counter(self, value):
        assert value == self._counter +1
        self._counter += 1

    def _prep_from_scratch(self, clobber, by_proc=False):
        if (rank > 0) and (not by_proc):
            return

        if by_proc:
            prefix_by_proc =\
                '{0!s}.{1!s}'.format(self.prefix, str(rank).zfill(3))
        else:
            prefix_by_proc = self.prefix

        if clobber:
            # Delete only the files made by this routine. Don't want to risk
            # deleting other files the user may have created with similar
            # naming convention!
            # These suffixes are always the same
            for suffix in ['logL', 'chain', 'facc', 'pinfo', 'rinfo',
                'binfo', 'setup', 'load', 'fail', 'timeout']:

                _fn1 = '{0!s}.{1!s}.pkl'.format(self.prefix, suffix)

                if os.path.exists(_fn1):
                    os.remove(_fn1)

                for _fn2 in glob.glob('{0!s}.*.{1!s}.pkl'.format(self.prefix,\
                    suffix)):

                    if os.path.exists(_fn2):
                        os.remove(_fn2)

            if os.path.exists('{!s}.prior_set.hdf5'.format(self.prefix)):
                os.remove('{!s}.prior_set.hdf5'.format(self.prefix))

            # These suffixes have their own suffixes
            for _fn in glob.glob('{!s}.blob_*.pkl'.format(self.prefix)):
                if os.path.exists(_fn):
                    os.remove(_fn)
            for _fn in glob.glob('{!s}.*.blob_*.pkl'.format(self.prefix)):
                if os.path.exists(_fn):
                    os.remove(_fn)

        # Each processor gets its own fail file
        f = open('{!s}.fail.pkl'.format(prefix_by_proc), 'wb')
        f.close()

        # Main output: MCMC chains (flattened)
        if self.checkpoint_append:
            f = open('{!s}.chain.pkl'.format(prefix_by_proc), 'wb')
            f.close()

            # Main output: log-likelihood
            f = open('{!s}.logL.pkl'.format(self.prefix), 'wb')
            f.close()

        # Store acceptance fraction
        f = open('{!s}.facc.pkl'.format(self.prefix), 'wb')
        f.close()

        # File for blobs themselves
        if self.blob_names is not None and self.checkpoint_append:

            for i, group in enumerate(self.blob_names):
                for blob in group:
                    fntup = (prefix_by_proc, self.blob_nd[i], blob)
                    f = open('{0!s}.blob_{1}d.{2!s}.pkl'.format(*fntup), 'wb')
                    f.close()

        # Parameter names and list saying whether they are log10 or not
        write_pickle_file((self.parameters, self.is_log),\
            '{!s}.pinfo.pkl'.format(self.prefix), ndumps=1, open_mode='w',\
            safe_mode=False, verbose=False)

        # "Run" info (MCMC only)
        if hasattr(self, 'steps'):
            write_pickle_file((self.nwalkers, self.save_freq, self.steps),\
                '{!s}.rinfo.pkl'.format(self.prefix), ndumps=1, open_mode='w',\
                safe_mode=False, verbose=False)

        # Priors!
        #self.prior_set.save(self.prefix + '.prior_set.hdf5')

        # Constant parameters being passed to ares.simulations.Global21cm
        tmp = self.base_kwargs.copy()

        to_axe = []
        to_up = {}
        for key in tmp:
            # these might be big, get rid of it
            if re.search('tau_instance', key):
                to_axe.append(key)
            if re.search('tau_table', key):
                to_axe.append(key)
            if re.search('hmf_instance', key):
                to_axe.append(key)
            if re.search('pop_psm_instance', key):
                to_axe.append(key)
            if re.search('pop_src_instance', key):
                to_axe.append(key)
            if re.search('pop_sed_by_Z', key):
                to_axe.append(key)
            if re.search('pop_histories', key):
                if type(tmp[key]) == str:
                    pass
                else:
                    if 'fn' in tmp[key]:
                        to_up[key] = tmp[key]['fn']
                    to_axe.append(key)
            if re.search('hmf_table', key):
                to_axe.append(key)
            if re.search('hmf_data', key):
                to_axe.append(key)
            if re.search('pop_sps_data', key):
                to_axe.append(key)
            if re.search('pop_synth_phot_cache', key):
                to_axe.append(key)

            # Apparently functions of any kind cause problems everywhere
            # but my laptop
            #
            # NOTE from KT: InstanceType class was checked for along with
            # FunctionType class but, since InstanceType class is deprecated in
            # Python 3 and KT's purpose in editing is to port to Python 3,
            # references to InstanceType were removed.
            if type(tmp[key]) is FunctionType:
                to_axe.append(key)
            elif type(tmp[key]) is tuple:
                for element in tmp[key]:
                    if type(element) is FunctionType:
                        to_axe.append(key)
                        break

        for key in to_axe:
            tmp[key] = None
        for key in to_up:
            tmp[key] = to_up[key]

        # If possible, include ares revision used to run this fit.
        tmp['revision'] = get_hash()

        # Write to disk.
        write_pickle_file(tmp, '{!s}.binfo.pkl'.format(self.prefix), ndumps=1,\
            open_mode='w', safe_mode=False, verbose=False)
        del tmp

    @property
    def debug(self):
        if not hasattr(self, '_debug'):
            self._debug = False
        return self._debug

    @debug.setter
    def debug(self, value):
        assert type(value) in [int, bool]
        self._debug = value

    def _reinitialize_walkers(self, chain, logL, burn_method=0):
        """
        Re-initialize walkers after burn-in.

        Doing this a little more carefully now. Just focusing on last
        `nwalkers` elements of the chain, and will iteratively find
        a new point around which to center walkers based on ranked list of
        likelihoods AND only accepting points within bulk of final distribution.

        Note that we're correcting for something that would likely be fixed
        by a longer burn-in, and as a result will print a warning message
        if we've kludged the walker positions relative to the naive
        approach.

        Parameters
        ----------
        chain : np.ndarray
            An array with shape (nwalkers, nparams). The last snapshot of the
            burn-in.
        logL : np.ndarray
            log likelihoods from burn-in
        """

        ilogL = np.argsort(logL)[-1::-1]

        mlpt = chain[np.argmax(logL)]
        orig = mlpt.copy()
        sigm = np.std(chain, axis=0)

        # If burn_method==0, just re-initialize all walkers around
        # max-likelihood point with jitter, not standard deviation of samples.
        # Use half the jitter prescribed by hand to initially set walkers
        # going.
        if burn_method == 0:
            pos = sample_ball(mlpt, self.jitter * 0.5, size=self.nwalkers)
            return self._fix_guesses(pos)

        # Find out if max likelihood point is within the bulk of
        # the distribution.
        j = 0
        ok = False
        while (not ok) and (j < len(ilogL) - 2):
            for i in range(chain.shape[1]):
                pdf, _bins = np.histogram(chain[:,i], bins=100)

                x = bin_e2c(_bins)
                cdf = np.cumsum(pdf) / float(np.sum(pdf))

                x1 = np.interp(0.16, cdf, x)
                x2 = np.interp(0.84, cdf, x)

                if x1 <= mlpt[i] <= x2:
                    ok = True
                else:
                    ok = False
                    break

            if not ok:
                j += 1

            mlpt = chain[ilogL[j]]

        # If we got to the end of the chain without finding a point
        # in the bulk of the distribution, just revert to the max
        # likelihood pt. This usually only happens during testing
        # when we have very few samples to work with.
        if j == (len(logL) - 1):
            mlpt = chain[np.argmax(logL)]

        # Add the systematic difference between the PDF and best fit?
        syst = abs(chain[np.argmax(logL)] - mlpt) * 0.5

        pos = sample_ball(mlpt, sigm + syst, size=self.nwalkers)

        if not np.all(mlpt == orig):
            if j+1 == 2:
                sup = 'nd'
            elif j+1 == 3:
                sup = 'rd'
            else:
                sup = 'th'
            print("# WARNING: Burn-in may have been too short based on location of")
            print("#          maximum likelihood position (outside 1-sigma dist. of")
            print("#          final walker positions). As a result, we have re-centered")
            print("#          around the {}{} best point from last snapshot of burn.".format(j+1, sup))

        return self._fix_guesses(pos)

    def run(self, prefix, steps=1e2, burn=0, clobber=False, restart=False,
        save_freq=500, reboot=False, recenter=False, burn_method=0):
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

        if rank == 0:
            if os.path.exists('{!s}.chain.pkl'.format(prefix)) and (not clobber):
                if not restart:
                    raise IOError(('{!s} exists! Remove manually, set ' +\
                        'clobber=True, or set restart=True to ' +\
                        'append.').format(prefix))

        if size > 1:
            MPI.COMM_WORLD.Barrier()

        #if self.checkpoint_append:
        #    if not os.path.exists('{!s}.chain.pkl'.format(prefix)) and restart:
        #        raise IOError(("This can't be a restart, {!s}*.pkl not " +\
        #            "found.").format(prefix))

        burn_prev = 0
        pref_prev = prefix + '.burn'
        restart_from_burn = False
        if restart:

            if clobber:
                raise IOError("If restart=True, should set clobber=False!")

            # below checks for checkpoint_append==True failure
            cptapdtrfl = (self.checkpoint_append and\
                (not os.path.exists('{!s}.chain.pkl'.format(prefix))))
            # below checks for checkpoint_append==False failure
            cptapdflsfl = ((not self.checkpoint_append) and\
                (not glob.glob('{!s}.dd*.pkl'.format(prefix))))

            cptapdtrfl_b = (self.checkpoint_append and\
                (not os.path.exists('{!s}.burn.chain.pkl'.format(prefix))))
            # below checks for checkpoint_append==False failure
            cptapdflsfl_b = ((not self.checkpoint_append) and\
                (not glob.glob('{!s}.burn.dd*.pkl'.format(prefix))))

            # either way, produce error
            if (cptapdtrfl or cptapdflsfl):

                if (cptapdtrfl_b + cptapdflsfl_b == 0):
                    restart_from_burn = True
                else:
                    raise IOError(("This can't be a restart, {!s}*.pkl not " +\
                        "found.").format(prefix))

            # Should make sure file isn't empty, either.
            # Is it too dangerous to set clobber=True, here?
            burn_prev = 0
            try:
                if self.checkpoint_append:
                    if restart_from_burn:
                        fn_last_chain = '{!s}.burn.chain.pkl'.format(prefix)
                    else:
                        fn_last_chain = '{!s}.chain.pkl'.format(prefix)

                    _chain = read_pickled_chain(fn_last_chain)

                else:
                    if restart_from_burn:
                        pref_prev = prefix + '.burn'
                    else:
                        pref_prev = prefix

                    fn_last_chain = \
                        self._latest_checkpoint_chain_file(pref_prev)

                    _chain = read_pickled_chain(fn_last_chain)

                    if restart_from_burn:
                        _anl_ = ModelSet(pref_prev, verbose=False)
                        burn_prev = _anl_.chain.shape[0] // self.nwalkers

                if (rank == 0) and (burn-burn_prev > 0):
                    print("# Will restart from {}".format(fn_last_chain))

            except ValueError:
                if rank == 0:
                    has_burn = os.path.exists('{!s}.burn.chain.pkl'.format(prefix))
                    if not has_burn:
                        restart = False
                        clobber = True

                    # Print warning to screen
                    not_a_restart(prefix, has_burn)

        ##
        # Initialize Pool
        ##
        if size > 1:
            self.pool = MPIPool()

            if not emcee_mpipool:
                self.pool.start()

            # Non-root processors wait for instructions until job is done,
            # at which point, they don't need to do anything below here.
            if not self.pool.is_master():

                if emcee_mpipool:
                    self.pool.wait()
                if not reboot:
                    sys.exit(0)

                del self.pool
                gc.collect()

                self.counter = self.counter + 1

                print("Commence reboot #{} (proc={})".format(self.counter,
                    rank))

                self.run(prefix, steps=steps, burn=0, clobber=False,
                    restart=True, save_freq=save_freq,
                    reboot=self.counter < reboot, burn_method=burn_method)

        else:
            self.pool = None

        ##
        # Only rank==0 processor ever makes it here
        ##

        self.steps = steps
        self.save_freq = save_freq

        ##
        # Speed-up tricks
        ##
        if self.save_hmf or self.save_src or self.save_hist:
            sim = self.simulator(**self.base_kwargs)

        if self.save_hmf:
            assert 'hmf_instance' not in self.base_kwargs

            # Can generalize more later...
            try:
                hmf = sim.halos
            except AttributeError:
                hmf = None
                for pop in sim.pops:
                    if hasattr(pop, 'halos'):
                        hmf = pop.halos
                        break

            if hmf is None:
                raise AttributeError('No `hmf` attributes available!')

            self.base_kwargs['hmf_instance'] = hmf

            if hmf is not None:
                print("Saved HaloMassFunction instance to limit I/O.")

        if self.save_hist and type(self.base_kwargs['pop_histories']) == str:
            hist = sim.load()
            self.base_kwargs['pop_histories'] = hist

            if hist is not None:
                print("Saved halo histories to limit I/O.")

        if self.save_src:
            # This maybe is unnecessary?
            #assert 'pop_psm_instance' not in self.base_kwargs

            try:
                srcs = [sim.src]
                ids = [None]
            except AttributeError:
                srcs = [pop.src for pop in sim.pops]
                ids = range(len(srcs))

            for idnum, src in enumerate(srcs):

                if ids[idnum] is None:
                    sid = ''
                else:
                    sid = '{{{}}}'.format(idnum)

                if isinstance(src, SynthesisModel):
                    assert 'pop_Z{}'.format(sid) not in self.parameters
                    print("Saved SynthesisModel instance for population #{} to limit I/O.".format(idnum))

                elif isinstance(src, BlackHole):
                    assert 'pop_mass{}'.format(sid) not in self.parameters
                    assert 'pop_eta{}'.format(sid) not in self.parameters
                    assert 'pop_fduty{}'.format(sid) not in self.parameters
                    assert 'pop_alpha{}'.format(sid) not in self.parameters
                    assert 'pop_logN{}'.format(sid) not in self.parameters
                    assert 'pop_fsc{}'.format(sid) not in self.parameters

                    print("Saved BlackHole instance for population #{} to limit I/O.".format(idnum))


                self.base_kwargs['pop_src_instance{}'.format(sid)] = src

        ##
        # Initialize sampler
        ##
        args = [self.prefix, self.parameters, self.is_log, self.prior_set_P,
            self.prior_set_B, self.blank_blob,
            self.base_kwargs, self.checkpoint_by_proc,
            self.simulator, self.fitters, self.debug]

        self.sampler = emcee.EnsembleSampler(self.nwalkers,
            self.Nd, loglikelihood, pool=self.pool, args=args)

        # If restart, will use last point from previous chain, or, if one
        # isn't found, will look for burn-in data.
        if restart_from_burn:
            pos = self.prep_output_files(restart, clobber, pref_prev)
        else:
            pos = self.prep_output_files(restart, clobber)

        # Optional re-centering of walkers
        if restart and recenter:
            print("# Recentering walkers...")
            mlpt = pos[np.argmax(prob)]
            pos = sample_ball(mlpt, np.std(pos, axis=0), size=self.nwalkers)

        state = None #np.random.RandomState(self.seed)

        ##
        # Run burn-in
        ##
        if (burn-burn_prev > 0) and ((not restart) or restart_from_burn):

            if (burn_method == 0) and (not restart_from_burn):
                if not hasattr(self, '_jitter'):
                    raise AttributeError("Must set jitter!")

            if restart_from_burn:
                if os.path.exists('{!s}.rstate.pkl'.format(prefix)):
                    state = read_pickle_file('{!s}.rstate.pkl'.format(prefix),\
                        nloads=1, verbose=False)
                    if rank == 0:
                        print("# Using pre-restart RandomState.")

                print("# Re-starting burn-in: {!s}".format(time.ctime()))
                if burn_prev > 0:
                    print("# Previous burn-in: {} step per walker complete.".format(burn_prev))
                    print("# Will burn-in for {} more steps per walker.".format(burn - burn_prev))
            else:
                print("# Starting burn-in: {!s}".format(time.ctime()))

            if save_freq >= (burn - burn_prev):
                print("# Note: you might want to set `save_freq` < `burn`!")

            if pos is None:
                pos = self.guesses

            t1 = time.time()
            pos, prob, state, blobs = \
                self._run(pos, burn-burn_prev, state=state,
                    save_freq=save_freq,
                    restart=restart_from_burn, prefix=pref_prev)
            t2 = time.time()

            print("# Burn-in complete in {0:.3g} seconds.".format(t2 - t1))

            if steps > 0:
                pos = self._reinitialize_walkers(pos, prob, burn_method)
                self.sampler.reset()
                gc.collect()
        elif not restart:
            pos = self.guesses
            state = None
        elif os.path.exists('{!s}.rstate.pkl'.format(prefix)):
            state = read_pickle_file('{!s}.rstate.pkl'.format(prefix),\
                nloads=1, verbose=False)
            if rank == 0:
                print("# Using pre-restart RandomState.")
        else:
            state = None

        ##
        # Burn-only mode.
        if steps == 0:
            if self.pool is not None and emcee_mpipool:
                self.pool.close()
            elif self.pool is not None:
                self.pool.stop()

            if rank == 0:
                print("# Burn-in complete, and `step`=0.")
                print("# Exiting: {!s}".format(time.ctime()))
            return



        ##
        # MAIN MCMC
        ##
        if rank == 0:
            print("# Starting MCMC: {!s}".format(time.ctime()))

            if save_freq >= steps:
                print("# Note: you might want to set `save_freq` < `steps`!")

            pos, prob, state, blobs = \
                self._run(pos, steps, state=state, save_freq=save_freq,
                    restart=restart and (not restart_from_burn),
                    prefix=self.prefix)

        if self.pool is not None and emcee_mpipool:
            self.pool.close()
        elif self.pool is not None:
            self.pool.stop()

        if rank == 0:
            print("# Finished on {!s}".format(time.ctime()))

        ##
        # New option: Reboot. See if destruction of Pool affects memory.
        ##
        if not reboot:
            return

        del self.pool, self.sampler
        gc.collect()

        self.counter = self.counter + 1

        if rank == 0:
            print("# Commence reboot #{} (proc={})".format(self.counter, rank))

        self.run(prefix, steps=steps, burn=0, clobber=False, restart=True,
            save_freq=save_freq, reboot=self.counter < reboot)

    def _run(self, pos, steps, state, save_freq, restart, prefix):
        """
        Run sampler and handle I/O.
        """

        if restart and (not self.checkpoint_append):
            ct = save_freq * (1 + max(self._saved_checkpoints(prefix)))
        else:
            ct = 0

        if emcee_v >= 3:
            kw = {}
        else:
            kw = {'storechain': False}

        # Take steps, append to pickle file every save_freq steps
        pos_all = []; prob_all = []; blobs_all = []
        for pos, prob, state, blobs in self.sampler.sample(pos,
            iterations=steps, rstate0=state, **kw):

            # If we're saving each checkpoint to its own file, this is the
            # identifier to use in the filename
            dd = 'dd' + str(ct // save_freq).zfill(4)

            # Increment counter
            ct += 1

            pos_all.append(pos.copy())
            prob_all.append(prob.copy())
            blobs_all.append(blobs)

            #del blobs

            if ct % save_freq != 0:
                gc.collect()
                continue

            # Remember that pos.shape = (nwalkers, ndim)
            # So, pos_all has shape = (nsteps, nwalkers, ndim)

            data = [flatten_chain(np.array(pos_all)),
                    flatten_logL(np.array(prob_all)),
                    blobs_all, state]

            self._write_checkpoint(data, dd, ct, prefix, save_freq)

            del pos_all, prob_all, blobs_all, data

            # Delete chain, logL, etc., to be conscious of memory
            self.sampler.reset()

            gc.collect()

            pos_all = []; prob_all = []; blobs_all = []


        ##
        # Means there's a mismatch in `save_freq` and (`steps` and/or `burn`)
        # so we never hit the last checkpoint. Don't let those samples go
        # to waste!
        ##
        if pos_all != []:
            print("Writing data one last time because save_freq > steps.")
            data = [flatten_chain(np.array(pos_all)),
                    flatten_logL(np.array(prob_all)),
                    blobs_all, state]

            self._write_checkpoint(data, dd, ct, prefix, save_freq)

            del pos_all, prob_all, blobs_all, data

        # Just return progress over last `save_freq` steps.
        return pos, prob, state, blobs

    def _write_checkpoint(self, data, dd, ct, prefix, save_freq):
        # The flattened version of pos_all has
        # shape = (save_freq * nwalkers, ndim)

        if self.checkpoint_append:
            mode = 'ab'
        else:
            mode = 'wb'

        for i, suffix in enumerate(['chain', 'logL', 'blobs']):

            # Blobs
            if suffix == 'blobs':
                if self.blob_names is None:
                    continue
                self.save_blobs(data[i], dd=dd, prefix=prefix)
            # Other stuff
            else:
                if self.checkpoint_append:
                    fn = '{0!s}.{1!s}.pkl'.format(prefix, suffix)
                else:
                    fn = '{0!s}.{1!s}.{2!s}.pkl'.format(prefix, dd, suffix)

                write_pickle_file(data[i], fn, ndumps=1,\
                    open_mode=mode[0], safe_mode=False, verbose=False)

        if self.checkpoint_append:
            fn_facc = '{0!s}.facc.pkl'.format(prefix)
        else:
            fn_facc = '{0!s}.{1!s}.facc.pkl'.format(prefix, dd)

        # This is a running total already so just save the end result
        # for this set of steps
        write_pickle_file(self.sampler.acceptance_fraction,
            fn_facc, ndumps=1, open_mode=mode[0],\
            safe_mode=False, verbose=False)

        if self.checkpoint_append:
            print("# Checkpoint #{0}: {1!s}".format(ct // save_freq,
                time.ctime()))
        else:
            print("# Wrote {0!s}*.pkl: {1!s}".format(prefix, time.ctime()))

        ##################################################################
        write_pickle_file(data[-1], '{!s}.rstate.pkl'.format(prefix),\
            ndumps=1, open_mode='w', safe_mode=False, verbose=False)
        ##################################################################

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

        # Right now, len(blobs) = save_freq
        # Each element within it has `nwalkers` elements.

        # Only need to deal with walkers for MCMC data, i.e., ModelGrid runs
        # don't have walkers.
        if uncompress:
            blobs_now = []
            for j in range(0, self.nwalkers):
                for k in range(blen):
                    blobs_now.append(blobs[k][j])
        else:
            blobs_now = blobs

        # Now, len(blobs_now) = save_freq * nwalkers

        # We're saving one file per blob
        # The shape of the array will be just blob_nd

        if self.blob_names is None:
            return

        # At this moment, blobs_now has dims =
        # (nwalkers * nsteps, num blob groups, num blobs by group)

        for j, group in enumerate(self.blob_names):
            for k, blob in enumerate(group):
                to_write = []
                for l in range(self.nwalkers * blen):
                    # indices: walkers*steps, blob group, blob
                    barr = blobs_now[l][j][k]
                    to_write.append(barr)

                if self.checkpoint_append:
                    mode = 'ab'
                    bfn = '{0!s}.blob_{1}d.{2!s}.pkl'.format(prefix,\
                        self.blob_nd[j], blob)
                else:
                    mode = 'wb'
                    bfn = '{0!s}.{1!s}.blob_{2}d.{3!s}.pkl'.format(prefix, dd,\
                        self.blob_nd[j], blob)

                    assert dd is not None, "checkpoint_append=False but no DDID!"

                write_pickle_file(np.array(to_write), bfn, ndumps=1,\
                    open_mode=mode[0], safe_mode=False, verbose=False)


    @property
    def simulator(self):
        if not hasattr(self, '_simulator'):
            raise AttributeError('Must supply simulator by hand!')
        return self._simulator

    @simulator.setter
    def simulator(self, value):
        self._simulator = value

        # Assert that this is an instance of something we know about?

    @property
    def fitters(self):
        return self._fitters

    def add_fitter(self, fitter):
        if not hasattr(self, '_fitters'):
            self._fitters = []

        if fitter in self._fitters:
            print("This fitter is already included!")
            return

        self._fitters.append(fitter)

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
