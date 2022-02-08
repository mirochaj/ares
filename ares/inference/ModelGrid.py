"""

ModelGrid.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Dec  5 15:49:16 MST 2013

Description: For working with big model grids. Setting them up, running them,
and analyzing them.

"""
from __future__ import print_function

import os
import gc
import re
import copy
import time
import signal
import subprocess
import numpy as np
from .GridND import GridND
from ..util import ProgressBar
from .ModelFit import ModelFit
from ..analysis import ModelSet
from ..simulations import Global21cm
from ..util.ReadData import concatenate
from ..analysis import Global21cm as _AnalyzeGlobal21cm
from ..util.Pickling import read_pickle_file, write_pickle_file

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1

class ModelGrid(ModelFit):
    """Create an object for setting up and running model grids."""

    @property
    def tol(self):
        if not hasattr(self, '_tol'):
            if self.grid.structured:
                self._tol = 1e-6
            else:
                self._tol = 1e-3

        return self._tol

    @property
    def phenomenological(self):
        if not hasattr(self, '_tanh'):
            self._phenomenological = False
            if 'tanh_model' in self.base_kwargs:
                if self.base_kwargs['tanh_model']:
                    self._phenomenological = True
            if 'gaussian_model' in self.base_kwargs:
                if self.base_kwargs['gaussian_model']:
                    self._phenomenological = True
            if 'parametric_model' in self.base_kwargs:
                if self.base_kwargs['parametric_model']:
                    self._phenomenological = True

        return self._phenomenological

    @property
    def simulator(self):
        if not hasattr(self, '_simulator'):
            self._simulator = Global21cm
        return self._simulator

    def _read_restart(self, prefix, procid=None):
        """
        Figure out which models have already been run.

        Note that how this works depends on if the gridding has been changed.

        Parameters
        ----------
        prefix : str
            File prefix of files ending in *.pkl to be read in.

        """

        # Array of ones/zeros: has this model already been done?
        # This is only the *new* grid points.
        if self.grid.structured:
            done = np.zeros(self.grid.shape)

        if procid is None:
            procid = rank

        if os.path.exists('{0!s}.{1!s}.chain.pkl'.format(prefix, str(procid).zfill(3))):
            prefix_by_proc = '{0!s}.{1!s}'.format(prefix, str(procid).zfill(3))
        else:
            return done
            #raise ValueError('This shouldn\'t happen anymore.')

        # Need to see if we're running with the same number of processors
        # We could probably set things up such that this isn't a problem.
        #fn_by_proc = lambda proc: '{0!s}.{1!s}.chain.pkl'.format(prefix, str(proc).zfill(3))
        #fn_size_p1 = fn_by_proc(size+1)
        #if os.path.exists(fn_size_p1):
        #    raise IOError('Original grid run with more processors!')
        #
        #    proc_id = size + 1
        #    while os.path.exists(fn_by_proc(proc_id)):
        #        proc_id += 1
        #        continue

        # Read in current status of model grid, i.e., the old
        # grid points.
        chain = concatenate(read_pickle_file('{!s}.chain.pkl'.format(\
            prefix_by_proc), nloads=None, verbose=False))

        # If we said this is a restart, but there are no elements in the
        # chain, just run the thing. It probably means the initial run never
        # made it to the first checkpoint.

        if chain.size == 0:
            if rank == 0:
                print(("Pre-existing chain file(s) empty for proc #{}. " +\
                    "Running from beginning.").format(rank))
            if not self.grid.structured:
                self.done = np.array([0])

            return done

        # Read parameter info
        (axes_names, is_log) = read_pickle_file(\
            '{!s}.pinfo.pkl'.format(prefix), nloads=1, verbose=False)

        # Prepare for blobs (optional)
        if os.path.exists('{!s}.binfo.pkl'.format(prefix)):
            self.pf = read_pickle_file('{!s}.binfo.pkl'.format(prefix),\
                nloads=1, verbose=False)
        elif os.path.exists('{!s}.setup.pkl'.format(prefix)):
            self.pf = read_pickle_file('{!s}.setup.pkl'.format(prefix),\
                nloads=1, verbose=False)

        if len(axes_names) != chain.shape[1]:
            raise ValueError('Cannot change dimensionality on restart!')

        if self.grid.structured:
            if axes_names != self.grid.axes_names:
                raise ValueError('Cannot change axes variables on restart!')
        else:
            for par in axes_names:
                if par in self.grid.axes_names:
                    continue
                raise ValueError('Cannot change axes variables on restart!')

        # What follows is irrelevant for unstructured grids.
        if (not self.grid.structured):
            self.done = done = np.array([chain.shape[0]])
            return done

        # Loop over chain read-in from disk and compare to grid.
        # Which models have already been computed?
        for link in chain:

            # Parameter set from pre-existing chain
            kw = {par:link[i] \
                for i, par in enumerate(self.grid.axes_names)}

            # Its location in *new* grid
            kvec = self.grid.locate_entry(kw, tol=self.tol)

            if None in kvec:
                continue

            done[kvec] = 1

        if done.sum() != len(chain):
            print("WARNING: Some chain elements not found.")

        return done

    @property
    def axes(self):
        return self.grid.axes

    @axes.setter
    def axes(self, kwargs):
        """
        Create GridND instance, construct N-D parameter space.
        """

        for kwarg in kwargs:
            assert kwargs[kwarg].size == np.unique(kwargs[kwarg]).size, \
                "Redundant elements detected for parameter={!s}".format(kwarg)

        self.grid = GridND()

        # Build parameter space
        self.grid.build(**kwargs)

        # Save for later access
        self.kwargs = kwargs

        # Shortcut to parameter names
        self.parameters = self.grid.axes_names

    @property
    def priors(self):
        # Need to override priors in ModelFit
        return {}

    def set_models(self, models):
        """
        Set all models by hand.

        Parameters
        ----------
        models : list
            List of models to run. Each entry in the list should be a
            dictionary of parameters that define that model. The
            base_kwargs will be updated with those values at run-time.

        """

        self.grid = GridND()

        # Build parameter space
        self.grid.all_kwargs = models
        self.grid.axes_names = list(models[0].keys())
        self.grid.Nd = len(self.grid.axes_names)

        # Shortcut to parameter names
        if not hasattr(self, 'parameters'):
            self.parameters = self.grid.axes_names

    def _reshape_assignments(self, assignments):
        assign = []
        for h, kwargs in enumerate(self.grid.all_kwargs):

            # Where does this model live in the grid?
            if self.grid.structured:
                kvec = self.grid.locate_entry(kwargs, tol=self.tol)
            else:
                kvec = h

            if self.is_restart:
                if hasattr(self, 'done'):
                    if self.done[kvec]:
                        continue

            assign.append(assignments[kvec])

        return np.array(assign, dtype=int)

    def prep_output_files(self, restart, clobber):
        """
        Stick this in utilities folder?
        """

        prefix_by_proc = '{0!s}.{1!s}'.format(self.prefix, str(rank).zfill(3))

        # Reshape assignments so it's Nlinks long.
        if self.grid.structured:
            assignments = self._reshape_assignments(self.assignments)

            if restart:
                if rank == 0:
                    write_pickle_file(assignments,\
                        '{!s}.load.pkl'.format(self.prefix), ndumps=1,\
                        open_mode='a', safe_mode=False, verbose=False)

                return
        else:
            if restart:
                return

        if rank > 0:
            return

        super(ModelGrid, self)._prep_from_scratch(clobber, by_proc=True)

        if self.grid.structured:
            write_pickle_file(assignments,\
                '{!s}.load.pkl'.format(self.prefix), ndumps=1, open_mode='w',\
                safe_mode=False, verbose=False)

        # ModelFit makes this file by default but grids don't use it.
        if os.path.exists('{!s}.logL.pkl'.format(self.prefix)) and (rank == 0):
            os.remove('{!s}.logL.pkl'.format(self.prefix))

        for par in self.grid.axes_names:
            if re.search('Tmin', par):
                f = open('{!s}.fcoll.pkl'.format(prefix_by_proc), 'wb')
                f.close()
                break

    @property
    def blank_blob(self):
        if not hasattr(self, '_blank_blob'):
            blob_names = self.base_kwargs['blob_names']

            if blob_names is None:
                self._blank_blob = []
                return []

            blob_ivars = self.base_kwargs['blob_ivars']
            blob_funcs = self.base_kwargs['blob_funcs']
            blob_nd = [len(grp) if grp is not None else 0 \
                for grp in blob_ivars]

            ##
            # Need to be a little careful with blob ivars due to
            # new-ish (ivar name, ivar values) approach.
            ##
            blob_dims = []
            for grp in blob_ivars:
                if grp is None:
                    blob_dims.append(None)
                    continue

                dims = []
                for element in grp:
                    ivarn, ivars = element
                    dims.append(len(ivars))

                blob_dims.append(tuple(dims))

            self._blank_blob = []
            for i, group in enumerate(blob_names):
                if blob_ivars[i] is None:
                    self._blank_blob.append([np.inf] * len(group))
                else:
                    if blob_nd[i] == 0:
                        self._blank_blob.append([np.inf] * len(group))
                    elif blob_nd[i] == 1:
                        arr = np.ones([len(group), blob_dims[i][0]])
                        self._blank_blob.append(arr * np.inf)
                    elif blob_nd[i] == 2:
                        dims = len(group), blob_dims[i][0], \
                            blob_dims[i][1]
                        arr = np.ones(dims)
                        self._blank_blob.append(arr * np.inf)

        return self._blank_blob

    @property
    def simulator(self):
        if not hasattr(self, '_simulator'):
            from ..simulations import Global21cm
            self._simulator = Global21cm
        return self._simulator

    @property
    def reuse_splines(self):
        if not hasattr(self, '_reuse_splines'):
            self._reuse_splines = True
            if 'feedback_LW' in self.base_kwargs:
                if self.base_kwargs['feedback_LW']:
                    self._reuse_splines = False

        return self._reuse_splines

    @reuse_splines.setter
    def reuse_splines(self, value):
        self._reuse_splines = value

    @property
    def tricks(self):
        if not hasattr(self, '_tricks'):
            self._tricks = []
        return self._tricks

    @tricks.setter
    def tricks(self, value):
        if not hasattr(self, '_tricks'):
            assert type(value) is tuple
            self._tricks = [value]
        else:
            self._tricks.append(value)

    @property
    def trick_names(self):
        return list(zip(*self.tricks))[0]

    @property
    def trick_files(self):
        return list(zip(*self.tricks))[1]

    #@property
    #def trick_data(self):
    #    if not hasattr(self, '_trick_data'):
    #        self._trick_data = {}
    #    return self._trick_data
    #
    #@trick_data.setter
    #def trick_data(self, value):
    #    if not hasattr(self, '_tricks'):
    #        assert type(value) is dict
    #        self._tricks = value
    #    else:
    #        self._tricks.update(value)

    @property
    def is_restart(self):
        if not hasattr(self, '_is_restart'):
            self._is_restart = False
        return self._is_restart

    @is_restart.setter
    def is_restart(self, value):
        self._is_restart = value

    def _prep_tricks(self): # pragma: no cover
        """
        Super non-general at the moment sorry.
        """

        if 'guess_popIII_sfrds' in self.trick_names:

            i = self.trick_names.index('guess_popIII_sfrds')
            fn = self.trick_files[i]
            if fn is not None:
                anl = ModelSet(fn)
                #if not anl.chain:
                #    return

                print("Ready to cheat!")

                # HARD CODING FOR NOW
                blob_name = 'popIII_Mmin'
                Mmin = anl.ExtractData(blob_name)[blob_name]
                zarr = anl.get_ivars(blob_name)[0]

                def func(**kw):

                    # First, figure out where (if anywhere) the parameters
                    # in hand live in the lookup table.

                    ind = []
                    for k, par in enumerate(anl.parameters):

                        if par not in kw:
                            ind.append(None)
                            continue

                        try:
                            i = list(anl.parameters).index(kw[par])
                        except ValueError:
                            i = np.argmin(np.abs(kw[par] - anl.unique_samples[k]))

                        ind.append(i)

                    score = 0.0
                    for k, par in enumerate(anl.parameters):
                        if ind[k] is None:
                            continue

                        vals = anl.chain[:,ind[k]]

                        print("{0!s} {1!s} {2!s}".format(k, par, kw[par]))

                        score += np.abs(vals - kw[par])

                    best = np.argmin(score)

                    print("{0!s} {1!s} {2!s}".format(zarr.shape, Mmin.shape,\
                        best))

                    f = lambda zz: np.interp(zz, zarr, Mmin[best])
                    return {'pop_Mmin{2}': f}

                    #if np.min(score) == 0:



                self.trick_funcs['guess_popIII_sfrds'] = func

    @property
    def trick_funcs(self):
        if not hasattr(self, '_trick_funcs'):
            self._trick_funcs = {}
        return self._trick_funcs

    @trick_funcs.setter
    def trick_funcs(self, value):
        if not hasattr(self, '_trick_funcs'):
            self._trick_funcs = {}

        assert type(value) is dict
        self._trick_funcs.update(value)

    @property
    def _exit_if_fail_streak(self):
        if not hasattr(self, '_exit_if_fail_streak_'):
            self._exit_if_fail_streak_ = False
        return self._exit_if_fail_streak_

    @_exit_if_fail_streak.setter
    def _exit_if_fail_streak(self, value):
        self._exit_if_fail_streak_ = bool(value)

    def _run_sim(self, kw, p):

        failct = 0
        sim = self.simulator(**p)

        if self.debug:
            sim.run()
            blobs = sim.blobs

        try:
            if not self.debug:
                sim.run()
                blobs = copy.deepcopy(sim.blobs)
        except RuntimeError:
            write_pickle_file(kw, '{0!s}.{1!s}.timeout.pkl'.format(\
                self.prefix, str(rank).zfill(3)), ndumps=1, open_mode='a',\
                safe_mode=False, verbose=False)

            blobs = copy.deepcopy(self.blank_blob)
        except MemoryError:
            raise MemoryError('This cannot be tolerated!')
        except:
            # For some reason "except Exception"  doesn't catch everything...
            # Write to "fail" file
            write_pickle_file(kw, '{0!s}.{1!s}.fail.pkl'.format(self.prefix,\
                str(rank).zfill(3)), ndumps=1, open_mode='a', safe_mode=False,\
                verbose=False)

            print("FAILURE: Processor #{}.".format(rank))

            failct += 1

            blobs = copy.deepcopy(self.blank_blob)

        if 'feedback_LW_guess' in self.tricks:
            try:
                self.trick_data['pop_Mmin{2}'] = \
                    np.interp(sim.pops[2].halos.tab_z,
                        sim.medium.field._zarr, sim.medium.field._Mmin_now)
            except AttributeError:
                del self.trick_data['pop_Mmin{2}']

        del sim

        return blobs, failct

    @property
    def debug(self):
        if not hasattr(self, '_debug'):
            self._debug = False
        return self._debug

    @debug.setter
    def debug(self, value):
        assert type(value) in [int, bool]
        self._debug = value

    def run(self, prefix, clobber=False, restart=False, save_freq=500,
        use_pb=True, use_checks=True, long_run=False, exit_after=None):
        """
        Run model grid, for each realization thru a given turning point.

        Parameters
        ----------
        prefix : str
            Prefix for all output files.
        save_freq : int
            Number of steps to take before writing data to disk. Note that if
            you're running in parallel, this is the number of steps *each
            processor* will take before writing to disk.
        clobber : bool
            Overwrite pre-existing files of the same prefix if one exists?
        restart : bool
            Append to pre-existing files of the same prefix if one exists?

        Returns
        -------

        """

        self.prefix = prefix
        self.save_freq = save_freq

        prefix_by_proc = '{0!s}.{1!s}'.format(prefix, str(rank).zfill(3))
        prefix_next_proc = '{0!s}.{1!s}'.format(prefix, str(rank+1).zfill(3))

        if rank == 0:
            print("Starting {}-element model grid.".format(self.grid.size))

        chain_exists = os.path.exists('{!s}.chain.pkl'.format(prefix_by_proc))

        # Kill this thing if we're about to delete files and we haven't
        # set clobber=True
        if chain_exists and (not clobber):
            if not restart:
                raise IOError(('{!s}*.pkl exists! Remove manually, set ' +\
                    'clobber=True, or set restart=True to append.').format(\
                    prefix_by_proc))

        restart_actual = True
        _restart_actual = np.zeros(size)
        if restart and (not chain_exists):
            print(("This can't be a restart (for proc #{0}), {1!s}*.pkl " +\
                "not found. Starting from scratch...").format(rank, prefix))
            # Note: this can occur if restarting with fewer processors
            # than we originally ran with.
        else:
            _restart_actual[rank] = 1
            restart_actual = True

        # Figure out if we're running with fewer processors than
        # pre-restart
        fewer_procs = False
        if size > 1:
            _restart_np1 = np.zeros(size)
            if os.path.exists('{!s}.chain.pkl'.format(prefix_next_proc)):
                _restart_np1[rank] = 1

            _tmp = np.zeros(size)
            MPI.COMM_WORLD.Allreduce(_restart_np1, _tmp)
            fewer_procs = sum(_tmp) >= size
        else:
            pass
            # Can't have fewer procs than 1!

        # Need to communicate results of restart_actual across all procs
        if size > 1:
            _all_restart = np.zeros(size)
            MPI.COMM_WORLD.Allreduce(_restart_actual, _all_restart)
            all_restart = bool(sum(_all_restart) == size)
            any_restart = bool(sum(_all_restart) > 0)
        else:
            all_restart = any_restart = _all_restart = _restart_actual

        # If user says it's not a restart, it's not a restart.
        any_restart *= restart

        self.is_restart = any_restart

        # Load previous results if this is a restart
        if any_restart:
            done = self._read_restart(prefix)

            if self.grid.structured:
                Ndone = int(done[done >= 0].sum())
            else:
                Ndone = 0

            # Important that this goes second, otherwise this processor
            # will count the models already run by other processors, which
            # will mess up the 'Nleft' calculation below.

            # Figure out what models have been run by *any* processor
            # in the old grid.
            if size > 1:
                if self.grid.structured:
                    tmp = np.zeros(self.grid.shape)
                    MPI.COMM_WORLD.Allreduce(done, tmp)
                    self.done = np.minimum(tmp, 1)
                else:
                    # In this case, self.done is just an integer.
                    # And apparently, we don't need to know which models are done?
                    tmp = np.array([0])
                    MPI.COMM_WORLD.Allreduce(done, tmp)
                    self.done = tmp[0]
            else:
                self.done = done

            # Find outputs from processors beyond those that we're currently
            # using.
            if fewer_procs:
                if rank == 0:
                    # Determine what the most number of processors to have
                    # run this grid (at some point) is
                    fn_by_proc = lambda proc: '{0!s}.{1!s}.chain.pkl'.format(\
                        prefix, str(proc).zfill(3))
                    fn_size_p1 = fn_by_proc(size+1)

                    _done_extra = np.zeros(self.grid.shape)
                    if os.path.exists(fn_size_p1):

                        proc_id = size + 1
                        while os.path.exists(fn_by_proc(proc_id)):

                            _done_extra += self._read_restart(prefix, proc_id)

                            proc_id += 1
                            continue

                    print(("This grid has been run with as many as {} " +\
                        "processors previously. Collectively, these " +\
                        "processors ran {1} models.").format(proc_id,\
                        _done_extra.sum()))
                    _done_all = self.done.copy()
                    _done_all += _done_extra
                    for i in range(1, size):
                        MPI.COMM_WORLD.Send(_done_all, dest=i, tag=10*i)

                else:
                    self.done = np.zeros(self.grid.shape)
                    MPI.COMM_WORLD.Recv(self.done, source=0, tag=10*rank)

        else:
            Ndone = 0

        if any_restart and self.grid.structured:
            mine_and_done = np.logical_and(self.assignments == rank,
                                           self.done == 1)

            Nleft = self.load[rank] - np.sum(mine_and_done)

        else:
            Nleft = self.load[rank]

        if Nleft == 0:
            print("Processor {} is done already!".format(rank))

        # Print out how many models we have (left) to compute
        if any_restart and self.grid.structured:
            if rank == 0:
                Ndone = self.done.sum()
                Ntot = self.done.size
                print(("Update               : {0} models down, {1} to " +\
                    "go.").format(Ndone, Ntot - Ndone))

            if size > 1:
                MPI.COMM_WORLD.Barrier()

            # Is everybody done?
            if np.all(self.done == 1):
                return

            print(("Update (processor #{0}): Running {1} more " +\
                "models.").format(rank, Nleft))

        elif rank == 0:
            if any_restart:
                print('Re-starting pre-existing model set ({} models done already).'.format(self.done))
                print('Running {} more models.'.format(self.grid.size))
            else:
                print('Running {}-element model grid.'.format(self.grid.size))

        # Make some blank files for data output
        self.prep_output_files(any_restart, clobber)

        # Dictionary for hmf tables
        fcoll = {}

        # Initialize progressbar
        pb = ProgressBar(Nleft, 'grid', use_pb)
        pb.start()

        chain_all = []; blobs_all = []

        t1 = time.time()

        ct = 0
        was_done = 0
        failct = 0

        # Loop over models, use StellarPopulation.update routine
        # to speed-up (don't have to re-load HMF spline as many times)
        for h, kwargs in enumerate(self.grid.all_kwargs):

            # Where does this model live in the grid?
            if self.grid.structured:
                kvec = self.grid.locate_entry(kwargs, tol=self.tol)
            else:
                kvec = h

            # Skip if it's a restart and we've already run this model
            if any_restart and self.grid.structured:
                if self.done[kvec]:
                    was_done += 1
                    pb.update(ct)
                    continue

            # Skip if this processor isn't assigned to this model
            # This could be moved above the previous check
            if self.assignments[kvec] != rank:
                pb.update(ct)
                continue

            # Grab Tmin index
            if self.Tmin_in_grid and self.LB == 1:
                Tmin_ax = self.grid.axes[self.grid.axisnum(self.Tmin_ax_name)]
                i_Tmin = Tmin_ax.locate(kwargs[self.Tmin_ax_name])
            else:
                i_Tmin = 0

            # Copy kwargs - may need updating with pre-existing lookup tables
            p = self.base_kwargs.copy()

            # Log-ify stuff if necessary
            kw = {}
            for i, par in enumerate(self.parameters):
                if self.is_log[i]:
                    kw[par] = 10**kwargs[par]
                else:
                    kw[par] = kwargs[par]

            p.update(kw)

            # Create new splines if we haven't hit this Tmin yet in our model grid.
            if self.reuse_splines and \
                i_Tmin not in fcoll.keys() and (not self.phenomenological):
                #raise NotImplementedError('help')
                sim = self.simulator(**p)

                pops = sim.pops

                if hasattr(self, 'Tmin_ax_popid'):
                    loc = self.Tmin_ax_popid
                    suffix = '{{{}}}'.format(loc)
                else:
                    if sim.pf.Npops > 1:
                        loc = 0
                        suffix = '{0}'
                    else:
                        loc = 0
                        suffix = ''

                hmf_pars = {'pop_Tmin{!s}'.format(suffix): sim.pf['pop_Tmin{!s}'.format(suffix)],
                    'fcoll{!s}'.format(suffix): copy.deepcopy(pops[loc].fcoll),
                    'dfcolldz{!s}'.format(suffix): copy.deepcopy(pops[loc].dfcolldz)}

                # Save for future iterations
                fcoll[i_Tmin] = hmf_pars.copy()

                p.update(hmf_pars)
            # If we already have matching fcoll splines, use them!
            elif self.reuse_splines and (not self.phenomenological):

                hmf_pars = {'pop_Tmin{!s}'.format(suffix): fcoll[i_Tmin]['pop_Tmin{!s}'.format(suffix)],
                    'fcoll{!s}'.format(suffix): fcoll[i_Tmin]['fcoll{!s}'.format(suffix)],
                    'dfcolldz{!s}'.format(suffix): fcoll[i_Tmin]['dfcolldz{!s}'.format(suffix)]}
                p.update(hmf_pars)
            else:
                pass

            # Write this set of parameters to disk before running
            # so we can troubleshoot later if the run never finishes.
            procid = str(rank).zfill(3)
            fn = '{0!s}.{1!s}.checkpt.pkl'.format(self.prefix, procid)
            write_pickle_file(kw, fn, ndumps=1, open_mode='w',\
                safe_mode=False, verbose=False)
            fn = '{0!s}.{1!s}.checkpt.txt'.format(self.prefix, procid)
            with open(fn, 'w') as f:
                print("Simulation began: {!s}".format(time.ctime()), file=f)

            # Kill if model gets stuck
            if self.timeout is not None:
                signal.signal(signal.SIGALRM, self._handler)
                signal.alarm(self.timeout)

            ##
            # Run simulation!
            ##
            blobs, _failct = self._run_sim(kw, p)
            failct += _failct

            # Disable the alarm
            if self.timeout is not None:
                signal.alarm(0)

            # If this is missing from a file, we'll know where things went south.
            fn = '{0!s}.{1!s}.checkpt.txt'.format(self.prefix, procid)
            with open(fn, 'a') as f:
                print("Simulation finished: {!s}".format(time.ctime()), file=f)

            chain = np.array([kwargs[key] for key in self.parameters])
            chain_all.append(chain)
            blobs_all.append(blobs)

            ct += 1

            ##
            # File I/O from here on out
            ##

            pb.update(ct)

            # Only record results every save_freq steps
            if ct % save_freq != 0:
                del p, chain, blobs
                gc.collect()
                continue

            # Not all processors will hit the final checkpoint exactly,
            # which can make collective I/O difficult. Hence the existence
            # of the will_hit_final_checkpoint and wont_hit_final_checkpoint
            # attributes

            if rank == 0 and use_checks:
                print("Checkpoint #{0}: {1!s}".format(ct // save_freq,\
                    time.ctime()))

            # First assemble data from all processors?
            # Analogous to assembling data from all walkers in MCMC
            write_pickle_file(chain_all,\
                '{!s}.chain.pkl'.format(prefix_by_proc), ndumps=1,\
                open_mode='a', safe_mode=False, verbose=False)

            self.save_blobs(blobs_all, False, prefix_by_proc)

            del p, chain, blobs
            del chain_all, blobs_all
            gc.collect()

            chain_all = []; blobs_all = []

            # If, after the first checkpoint, we only have 'failed' models,
            # raise an error.
            if (ct == failct) and self._exit_if_fail_streak:
                raise ValueError('Only failed models up to first checkpoint!')

            # This is meant to prevent crashes due to memory fragmentation.
            # For it to work (when we run for a really long time), we need
            # to write a shell script that calls the .py script that
            # executes ModelGrid.run many times, with each call setting
            # exit_after to 1 or perhaps a few, depending on the amount
            # of memory on hand. This is apparently less of an issue in Python 3.3
            if exit_after is not None:
                if exit_after == (ct // save_freq):
                    break

        pb.finish()

        # Need to make sure we write results to disk if we didn't
        # hit the last checkpoint
        if chain_all:
            write_pickle_file(chain_all,\
                '{!s}.chain.pkl'.format(prefix_by_proc), ndumps=1,\
                open_mode='a', safe_mode=False, verbose=False)

        if blobs_all:
            self.save_blobs(blobs_all, False, prefix_by_proc)

        print("Processor {0}: Wrote {1!s}.*.pkl ({2!s})".format(rank, prefix,\
            time.ctime()))

        # You. shall. not. pass.
        # Maybe unnecessary?
        if size > 1:
            MPI.COMM_WORLD.Barrier()

        t2 = time.time()

        ##
        # FINAL INFO
        ##
        if rank == 0:
            print("Calculation complete: {!s}".format(time.ctime()))
            dt = t2 - t1
            if dt > 3600:
                print("Elapsed time (hr)   : {0:.3g}".format(dt / 3600.))
            else:
                print("Elapsed time (min)  : {0:.3g}".format(dt / 60.))

    @property
    def Tmin_in_grid(self):
        """
        Determine if Tmin is an axis in our model grid.
        """

        if not hasattr(self, '_Tmin_in_grid'):

            ct = 0
            name = None
            self._Tmin_in_grid = False
            for par in self.grid.axes_names:

                if re.search('Tmin', par):
                    ct += 1
                    self._Tmin_in_grid = True
                    name = par

            self.Tmin_ax_name = name

            if ct > 1:
                raise NotImplemented('Trouble w/ multiple Tmin axes!')

        return self._Tmin_in_grid

    @property
    def nwalkers(self):
        # Each processor writes its own data
        return 1

    @property
    def assignments(self):
        if not hasattr(self, '_assignments'):
            #if hasattr(self, 'grid'):
            #    if self.grid.structured:
            #        self._structured_balance(method=0)
            #        return

            self.LoadBalance()

        return self._assignments

    @assignments.setter
    def assignments(self, value):
        self._assignments = value

    @property
    def load(self):
        if not hasattr(self, '_load'):
            self._load = [np.array(self.assignments == i).sum() \
                for i in range(size)]

            self._load = np.array(self._load)

        return self._load

    @property
    def LB(self):
        if not hasattr(self, '_LB'):
            self._LB = 0

        return self._LB

    @LB.setter
    def LB(self, value):
        self._LB = value

    def _balance_via_grouping(self, par):
        pass

    def _balance_via_sorting(self, par):
        pass

    def LoadBalance(self, method=0, par=None):

        if self.grid.structured:
            self._structured_balance(method=method, par=par)
        else:
            self._unstructured_balance(method=method, par=par)

    def _unstructured_balance(self, method=0, par=None):

        if rank == 0:

            order = list(np.arange(size))
            self._assignments = []
            while len(self.assignments) < self.grid.size:
                self._assignments.extend(order)

            self._assignments = np.array(self._assignments[0:self.grid.size])

            if size == 1:
                self.LB = 0
                return

            # Communicate assignments to workers
            for i in range(1, size):
                MPI.COMM_WORLD.Send(self._assignments, dest=i, tag=10*i)

        else:
            self._assignments = np.empty(self.grid.size, dtype=np.int)
            MPI.COMM_WORLD.Recv(self._assignments, source=0,
                tag=10*rank)

        self.LB = 0

    def _structured_balance(self, method=0, par=None):
        """
        Determine which processors are to run which models.

        Parameters
        ----------
        method : int
            0 : OFF
            1 : Minimize the number of values of `par' each processor gets.
                Good for, e.g., Tmin.
            2 : Maximize the number of values of `par' each processor gets.
                Useful if increasing `par' slows down the calculation.

        Returns
        -------
        Nothing. Creates "assignments" attribute, which has the same shape
        as the grid, with each element the rank of the processor assigned to
        that particular model.

        """

        self.LB = method

        if size == 1:
            self._assignments = np.zeros(self.grid.shape, dtype=int)
            return

        if method in [1, 2]:
            assert par in self.grid.axes_names, \
                "Supplied load-balancing parameter {!s} not in grid!".format(par)

            par_i = self.grid.axes_names.index(par)
            par_ax = self.grid.axes[par_i]
            par_N = par_ax.size
        else:
            par_N = np.inf

        if method not in [0, 1, 2, 3]:
            raise NotImplementedError('Unrecognized load-balancing method {}'.format(method))

        # No load balancing. Equal # of models per processor
        if method == 0 or (par_N < size):

            k = 0
            tmp_assignments = np.zeros(self.grid.shape, dtype=int)
            for loc, value in np.ndenumerate(tmp_assignments):

                if k % size != rank:
                    k += 1
                    continue

                tmp_assignments[loc] = rank

                k += 1

            # Communicate results
            self._assignments = np.zeros(self.grid.shape, dtype=int)
            MPI.COMM_WORLD.Allreduce(tmp_assignments, self._assignments)

        # Load balance over expensive axis
        elif method in [1, 2]:

            self._assignments = np.zeros(self.grid.shape, dtype=int)

            slc = [slice(0,None,1) for i in range(self.grid.Nd)]

            k = 0 # only used for method 1

            # Disclaimer: there's a probably a much more slick way of doing this

            # For each value of the input 'par', split up the work.
            # If method == 1, make it so that each processor gets only a
            # small subset of values for that parameter (e.g., sensible
            # for pop_Tmin), or method == 2 make it so that all processors get
            # a variety of values of input parameter, which is useful when
            # increasing values of this parameter slow down the calculation.
            for i in range(par_N):

                # Ellipses in all dimensions except that corresponding to a
                # particular value of input 'par'
                slc[par_i] = i

                if method == 1:
                    self._assignments[slc] = k \
                        * np.ones_like(self._assignments[slc], dtype=int)

                    # Cycle through processor numbers
                    k += 1
                    if k == size:
                        k = 0
                elif method == 2:
                    tmp = np.ones_like(self._assignments[slc], dtype=int)

                    leftovers = tmp.size % size

                    assign = np.arange(size)
                    arr = np.array([assign] * int(tmp.size // size)).ravel()
                    if leftovers != 0:
                        # This could be a little more efficient
                        arr = np.concatenate((arr, assign[0:leftovers]))

                    self._assignments[slc] = np.reshape(arr, tmp.size)
                else:
                    raise ValueError('No method={}!'.format(method))

        elif method == 3:

            # Do it randomly. Need to be careful in parallel.
            if rank != 0:
                buff = np.zeros(self.grid.dims, dtype=int)
            else:
                # Could do the assignment 100 times and pick the realization
                # with the most even distribution of work (as far as we
                # can tell a-priori), but eh.
                arr = np.random.randint(low=0, high=size, size=self.grid.size)
                buff = np.reshape(arr, self.grid.dims)

            self._assignments = np.zeros(self.grid.dims, dtype=int)
            nothing = MPI.COMM_WORLD.Allreduce(buff, self._assignments)

        else:
            raise ValueError('No method={}!'.format(method))
