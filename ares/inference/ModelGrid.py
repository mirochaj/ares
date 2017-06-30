"""

ModelGrid.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Dec  5 15:49:16 MST 2013

Description: For working with big model grids. Setting them up, running them,
and analyzing them.

"""

import signal
import pickle
import subprocess
import numpy as np
import copy, os, gc, re, time
from .ModelFit import ModelFit
from ..analysis import ModelSet
from ..simulations import Global21cm
from ..util import GridND, ProgressBar
from ..analysis import Global21cm as _AnalyzeGlobal21cm
from ..util.ReadData import read_pickle_file, read_pickled_dict

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1
        
def run_prog(prefix):
    
    pfn = '%s.fork_par.pkl' % prefix
    sfn = '%s.fork_sim' % prefix
        
    s = "f = open(\'%s\', \'rb\'); " % pfn
    s += "import pickle; pars = pickle.load(f); f.close(); "
    s += "import ares; sim = ares.simulations.Global21cm(**pars); "
    s += "sim.run(); sim.save(\'%s\', clobber=True)" % sfn
    
    #subprocess.call(s)
    os.system('python -c \"%s\"' % s)    
    
class ModelGrid(ModelFit):
    """Create an object for setting up and running model grids."""
    
    @property
    def tol(self):
        if not hasattr(self, '_tol'):
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
            
    def _read_restart(self, prefix):
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
            self.done = np.zeros(self.grid.shape)
        
        if os.path.exists('%s.%s.chain.pkl' % (prefix, str(rank).zfill(3))):
            prefix_by_proc = prefix + '.%s' % (str(rank).zfill(3))
        else:
            raise ValueError('This shouldn\'t happen anymore.')
            
        # Need to see if we're running with the same number of processors
        # We could probably set things up such that this isn't a problem.
        fn_by_proc = lambda proc: '%s.%s.chain.pkl' % (prefix, str(proc).zfill(3))
        fn_size_p1 = fn_by_proc(size+1)
        if os.path.exists(fn_size_p1):
            raise IOError('Original grid run with more processors!')

            proc_id = size + 1
            while os.path.exists(fn_by_proc(proc_id)):
                proc_id += 1
                continue

        # Read in current status of model grid, i.e., the old 
        # grid points.
        chain = read_pickle_file('%s.chain.pkl' % prefix_by_proc)

        # If we said this is a restart, but there are no elements in the 
        # chain, just run the thing. It probably means the initial run never
        # made it to the first checkpoint.

        if chain.size == 0:
            if rank == 0:
                print "Pre-existing chain file(s) empty. Running from beginning."
            if not self.grid.structured:
                self.done = np.array([0])
            
            return

        # Read parameter info
        f = open('%s.pinfo.pkl' % prefix, 'rb')
        axes_names, is_log = pickle.load(f)
        f.close()

        # Prepare for blobs (optional)
        if os.path.exists('%s.binfo.pkl' % prefix):
            f = open('%s.binfo.pkl' % prefix, 'rb')
            self.pf = pickle.load(f)
            f.close()
        elif os.path.exists('%s.setup.pkl' % prefix):
            f = open('%s.setup.pkl' % prefix, 'rb')
            self.pf = pickle.load(f)
            f.close()

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
            self.done = np.array([chain.shape[0]])
            return

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
            
            self.done[kvec] = 1
            
        return self.done
            
    @property            
    def axes(self):
        return self.grid.axes
    
    @axes.setter            
    def axes(self, kwargs):
        """
        Create GridND instance, construct N-D parameter space.
        """

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
        self.grid.axes_names = models[0].keys()
        self.grid.Nd = len(self.grid.axes_names)

        # Shortcut to parameter names
        if not hasattr(self, 'parameters'):
            self.parameters = self.grid.axes_names

    def _reshape_assignments(self, assignments):
        assign = []
        for h, kwargs in enumerate(self.grid.all_kwargs):

            # Where does this model live in the grid?
            if self.grid.structured:
                kvec = self.grid.locate_entry(kwargs)
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
                        
        prefix_by_proc = self.prefix + '.%s' % (str(rank).zfill(3))

        # Reshape assignments so it's Nlinks long.
        if self.grid.structured:
            assignments = self._reshape_assignments(self.assignments)
                
            if restart:
                if rank == 0:
                    f = open('%s.load.pkl' % self.prefix, 'ab')
                    pickle.dump(assignments, f)
                    f.close()
                
                return
        else:
            if restart:
                return
        
        if rank > 0:
            return
                        
        super(ModelGrid, self)._prep_from_scratch(clobber, by_proc=True)
            
        if self.grid.structured:    
            f = open('%s.load.pkl' % self.prefix, 'wb')
            pickle.dump(assignments, f)
            f.close()
    
        # ModelFit makes this file by default but grids don't use it.
        if os.path.exists('%s.logL.pkl' % self.prefix) and (rank == 0):
            os.remove('%s.logL.pkl' % self.prefix)

        for par in self.grid.axes_names:
            if re.search('Tmin', par):
                f = open('%s.fcoll.pkl' % prefix_by_proc, 'wb')
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
        return zip(*self.tricks)[0]
    
    @property
    def trick_files(self):
        return zip(*self.tricks)[1]    
            
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
        
    def _prep_tricks(self):
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
                    
                print "Ready to cheat!"    
                    
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

                        print k, par, kw[par]

                        score += np.abs(vals - kw[par])
                    
                    best = np.argmin(score)
                    
                    print zarr.shape, Mmin.shape, best
                    
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
            
    def _run_sim(self, kw, p):
        
        failct = 0
        sim = self.simulator(**p)
                        
        try:
            sim.run()            
            blobs = copy.deepcopy(sim.blobs)
        except RuntimeError:
            f = open('%s.%s.timeout.pkl' % (self.prefix, str(rank).zfill(3)), 'ab')
            pickle.dump(kw, f)
            f.close()
            
            blobs = copy.deepcopy(self.blank_blob)
        except MemoryError:
            raise MemoryError('This cannot be tolerated!')
        except:
            # For some reason "except Exception"  doesn't catch everything...
            # Write to "fail" file
            f = open('%s.%s.fail.pkl' % (self.prefix, str(rank).zfill(3)), 'ab')
            pickle.dump(kw, f)
            f.close()
            
            print "FAILURE: Processor #%i." % rank
            
            failct += 1
            
            blobs = copy.deepcopy(self.blank_blob)
            
        if 'feedback_LW_guess' in self.tricks:            
            try:
                self.trick_data['pop_Mmin{2}'] = \
                    np.interp(sim.pops[2].halos.z, 
                        sim.medium.field._zarr, sim.medium.field._Mmin_now)
            except AttributeError:
                del self.trick_data['pop_Mmin{2}']    
            
        del sim    
            
        return blobs, failct
    
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
        self.is_restart = restart
        self.save_freq = save_freq
        
        prefix_by_proc = prefix + '.%s' % (str(rank).zfill(3))
                
        # Kill this thing if we're about to delete files and we haven't 
        # set clobber=True        
        if os.path.exists('%s.chain.pkl' % prefix_by_proc) and (not clobber):
            if not restart:
                raise IOError('%s*.pkl exists! Remove manually, set clobber=True, or set restart=True to append.' 
                    % prefix_by_proc)
                    
        if not os.path.exists('%s.chain.pkl' % prefix_by_proc) and restart:
            if rank == 0:
                print "This can't be a restart, %s*.pkl not found." % prefix
                print "Starting from scratch..."
            restart = False

        # Load previous results if this is a restart
        if restart:
            self._read_restart(prefix)
            
            if self.grid.structured:
                done = int(self.done[self.done >= 0].sum())
            else:
                done = 0
                                
            # Important that this goes second, otherwise this processor
            # will count the models already run by other processors, which
            # will mess up the 'Nleft' calculation below.
            
            # Figure out what models have been run by *any* processor
            # in the old grid.
            if self.grid.structured:
                tmp = np.zeros(self.grid.shape)
                MPI.COMM_WORLD.Allreduce(self.done, tmp)
                self.done = tmp
            else:
                # In this case, self.done is just an integer
                tmp = np.array([0])
                MPI.COMM_WORLD.Allreduce(self.done, tmp)
                self.done = tmp[0]
        else:
            done = 0
        
        if restart and self.grid.structured:
            mine_and_done = np.logical_and(self.assignments == rank,
                                           self.done >= 1)
            
            Nleft = self.load[rank] - np.sum(mine_and_done)
        else:
            Nleft = self.load[rank]

        if Nleft == 0:
            if rank == 0:
                print 'This model grid is complete.'
            return

        # Print out how many models we have (left) to compute
        if restart and self.grid.structured:
            if rank == 0:
                Ndone = self.done[self.done >= 0].sum()
                Ntot = self.done.size
                print "Update               : %i models down, %i to go." \
                    % (Ndone, Ntot - Ndone)
            
            MPI.COMM_WORLD.Barrier()
                
            print "Update (processor #%i): Running %i more models." \
                % (rank, Nleft)

        elif rank == 0:
            if restart:
                print 'Re-starting pre-existing model set (%i models done already).' \
                    % self.done
                print 'Running %i more models.' % self.grid.size
            else:
                print 'Running %i-element model grid.' % self.grid.size

        # Make some blank files for data output                 
        self.prep_output_files(restart, clobber)        

        # Dictionary for hmf tables
        fcoll = {}
        
        # Yep
        #self._prep_tricks()

        # Initialize progressbar
        pb = ProgressBar(Nleft, 'grid', use_pb)
        pb.start()
        
        chain_all = []; blobs_all = []
        
        t1 = time.time()

        ct = 0
        failct = 0

        # Loop over models, use StellarPopulation.update routine 
        # to speed-up (don't have to re-load HMF spline as many times)
        for h, kwargs in enumerate(self.grid.all_kwargs):
            
            # Where does this model live in the grid?
            if self.grid.structured:
                kvec = self.grid.locate_entry(kwargs)
            else:
                kvec = h

            # Skip if it's a restart and we've already run this model
            if restart and self.grid.structured:
                if self.done[kvec]:
                    pb.update(ct)
                    continue

            # Skip if this processor isn't assigned to this model        
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
            
            #if 'guess_popIII_sfrds' in self.trick_names:
            #    trick = self.trick_funcs['guess_popIII_sfrds'](**kw)
            #                    
            #    import matplotlib.pyplot as pl
            #    print trick
            #    zarr = np.arange(6, 50)
            #    pl.semilogy(zarr, trick['pop_Mmin{2}'](zarr))
            #    raw_input('<enter>')
            #    
            #    #if score > 0:
            #    #    trick['feedback_LW_assume_perfect_guesses'] = False 
            #    #
            #    p.update(trick)
            
            # Create new splines if we haven't hit this Tmin yet in our model grid.    
            if self.reuse_splines and \
                i_Tmin not in fcoll.keys() and (not self.phenomenological):
                #raise NotImplementedError('help')
                sim = self.simulator(**p)
                               
                pops = sim.pops
                
                if hasattr(self, 'Tmin_ax_popid'):
                    loc = self.Tmin_ax_popid
                    suffix = '{%i}' % loc
                else:
                    if sim.pf.Npops > 1:
                        loc = 0
                        suffix = '{0}'
                    else:    
                        loc = 0
                        suffix = ''
                
                hmf_pars = {'pop_Tmin%s' % suffix: sim.pf['pop_Tmin%s' % suffix],
                    'fcoll%s' % suffix: copy.deepcopy(pops[loc].fcoll), 
                    'dfcolldz%s' % suffix: copy.deepcopy(pops[loc].dfcolldz)}
                
                # Save for future iterations
                fcoll[i_Tmin] = hmf_pars.copy()

                p.update(hmf_pars)
            # If we already have matching fcoll splines, use them!
            elif self.reuse_splines and (not self.phenomenological):
                                        
                hmf_pars = {'pop_Tmin%s' % suffix: fcoll[i_Tmin]['pop_Tmin%s' % suffix],
                    'fcoll%s' % suffix: fcoll[i_Tmin]['fcoll%s' % suffix],
                    'dfcolldz%s' % suffix: fcoll[i_Tmin]['dfcolldz%s' % suffix]}
                p.update(hmf_pars)
            else:
                pass
                
            # Write this set of parameters to disk before running 
            # so we can troubleshoot later if the run never finishes.
            procid = str(rank).zfill(3)
            fn = '%s.%s.checkpt.pkl' % (self.prefix, procid)
            with open(fn, 'wb') as f:
                pickle.dump(kw, f)
            fn = '%s.%s.checkpt.txt' % (self.prefix, procid)
            with open(fn, 'w') as f:
                print >> f, "Simulation began: %s" % time.ctime()

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
            fn = '%s.%s.checkpt.txt' % (self.prefix, procid)
            with open(fn, 'a') as f:
                print >> f, "Simulation finished: %s" % time.ctime()

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
                print "Checkpoint #%i: %s" % (ct / save_freq, time.ctime())
                
            # First assemble data from all processors?
            # Analogous to assembling data from all walkers in MCMC
            f = open('%s.chain.pkl' % prefix_by_proc, 'ab')
            pickle.dump(chain_all, f)
            f.close()

            self.save_blobs(blobs_all, False, prefix_by_proc)

            del p, chain, blobs
            del chain_all, blobs_all
            gc.collect()

            chain_all = []; blobs_all = []
            
            # If, after the first checkpoint, we only have 'failed' models,
            # raise an error.
            if ct == failct:
                raise ValueError('Only failed models up to first checkpoint!')
                
            # This is meant to prevent crashes due to memory fragmentation. 
            # For it to work (when we run for a really long time), we need
            # to write a shell script that calls the .py script that
            # executes ModelGrid.run many times, with each call setting
            # exit_after to 1 or perhaps a few, depending on the amount
            # of memory on hand. This is apparently less of an issue in Python 3.3
            if exit_after is not None:
                if exit_after == ct / save_freq:
                    break

        pb.finish()
           
        # Need to make sure we write results to disk if we didn't 
        # hit the last checkpoint
        if chain_all:
            with open('%s.chain.pkl' % prefix_by_proc, 'ab') as f:
                pickle.dump(chain_all, f)
        
        if blobs_all:
            self.save_blobs(blobs_all, False, prefix_by_proc)
        
        print "Processor %i: Wrote %s.*.pkl (%s)" \
            % (rank, prefix, time.ctime())

        # You. shall. not. pass.
        # Maybe unnecessary?
        MPI.COMM_WORLD.Barrier()
        
        t2 = time.time()

        ##
        # FINAL INFO 
        ##    
        if rank == 0:
            print "Calculation complete: %s" % time.ctime()
            dt = t2 - t1
            if dt > 3600:
                print "Elapsed time (hr)   : %.3g" % (dt / 3600.)
            else:    
                print "Elapsed time (min)  : %.3g" % (dt / 60.)
                
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
            self.LoadBalance()
            
        return self._assignments
            
    @assignments.setter
    def assignments(self, value):
        self._assignments = value
        
    @property
    def load(self):
        if not hasattr(self, '_load'):
            self._load = [np.array(self.assignments == i).sum() \
                for i in xrange(size)]
                
            self._load = np.array(self._load)    
        
        return self._load
        
    @property
    def will_hit_final_checkpoint(self):
        if not hasattr(self, '_will_hit_final_checkpoint'):
            self._will_hit_final_checkpoint = self.load % self.save_freq == 0
        
        return self._will_hit_final_checkpoint
    
    @property
    def wont_hit_final_checkpoint(self):
        if not hasattr(self, '_will_hit_final_checkpoint'):
            self._wont_hit_final_checkpoint = self.load % self.save_freq != 0
    
        return self._wont_hit_final_checkpoint
    
    @property
    def Ncheckpoints(self):
        if not hasattr(self, '_Ncheckpoints'):
            self._Ncheckpoints = self.load / self.save_freq
        return self._Ncheckpoints
        
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
            self.assignments = []
            while len(self.assignments) < self.grid.size:
                self.assignments.extend(order)
                
            self.assignments = np.array(self.assignments[0:self.grid.size])
            
            if size == 1:
                self.LB = 0
                return
            
            # Communicate assignments to workers
            for i in xrange(1, size):    
                MPI.COMM_WORLD.Send(self.assignments, dest=i, tag=10*i)    

        else:
            self.assignments = np.empty(self.grid.size, dtype=np.int)    
            MPI.COMM_WORLD.Recv(self.assignments, source=0,  
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
            self.assignments = np.zeros(self.grid.shape, dtype=int)
            return
            
        if method in [1, 2]:
            assert par in self.grid.axes_names, \
                "Supplied load-balancing parameter %s not in grid!" % par  
        
            par_i = self.grid.axes_names.index(par)
            par_ax = self.grid.axes[par_i]
            par_N = par_ax.size  
        else:
            par_N = np.inf    
        
        if method not in [0, 1, 2, 3]:
            raise NotImplementedError('Unrecognized load-balancing method %i' % method)
                
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
            self.assignments = np.zeros(self.grid.shape, dtype=int)
            MPI.COMM_WORLD.Allreduce(tmp_assignments, self.assignments)
                        
        # Load balance over expensive axis    
        elif method in [1, 2]:
            
            self.assignments = np.zeros(self.grid.shape, dtype=int)
                        
            slc = [slice(0,None,1) for i in xrange(self.grid.Nd)]
            
            k = 0 # only used for method 1
            
            # Disclaimer: there's a probably a much more slick way of doing this
                
            # For each value of the input 'par', split up the work.
            # If method == 1, make it so that each processor gets only a 
            # small subset of values for that parameter (e.g., sensible
            # for pop_Tmin), or method == 2 make it so that all processors get
            # a variety of values of input parameter, which is useful when
            # increasing values of this parameter slow down the calculation.
            for i in xrange(par_N):
                
                # Ellipses in all dimensions except that corresponding to a
                # particular value of input 'par'
                slc[par_i] = i
                
                if method == 1:
                    self.assignments[slc] = k \
                        * np.ones_like(self.assignments[slc], dtype=int)
                
                    # Cycle through processor numbers    
                    k += 1
                    if k == size:
                        k = 0
                elif method == 2:
                    tmp = np.ones_like(self.assignments[slc], dtype=int)
                    
                    leftovers = tmp.size % size
                    
                    assign = np.arange(size)
                    arr = np.array([assign] * int(tmp.size / size)).ravel()
                    if leftovers != 0:
                        # This could be a little more efficient
                        arr = np.concatenate((arr, assign[0:leftovers]))
                        
                    self.assignments[slc] = np.reshape(arr, tmp.size)
                else:
                    raise ValueError('No method=%i!' % method)

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
                            
            self.assignments = np.zeros(self.grid.dims, dtype=int)
            nothing = MPI.COMM_WORLD.Allreduce(buff, self.assignments)
                        
        else:
            raise ValueError('No method=%i!' % method)

