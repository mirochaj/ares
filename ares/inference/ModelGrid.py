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
import numpy as np
import copy, os, gc, re, time
from .ModelFit import ModelFit
from ..simulations import Global21cm
from ..util import GridND, ProgressBar
from ..util.ReadData import read_pickle_file, read_pickled_dict

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1

def_kwargs = {'verbose': False, 'progress_bar': False}    

class ModelGrid(ModelFit):
    """Create an object for setting up and running model grids."""
    
    @property
    def tol(self):
        if not hasattr(self, '_tol'):
            self._tol = 1e-3
        return self._tol
    
    @property 
    def tanh(self):
        if not hasattr(self, '_tanh'):
            if 'tanh_model' in self.base_kwargs:
                if self.base_kwargs['tanh_model']:
                    self._tanh = True
                else:
                    self._tanh = False
            else:
                self._tanh = False
        return self._tanh                  
                                        
    @property
    def simulator(self):
        if not hasattr(self, '_simulator'):
            self._simulator = Global21cm
        return self._simulator
            
    def _read_restart(self, prefix):
        """
        Figure out which models have already been run.
        
        Parameters
        ----------
        prefix : str
            File prefix of files ending in *.pkl to be read in.
            
        """
        
        if os.path.exists('%s.%s.chain.pkl' % (prefix, str(rank).zfill(3))):
            save_by_proc = True
            prefix_by_proc = prefix + '.%s' % (str(rank).zfill(3))
        else:
            save_by_proc = False
            prefix_by_proc = prefix

        # Read in current status of model grid
        chain = read_pickle_file('%s.chain.pkl' % prefix_by_proc)

        # Read parameter info
        f = open('%s.pinfo.pkl' % prefix, 'rb')
        axes_names, is_log = pickle.load(f)
        f.close()
        
        # Prepare for blobs (optional)
        if os.path.exists('%s.setup.pkl' % prefix):
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
        
        # Figure out axes
        axes = {}
        for i in range(chain.shape[1]):
            axes[axes_names[i]] = np.unique(chain[:,i])

        if (not self.grid.structured):
            return

        # Array of ones/zeros: has this model already been done?
        self.done = np.zeros(self.grid.shape)
        
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

    def prep_output_files(self, restart, clobber):
        """
        Stick this in utilities folder?
        """
        
        if restart:
            return
        
        if self.save_by_proc:
            prefix_by_proc = self.prefix + '.%s' % (str(rank).zfill(3))
        else:
            prefix_by_proc = self.prefix
        
            if rank != 0:
                return
            
        prefix = self.prefix
        super(ModelGrid, self)._prep_from_scratch(clobber, 
            by_proc=self.save_by_proc)
    
        if os.path.exists('%s.logL.pkl' % prefix) and (rank == 0):
            os.remove('%s.logL.pkl' % prefix)

        for par in self.grid.axes_names:
            if re.search('Tmin', par):
                f = open('%s.fcoll.pkl' % prefix_by_proc, 'wb')
                f.close()
                break

    @property
    def simulator(self):
        if not hasattr(self, '_simulator'):
            from ..simulations import Global21cm
            self._simulator = Global21cm
        return self._simulator
    
    def run(self, prefix, clobber=False, restart=False, save_freq=500):
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
        
        if self.save_by_proc:
            prefix_by_proc = prefix + '.%s' % (str(rank).zfill(3))
        else:
            prefix_by_proc = prefix
                
        if os.path.exists('%s.chain.pkl' % prefix_by_proc) and (not clobber):
            # Root processor will setup files so be careful
            if (not self.save_by_proc) and (rank > 0):
                pass
            elif not restart:
                raise IOError('%s*.pkl exists! Remove manually, set clobber=True, or set restart=True to append.' 
                    % prefix_by_proc)

        if not os.path.exists('%s.chain.pkl' % prefix_by_proc) and restart:
            raise IOError("This can't be a restart, %s*.pkl not found." % prefix_by_proc)
        
        # Load previous results if this is a restart
        if restart:
            if (not self.save_by_proc) and (rank != 0):
                MPI.COMM_WORLD.Recv(np.zeros(1), rank-1, tag=rank-1)
                
            self._read_restart(prefix)
            
            if (not self.save_by_proc) and (rank != (size-1)):
                MPI.COMM_WORLD.Send(np.zeros(1), rank+1, tag=rank)
                
            if self.grid.structured:
                ct0 = self.done.sum()
            else:
                ct0 = 0

        else:
            ct0 = 0

        ct = 0
        
        if restart:
            tot = np.sum(self.assignments == rank)
            Nleft = tot - ct0
        else:
            Nleft = np.sum(self.assignments == rank)
                        
        if Nleft == 0:
            if rank == 0:
                print 'This model grid is complete.'
            return
        
        # Print out how many models we have (left) to compute
        
        if restart and self.grid.structured:
            print "Update (processor #%i): %i models down, %i to go." \
                % (rank, ct0, Nleft)
        
        elif rank == 0:
            if restart:
                print 'Expanding pre-existing model set with %i more models.' \
                    % self.grid.size
            else:
                print 'Running %i-element model grid.' % self.grid.size
                
        # Make some blank files for data output                 
        self.prep_output_files(restart, clobber)                 

        # Dictionary for hmf tables
        fcoll = {}

        # Initialize progressbar
        pb = ProgressBar(Nleft, 'grid')
        pb.start()
        
        if pb.has_pb:
            use_checks = False
        else:
            use_checks = True
        
        chain_all = []; blobs_all = []
        
        t1 = time.time()

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

            # Create new splines if we haven't hit this Tmin yet in our model grid.    
            if i_Tmin not in fcoll.keys() and (not self.tanh):
                sim = self.simulator(**p)
                
                self.sim = sim
                
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

            # If we already have matching fcoll splines, use them!
            elif not self.tanh:        
                                        
                hmf_pars = {'pop_Tmin%s' % suffix: fcoll[i_Tmin]['pop_Tmin%s' % suffix],
                    'fcoll%s' % suffix: fcoll[i_Tmin]['fcoll%s' % suffix],
                    'dfcolldz%s' % suffix: fcoll[i_Tmin]['dfcolldz%s' % suffix]}
                p.update(hmf_pars)
                sim = self.simulator(**p)
                
            else:
                sim = self.simulator(**p)
                
            # Write this set of parameters to disk before running 
            # so we can troubleshoot later if the run never finishes.
            procid = str(rank).zfill(3)
            fn = '%s.%s.checkpt.pkl' % (self.prefix, procid)
            with open(fn, 'wb') as f:
                pickle.dump(kw, f)
                
            # Kill if model gets stuck    
            if self.timeout is not None:
                signal.signal(signal.SIGALRM, self._handler)
                signal.alarm(self.timeout)
            
            # Run simulation!
            try:
                sim.run()
            except Exception:                                 
                # Write to "fail" file
                f = open('%s.%s.fail.pkl' % (self.prefix, str(rank).zfill(3)), 
                    'ab')
                pickle.dump(kw, f)
                f.close()

            # Disable the alarm
            if self.timeout is not None:
                signal.alarm(0)

            chain = np.array([kwargs[key] for key in self.parameters])

            chain_all.append(chain)
            blobs_all.append(sim.blobs)

            ct += 1

            ##
            # File I/O from here on out
            ##

            pb.update(ct)

            # Only record results every save_freq steps
            if ct % save_freq != 0:
                del p, sim
                gc.collect()
                continue

            if rank == 0 and use_checks:
                print "Checkpoint #%i: %s" % (ct / save_freq, time.ctime())

            if (not self.save_by_proc) and (rank != 0):
                # Here we wait until we get the key      
                MPI.COMM_WORLD.Recv(np.zeros(1), rank-1, tag=rank-1)
                    
            # First assemble data from all processors?
            # Analogous to assembling data from all walkers in MCMC
            f = open('%s.chain.pkl' % prefix_by_proc, 'ab')
            pickle.dump(chain_all, f)
            f.close()
            
            self.save_blobs(blobs_all, False, prefix_by_proc)
            
            # Send the key to the next processor
            if (not self.save_by_proc) and (rank != (size-1)):
                MPI.COMM_WORLD.Send(np.zeros(1), rank+1, tag=rank)

            del p, sim
            del chain_all, blobs_all
            gc.collect()

            chain_all = []; blobs_all = []

        pb.finish()

        # Need to make sure we write results to disk if we didn't 
        # hit the last checkpoint
        if (not self.save_by_proc) and (rank != 0):
            MPI.COMM_WORLD.Recv(np.zeros(1), rank-1, tag=rank-1)
    
        if chain_all:
            with open('%s.chain.pkl' % prefix_by_proc, 'ab') as f:
                pickle.dump(chain_all, f)
        
        if blobs_all:
            self.save_blobs(blobs_all, False, prefix_by_proc)
        
        print "Processor %i: Wrote %s.*.pkl (%s)" \
            % (rank, prefix, time.ctime())

        # Send the key to the next processor
        if (not self.save_by_proc) and (rank != (size-1)):
            MPI.COMM_WORLD.Send(np.zeros(1), rank+1, tag=rank)
  
        # You. shall. not. pass.
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
    def save_by_proc(self):
        if not hasattr(self, '_save_by_proc'):
            self._save_by_proc = True
        return self._save_by_proc
        
    @save_by_proc.setter
    def save_by_proc(self, value):
        self._save_by_proc = value
            
    @property
    def assignments(self):
        if not hasattr(self, '_assignments'):
            self._assignments = np.zeros(self.grid.size)
        return self._assignments
            
    @assignments.setter
    def assignments(self, value):
        self._assignments = value
        
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
    
    def LoadBalance(self, method=None, par=None):
                
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
            for i in range(1, size):    
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
            1 : By input parameter.
            2 : 
            
        Returns
        -------
        Nothing. Creates "assignments" attribute, which has the same shape
        as the grid, with each element the rank of the processor assigned to
        that particular model.
        
        """
        
        self.LB = method
        
        if size == 1:
            self.assignments = np.zeros(self.grid.shape)
            return
            
        if method > 0:
            assert par in self.grid.axes_names, \
                "Supplied load-balancing parameter %s not in grid!" % par  
        
            par_i = self.grid.axes_names.index(par)
            par_ax = self.grid.axes[par_i]
            par_N = par_ax.size  
        
        if method not in [0, 1, 2]:
            raise NotImplementedError('Unrecognized load-balancing method %i' % method)
                
        # No load balancing. Equal # of models per processor
        if method == 0 or (par_N < size):
            
            k = 0
            tmp_assignments = np.zeros(self.grid.shape)
            for loc, value in np.ndenumerate(tmp_assignments):

                if hasattr(self, 'done'):
                    if self.done[loc]:
                        continue

                if k % size != rank:
                    k += 1
                    continue

                tmp_assignments[loc] = rank    

                k += 1

            # Communicate results
            self.assignments = np.zeros(self.grid.shape)
            MPI.COMM_WORLD.Allreduce(tmp_assignments, self.assignments)
                        
        # Load balance over Tmin axis    
        elif method in [1, 2]:
            
            self.assignments = np.zeros(self.grid.shape)
                        
            slc = [Ellipsis for i in range(self.grid.Nd)]
            
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
                    self.assignments[slc] = k \
                        * np.ones_like(self.assignments[slc])
                
                    # Cycle through processor numbers    
                    k += 1
                    if k == size:
                        k = 0
                elif method == 2:
                    tmp = np.ones_like(self.assignments[slc])
                    arr = np.array([np.arange(size)] * int(tmp.size / size)).ravel()
                    self.assignments[slc] = np.reshape(arr, tmp.size)
                else:
                    raise ValueError('No method=%i!' % method)

        elif method == 3:
            # Do it randomly
            arr = np.random.randint(low=0, high=size, size=self.grid.size, 
                dtype=int)

            self.assignments = np.reshape(arr, self.grid.shape)
                    
        else:
            raise ValueError('No method=%i!' % method)

