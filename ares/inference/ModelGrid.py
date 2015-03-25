"""

ModelGrid.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Dec  5 15:49:16 MST 2013

Description: For working with big model grids. Setting them up, running them,
and analyzing them.

"""

import numpy as np
import copy, os, gc, re, time
from ..simulations import Global21cm
from ..util import GridND, ProgressBar
from ..analysis.InlineAnalysis import InlineAnalysis
from ..util.ReadData import read_pickle_file, read_pickled_dict
from ..util.SetDefaultParameterValues import _blob_names, _blob_redshifts

#try:
#    import cPickle as pickle
#except:
import pickle

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1

def_kwargs = {'track_extrema': True, 'verbose': False, 'progress_bar': False}    

class ModelGrid:
    """Create an object for setting up and running model grids."""
    def __init__(self, tol=1e-5, **kwargs):
        """
        Initialize a model grid.
        
        Parameters
        ----------
        tol : float
            Absolute tolerance that determines whether parameters of a 
            pre-existing grid are different than those in a new grid.
            Only applied on restarts.
        prefix : str
            Will look for a file called <prefix>.grid.hdf5
        
        grid : instance

        verbose : bool

        """
        
        self.tol = tol
        self.base_kwargs = def_kwargs.copy()
        self.base_kwargs.update(kwargs)
        
        self.tanh = False
        if 'tanh_model' in self.base_kwargs:
            if self.base_kwargs['tanh_model']:
                self.tanh = True
        
    @property
    def blob_names(self):
        if hasattr(self, '_blob_names'):
            return self._blob_names
                
        if 'auto_generate_blobs' in self.base_kwargs:
            kw = self.base_kwargs.copy()
                        
            sim = Global21cm(**kw)
            anl = InlineAnalysis(sim)
            
            self._blob_names, self._blob_redshifts = \
                anl.generate_blobs()
            
            del sim, anl
            
        elif 'inline_analysis' in self.base_kwargs:
            self._blob_names, self._blob_redshifts = \
                self.base_kwargs['inline_analysis']
        else:
            self._blob_names = _blob_names
            self._blob_redshifts = _blob_redshifts
            
        return self._blob_names
            
    @blob_names.setter
    def blob_names(self, value):
        self._blob_names = value        
            
    @property 
    def blob_redshifts(self):
        if hasattr(self, '_blob_redshifts'):
            return self._blob_redshifts
        
        names, z = self.blob_names    
            
        return self._blob_redshifts

    @blob_redshifts.setter
    def blob_redshifts(self, value):
        self._blob_redshifts = value

    def _read_restart(self, prefix):
        """
        Figure out which models have already been run.
        
        Parameters
        ----------
        prefix : str
            File prefix of files ending in *.pkl to be read in.
            
        """
        
        # Read in current status of model grid
        fails = read_pickled_dict('%s.fail.pkl' % prefix)
        chain = read_pickle_file('%s.chain.pkl' % prefix)

        # Read parameter info
        f = open('%s.pinfo.pkl' % prefix, 'rb')
        axes_names, is_log = pickle.load(f)
        f.close()
        
        # Prepare for blobs (optional)
        if os.path.exists('%s.binfo.pkl' % prefix):
            f = open('%s.binfo.pkl' % prefix, 'rb')
            n, z = pickle.load(f)
            f.close()
            
            self.blob_names = n
            self.blob_redshifts = z 
        
        if len(axes_names) != chain.shape[1]:
            raise ValueError('Cannot change dimensionality on restart!')
            
        if axes_names != self.grid.axes_names:
            raise ValueError('Cannot change axes variables on restart!')
        
        # Figure out axes
        axes = {}
        for i in range(chain.shape[1]):
            axes[axes_names[i]] = np.unique(chain[:,i])

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
            
        for fail in fails:
            
            kvec = self.grid.locate_entry(fail, tol=self.tol)
            
            if None in kvec:
                continue
            
            self.done[kvec] = 1
                
    def set_axes(self, **kwargs):
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
        self.parameters = self.grid.axes_names
            
    @property
    def is_log(self):
        if not hasattr(self, '_is_log'):
            self._is_log = [False] * self.grid.Nd
        
        return self._is_log
        
    @is_log.setter
    def is_log(self, value):
        assert(len(value) == len(self.parameters))
        self._is_log = value    
        
    def prep_output_files(self, prefix, restart):
        """
        Stick this in utilities folder?
        """
        
        if rank != 0:
            return
        
        if restart:
            return
    
        # Say what processor computed which models.
        # Really just to make sure load-balancing etc. is working
        f = open('%s.load.pkl' % prefix, 'wb')
        f.close()
    
        # Main output: MCMC chains (flattened)
        f = open('%s.chain.pkl' % prefix, 'wb')
        f.close()
        
        # Failed models
        f = open('%s.fail.pkl' % prefix, 'wb')
        f.close()
        
        # Parameter names and list saying whether they are log10 or not
        f = open('%s.pinfo.pkl' % prefix, 'wb')
        pickle.dump((self.grid.axes_names, self.is_log), f)
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

        if 'Tmin' in self.grid.axes_names:
            f = open('%s.fcoll.pkl' % prefix, 'wb')
            f.close()
        
        # Outputs for arbitrary meta-data blobs
        if hasattr(self, 'blob_names'):

            # File for blobs themselves
            f = open('%s.blobs.pkl' % prefix, 'wb')
            f.close()
            
            # Blob names and list of redshifts at which to track them
            f = open('%s.binfo.pkl' % prefix, 'wb')
            pickle.dump((self.blob_names, self.blob_redshifts), f)
            f.close()

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

        if os.path.exists('%s.chain.pkl' % prefix) and (not clobber):
            if not restart:
                raise IOError('%s exists! Remove manually, set clobber=True, or set restart=True to append.' 
                    % prefix)

        if not os.path.exists('%s.chain.pkl' % prefix) and restart:
            raise IOError("This can't be a restart, %s*.pkl not found." % prefix)
        
        # Load previous results if this is a restart
        if restart:
            if rank != 0:
                MPI.COMM_WORLD.Recv(np.zeros(1), rank-1, tag=rank-1)
                
            self._read_restart(prefix)
            
            if rank != (size-1):
                MPI.COMM_WORLD.Send(np.zeros(1), rank+1, tag=rank)
                
            # Re-run load-balancing
            self.LoadBalance(self.LB)

            ct0 = self.done.sum() 

        else:
            ct0 = 0

        ct = 0
        
        if restart:
            Nleft = self.grid.size - ct0
        else:
            Nleft = self.grid.size
                        
        if Nleft == 0:
            if rank == 0:
                print 'This model grid is complete.'
            return
        
        # Print out how many models we have (left) to compute
        if rank == 0:
            if restart:
                print "Update: %i models down, %i to go." % (ct0, Nleft)
            else:
                print 'Running %i-element model-grid.' % self.grid.size
                
        # Make some blank files for data output                 
        self.prep_output_files(prefix, restart)                 

        if not hasattr(self, 'LB'):
            self.LoadBalance(0)                    
                            
        # Dictionary for hmf tables
        fcoll = {}

        # Initialize progressbar
        pb = ProgressBar(Nleft, 'grid')
        pb.start()

        chain_all = []; blobs_all = []; load_all = []

        # Loop over models, use StellarPopulation.update routine 
        # to speed-up (don't have to re-load HMF spline as many times)
        for h, kwargs in enumerate(self.grid.all_kwargs):

            # Where does this model live in the grid?
            if self.grid.structured:
                kvec = self.grid.locate_entry(kwargs)
            else:
                kvec = h

            if restart:
                pb_i = min(ct * size, Nleft - 1)
            else:
                pb_i = h
            
            # Skip if it's a restart and we've already run this model
            if restart:
                if self.done[kvec]:
                    pb.update(pb_i)
                    continue

            # Skip if this processor isn't assigned to this model        
            if self.assignments[kvec] != rank:
                pb.update(pb_i)
                continue

            # Grab Tmin index
            if self.Tmin_in_grid:
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
                sim = Global21cm(**p)
                
                if hasattr(self, 'Tmin_ax_popid'):
                    loc = self.Tmin_ax_popid
                    suffix = '{%i}' % loc
                else:
                    loc = 0
                    suffix = ''
                                
                hmf_pars = {'Tmin%s' % suffix: sim.pf['Tmin%s' % suffix],
                    'fcoll%s' % suffix: copy.deepcopy(sim.pops.pops[loc].fcoll), 
                    'dfcolldz%s' % suffix: copy.deepcopy(sim.pops.pops[loc].dfcolldz)}

                # Save for future iterations
                fcoll[i_Tmin] = hmf_pars.copy()

            # If we already have matching fcoll splines, use them!
            elif not self.tanh:        
                                        
                hmf_pars = {'Tmin%s' % suffix: fcoll[i_Tmin]['Tmin%s' % suffix],
                    'fcoll%s' % suffix: fcoll[i_Tmin]['fcoll%s' % suffix],
                    'dfcolldz%s' % suffix: fcoll[i_Tmin]['dfcolldz%s' % suffix]}
                p.update(hmf_pars)
                sim = Global21cm(**p)
                
            else:
                sim = Global21cm(**p)

            # Run simulation!
            try:
                sim.run()

                tps = sim.turning_points
            
            # Timestep error
            except SystemExit:
 
                sim.run_inline_analysis()
                tps = sim.turning_points
                
            except:         
                # Write to "fail" file - this might cause problems in parallel
                f = open('%s.fail.pkl' % self.prefix, 'ab')
                pickle.dump(kwargs, f)
                f.close()

                del p, sim
                gc.collect()

                pb.update(pb_i)
                ct += 1
                continue

            ct += 1
            
            chain = np.array([kwargs[key] for key in self.parameters])
            
            chain_all.append(chain)
            blobs_all.append(sim.blobs)
            load_all.append(rank)

            del p, sim
            gc.collect()

            ##
            # File I/O from here on out
            ##
            
            pb.update(ct)
            
            # Only record results every save_freq steps
            if ct % save_freq != 0:
                continue

            # Here we wait until we get the key
            if rank != 0:
                MPI.COMM_WORLD.Recv(np.zeros(1), rank-1, tag=rank-1)

            f = open('%s.chain.pkl' % self.prefix, 'ab')
            pickle.dump(chain_all, f)
            f.close()
            
            f = open('%s.blobs.pkl' % self.prefix, 'ab')
            pickle.dump(blobs_all, f)
            f.close()
            
            f = open('%s.load.pkl' % self.prefix, 'ab')
            pickle.dump(load_all, f)
            f.close()

            # Send the key to the next processor
            if rank != (size-1):
                MPI.COMM_WORLD.Send(np.zeros(1), rank+1, tag=rank)
            
            del chain_all, blobs_all, load_all
            gc.collect()

            chain_all = []; blobs_all = []; load_all = []

        pb.finish()

        # Need to make sure we write results to disk if we didn't 
        # hit the last checkpoint
        if rank != 0:
            MPI.COMM_WORLD.Recv(np.zeros(1), rank-1, tag=rank-1)
    
        if chain_all:
            f = open('%s.chain.pkl' % self.prefix, 'ab')
            pickle.dump(chain_all, f)
            f.close()
        
        if blobs_all:
            f = open('%s.blobs.pkl' % self.prefix, 'ab')
            pickle.dump(blobs_all, f)
            f.close()
        
        if load_all:
            f = open('%s.load.pkl' % self.prefix, 'ab')
            pickle.dump(load_all, f)
            f.close()            
        
        # Send the key to the next processor
        if rank != (size-1):
            MPI.COMM_WORLD.Send(np.zeros(1), rank+1, tag=rank)
        
        print "Processor %i done." % rank
        
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

                if par == 'Tmin':
                    ct += 1
                    self._Tmin_in_grid = True
                    name = par
                    continue

                if not re.search(par, 'Tmin'):
                    continue

                # Look for populations
                m = re.search(r"\{([0-9])\}", par)
            
                if m is None:
                    continue
            
                # Population ID number
                num = int(m.group(1))
                self.Tmin_ax_popid = num
            
                # Pop ID including curly braces
                prefix = par.strip(m.group(0))
                
                if prefix == 'Tmin':
                    ct += 1
                    self._Tmin_in_grid = True
                    name = par
                    continue
                    
            self.Tmin_ax_name = name
            
            if ct > 1:
                raise NotImplemented('Trouble w/ multiple Tmin axes!')
                
        return self._Tmin_in_grid
            
    def LoadBalance(self, method=0):
        
        if self.grid.structured:
            self._structured_balance(method=method)       
        else: 
            self._unstructured_balance(method=method)       
            
    def _unstructured_balance(self, method=0):
        
        if rank == 0:

            order = list(np.arange(size))
            self.assignments = []
            while len(self.assignments) < self.grid.size:
                self.assignments.extend(order)
                
            self.assignments = np.array(self.assignments[0:self.grid.size])
            
            if size == 1:
                return
            
            # Communicate assignments to workers
            for i in range(1, size):    
                MPI.COMM_WORLD.Send(self.assignments, dest=i, tag=10*i)    

        else:
            self.assignments = np.empty(self.grid.size, dtype=np.int)    
            MPI.COMM_WORLD.Recv(self.assignments, source=0,  
                tag=10*rank)
                   
        self.LB = 0                
                        
    def _structured_balance(self, method=0):
        """
        Determine which processors are to run which models.
        
        Parameters
        ----------
        method : int
            0 : OFF
            1 : By Tmin, cleverly
            
        Returns
        -------
        Nothing. Creates "assignments" attribute, which has the same shape
        as the grid, with each element the rank of the processor assigned to
        that particular model.
        
        """
        
        self.LB = True
        
        if size == 1:
            self.assignments = np.zeros(self.grid.shape)
            return
            
        have_Tmin = self.Tmin_in_grid  
        
        if have_Tmin:
            Tmin_i = self.grid.axes_names.index(self.Tmin_ax_name)
            Tmin_ax = self.grid.axes[Tmin_i]
            Tmin_N = Tmin_ax.size  
        
        # No load balancing. Equal # of models per processor
        if method == 0 or (not have_Tmin) or (Tmin_N < size):
            
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

            self.LB = False 
                        
        # Load balance over Tmin axis    
        elif method == 1:
            
            Tmin_slc = []
            
            for i in range(self.grid.Nd):
                if i == Tmin_i:
                    Tmin_slc.append(i)
                else:
                    Tmin_slc.append(Ellipsis)
            
            Tmin_slc = tuple(Tmin_slc)
            
            procs = np.arange(size)
                
            self.assignments = np.zeros(self.grid.shape)
            
            sequence = np.concatenate((procs, procs[-1::-1]))
            
            slc = [Ellipsis for i in range(self.grid.Nd)]
            
            k = 0
            for i in range(Tmin_N):
                
                slc[Tmin_i] = i
                
                self.assignments[slc] = k \
                    * np.ones_like(self.assignments[slc])
            
                k += 1
                if k == (len(sequence) / 2):
                    k = 0

        else:
            raise ValueError('No method=%i!' % method)

