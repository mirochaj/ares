"""

ModelGrid.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Dec  5 15:49:16 MST 2013

Description: For working with big model grids. Setting them up, running them,
and analyzing them.

"""

import numpy as np
import copy, os, pickle, gc, re
from ..simulations import Global21cm
from ..util import GridND, ProgressBar
from ..util.ReadData import read_pickled_dataset, read_pickled_dict

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
    def __init__(self, **kwargs):
        """
        Initialize a model grid.
        
        Parameters
        ----------

        prefix : str
            Will look for a file called <prefix>.grid.hdf5
        
        grid : instance

        verbose : bool

        """
        
        self.base_kwargs = def_kwargs.copy()
        self.base_kwargs.update(kwargs)
        
        # Prepare for blobs (optional)
        if 'inline_analysis' in self.base_kwargs:
            self.blob_names, self.blob_redshifts = \
                self.base_kwargs['inline_analysis']


        #if isinstance(grid, GridND):
        #    self.grid = grid
        #elif (prefix is not None):
        #    
        #    self.is_restart = False
        #    if os.path.exists('%s.grid.hdf5' % prefix):
        #        self.grid = GridND('%s.grid.hdf5' % prefix)
        #    elif restart:
        #        if os.path.exists('%s.grid.pkl' % prefix):
        #            try:
        #                self._read_restart('%s.grid.pkl' % prefix)
        #                self.is_restart = True
        #            except IOError:
        #                pass
        #            
        #    if os.path.exists('%s.extras.pkl' % prefix):
        #        self.extras = self.load_extras('%s.extras.pkl' % prefix)
        #
        #if not hasattr(self, 'grid'):
        #    self.grid = None
            
    def __getitem__(self, name):
        return self.grid[name]
                        
    def set_blobs(self):
        pass                    
        
    def _read_restart(self, prefix):
        """
        Figure out which models have already been run.
        """

        fails = read_pickled_dict('%s.fail.pkl' % prefix)
        chain = read_pickled_dataset('%s.chain.pkl' % prefix)
        
        f = open('%s.grid.pkl' % prefix, 'rb')
        axes = pickle.load(f)
        self.base_kwargs = pickle.load(f)
        f.close()
        
        # Prepare for blobs (optional)
        if 'inline_analysis' in self.base_kwargs:
            self.blob_names, self.blob_redshifts = \
                self.base_kwargs['inline_analysis']
        
        self.set_axes(**axes)

        # Array of ones/zeros: has this model already been done?
        self.done = np.zeros(self.grid.shape)

        for link in chain:
            kw = {par : link[i] \
                for i, par in enumerate(self.grid.axes_names)}
            
            kvec = self.grid.locate_entry(kw)
            
            self.done[kvec] = 1
            
        for fail in fails:
            
            kvec = self.grid.locate_entry(fail)
            
            self.done[kvec] = 1

    def set_axes(self, **kwargs):
        """
        Create GridND instance, construct N-D parameter space.

        Parameters
        ----------

        """

        self.grid = GridND()

        if rank == 0:
            print "Building parameter space..."

        # Build parameter space
        self.grid.build(**kwargs)

        # Save for later access
        self.kwargs = kwargs

        # Shortcut to parameter names
        self.parameters = self.grid.axes_names

    @property
    def to_solve(self):
        if not hasattr(self, '_to_solve'):
            self._to_solve = self.load_balance()

        return self._to_solve

    @property
    def is_log(self):
        if not hasattr(self, '_is_log'):
            self._is_log = [False] * self.grid.Nd
        
        return self._is_log
        
    def prep_output_files(self, prefix, restart):
        """
        Stick this in utilities folder?
        """
        
        if rank != 0:
            return
        
        if restart:
            return
    
        # Main output: MCMC chains (flattened)
        f = open('%s.grid.pkl' % prefix, 'wb')
        pickle.dump(self.kwargs, f)
        pickle.dump(self.base_kwargs, f)
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

    def run(self, prefix, clobber=False, restart=False, save_freq=10):
        """
        Run model grid, for each realization thru a given turning point.
        
        Parameters
        ----------
        prefix : str
            Prefix for all output files.
        save_freq : int
            Number of steps to take before writing data to disk.
        clobber : bool
            Overwrite pre-existing files of the same prefix if one exists?
        restart : bool
            Append to pre-existing files of the same prefix if one exists?

        Returns
        -------
        
        """
        
        if not hasattr(self, 'blob_names'):
            raise IOError('If you dont save anything this will be a useless exercise!')

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
        
        # Print out how many models we have (left) to compute
        if rank == 0:
            if restart:
                print "Update: %i models down, %i to go." \
                    % (self.done.sum(), self.grid.size - self.done.sum())
            else:
                print 'Running %i-element model-grid.' % self.grid.size
                         
        # Make some blank files for data output                 
        self.prep_output_files(prefix, restart)                 
                            
        # Dictionary for hmf tables
        fcoll = {}

        # Initialize progressbar
        pb = ProgressBar(self.grid.size, 'grid')
        pb.start()

        ct = 0    
        chain_all = []; blobs_all = []

        # Loop over models, use StellarPopulation.update routine 
        # to speed-up (don't have to re-load HMF spline as many times)
        for h, kwargs in enumerate(self.grid.all_kwargs):

            # Where does this model live in the grid?
            kvec = self.grid.locate_entry(kwargs)

            # Skip if it's a restart and we've already run this model
            if restart:
                if self.done[kvec]:
                    pb.update(h)
                    continue

            # Grab Tmin index
            try:
                if self.LB > 0:
                    Tminax = self.grid.axes[self.grid.axisnum('Tmin')]
                    i_Tmin = Tminax.locate(kwargs['Tmin'])

                    # Follow load-balancing procedure
                    if self.to_solve[i_Tmin] != rank:
                        continue
                else:
                    i_Tmin = 0
                    if h % size != rank:
                        continue

            except AttributeError:
                self.LB = 0
                i_Tmin = 0

                if h % size != rank:
                    continue

            # Copy kwargs - may need updating with pre-existing lookup tables
            p = self.base_kwargs.copy()
            p.update(kwargs)
            
            # Create new splines if we haven't hit this Tmin yet in our model grid.
            if i_Tmin not in fcoll.keys():
                sim = Global21cm(**p)
                hmf_pars = {'Tmin': sim.pops.pops[0].pf['Tmin'],
                    'fcoll': copy.deepcopy(sim.pops.pops[0].fcoll), 
                    'dfcolldz': copy.deepcopy(sim.pops.pops[0].dfcolldz),
                    'd2fcolldz2': copy.deepcopy(sim.pops.pops[0].d2fcolldz2)}

                # Save for future iterations
                fcoll[i_Tmin] = hmf_pars.copy()

            # If we already have matching fcoll splines, use them!
            else:
                hmf_pars = {'Tmin': fcoll[i_Tmin]['Tmin'],
                    'fcoll': fcoll[i_Tmin]['fcoll'],
                    'dfcolldz': fcoll[i_Tmin]['dfcolldz'],
                    'd2fcolldz2': fcoll[i_Tmin]['d2fcolldz2']}
                p.update(hmf_pars)
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

                pb.update(h)
                continue

            ct += 1
            
            chain = np.array([kwargs[key] for key in self.parameters])
            
            chain_all.append(chain)
            blobs_all.append(sim.blobs)

            ##
            # File I/O from here on out
            ##

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
            
            # Send the key to the next processor
            if rank != (size-1):
                MPI.COMM_WORLD.Send(np.zeros(1), rank+1, tag=rank)
            
            del chain_all, blobs_all
            gc.collect()

            chain_all = []; blobs_all = []

            pb.update(h)

            del p, sim

        pb.finish()

        print "Processor %i done." % rank
                    
    def save_extras(self, prefix):
        """
        Write extra fields to disk (npz format).
        
        Use "lock and key" method to avoid overwriting file contents.
        """
        
        fn = prefix + '.extras.pkl'
    
        key = np.ones(1)
    
        # Wait for the key
        if rank != 0:
            MPI.COMM_WORLD.Recv(key, source=rank-1, tag=rank-1)
    
        if rank == 0:
            f = open('%s' % fn, 'wb')
        else:
            f = open('%s' % fn, 'ab')
        
        pickle.dump(self.extras, f)
        f.close()
        
        print 'Processor #%i: wrote to %s' % (rank, fn)
            
        # Pass the key to the next processor
        if rank != (size-1):
            MPI.COMM_WORLD.Send(key, dest=rank+1, tag=rank)

    def load_extras(self, fn):
        """
        If "extras" were saved in model-grid search, load them back in.
        """
        
        output = {}
        
        f = open(fn, 'rb')     
        
        while True:
            try:    
                output.update(pickle.load(f))
            except EOFError:
                break
                
        f.close()
                                
        if rank == 0:
            print 'Read %s' % fn
            
        return output
        
    def load_balance(self, method=1):
        """
        Load balance model grid run.
        
        Parameters
        ----------
        method : int
            0 : OFF
            1 : By Tmin, cleverly
            2 : By Tmin, randomly
            3 : By Tmin, in ascending order
            
        Returns
        -------

        """
        
        if 'Tmin' not in self.grid.axes_names:
            if rank == 0:
                print "Tmin not in grid axes. Load balancing OFF."
            
            method = 0
            to_solve = np.arange(size)
        else:
            # "expensive axis" over which to load balance
            exp_ax = self.grid.axis('Tmin')
            
        if method == 1:
            Tmods = exp_ax.size
            procs = np.arange(size)

            sequence = np.concatenate((procs, procs[-1::-1]))

            i = 0
            j = 0
            to_solve = np.zeros(exp_ax.size)
            while i < exp_ax.size:
                to_solve[i] = sequence[j]

                j += 1
                if j == len(sequence):
                    j = 0

                i += 1
                
            if Tmods % size != 0 and rank == 0:
                print "\nWARNING: If load-balancing over N Tmin points,"
                print "it is advantageous to run with M processors, where N is", 
                print "divisible by M."
                
        elif method == 2:
            if (Tmods / size) % 1 != 0:
                raise ValueError('Tast_pts must be divisible by Nprocs.')
            to_solve = list(np.arange(size)) * Tmods / size
            np.random.shuffle(to_solve)
        elif method == 3:
            Tmods = np.arange(exp_ax.size)
            procs = np.arange(size)
            to_solve = Tmods % size
        
        self.LB = method
        
        return to_solve
    
        


