"""

ModelGrid.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Dec  5 15:49:16 MST 2013

Description: For working with big model grids. Setting them up, running them,
and analyzing them.

"""

import numpy as np
import time, copy, os, pickle
from ..simulations import Global21cm
from ..util import GridND, ProgressBar

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1
    
try:
    from scipy.interpolate import interp1d
except ImportError:
    pass    
    
class ModelGrid:
    """Create an object for setting up and running model grids."""
    def __init__(self, prefix=None, grid=None, verbose=True, **kwargs):
        """
        Parameters
        ----------
        
        prefix : str
            Will look for a file called <prefix>.grid.hdf5
        
        grid : instance

        verbose : bool

        """

        self.verbose = verbose
            
        if isinstance(grid, GridND):
            self.grid = grid
        elif (prefix is not None):
            
            self.is_restart = False
            if os.path.exists('%s.grid.hdf5' % prefix):
                self.grid = GridND('%s.grid.hdf5' % prefix)
            else:
                try:
                    self._read_restart('%s.grid.pkl' % prefix)
                    self.is_restart = True
                except IOError:
                    pass
                    
            if os.path.exists('%s.extras.pkl' % prefix):
                self.extras = self.load_extras('%s.extras.pkl' % prefix)

        if not hasattr(self, 'grid'):
            self.grid = None

    def __getitem__(self, name):
        return self.grid[name]
                        
    def set_blobs(self):
        pass                    
        
    def _read_restart(self, fn):
        f = open(fn, 'rb')

        self.grid = pickle.load(f)

        indices, buffers, extras = [], [], []
        while True:
            try:
                tmp1, tmp2, tmp3 = pickle.load(f)
            except EOFError:
                break
                
            indices.append(tmp1)
            buffers.append(tmp2)
            extras.append(tmp3)            
                
        f.close()    
        
        self.indices = indices
        self.buffers = buffers
        self.extras = extras
                        
    def set_axes(self, **kwargs):
        """
        Create GridND instance, construct N-D parameter space.

        Parameters
        ----------
        
        """

        self.grid = GridND()
        self.bgrid = GridND()

        if rank == 0:
            print "Building parameter space..."

        # Build parameter space
        self.grid.build(**kwargs)
        self.bgrid.build(**kwargs)
        
        # Save for later access
        self.kwargs = kwargs

    @property
    def to_solve(self):
        if not hasattr(self, '_to_solve'):
            self._to_solve = self.load_balance()
        
        return self._to_solve
        
    def run(self, prefix, thru=None, save_fields=None, blobs=None, 
        **pars):
        """
        Run model grid, for each realization thru a given turning point.
        
        Parameters
        ----------
        prefix : str
            Prefix for output files. Suffixes will be .hdf5, blobs.hdf5, 
            extras.hdf5.
        save_fields : list
            Elements in glorb.Simulation.history to save (in their entirety).
            
        Returns
        -------
        
        """

        fn = '%s.grid.hdf5' % prefix
        
        # Create grid, or load in preexisting one
        if os.path.exists(fn):
            if rank == 0:
                print "File %s exists! Exiting." % fn
            return  

        self.is_restart = False
        if os.path.exists('%s.grid.pkl' % prefix):
            if rank == 0:
                print "Re-starting from %s.grid.pkl" % prefix
            self.is_restart = True

        if rank == 0:
            if self.is_restart:
                print "Update: %i models down, %i to go." \
                    % (len(self.indices), self.grid.size - len(self.indices))
            else:
                print 'Running %i-element model-grid.' % self.grid.size
                
        if blobs is not None:
            self.blob_names, self.blob_redshifts = blobs
            
            self.blob_shape = copy.deepcopy(self.grid.shape)
            self.blob_shape.append(len(self.blob_names))
            del blobs
        else:
            self.blob_names = ['z', 'dTb']
            self.blob_redshifts = list('BCD')
            
        # To store results
        shape = copy.deepcopy(self.grid.shape)
        
        # redshift and temperature of each turning pt.
        shape.append(2)
        
        # Container for elements of sim.history
        buffers = []        
        extras = {}
        blobs = []

        for option in self.blob_redshifts:
            buffers.append(np.zeros(shape))
            
            #if hasattr(self, 'blob_names'):
            #    blobs.append(np.zeros(self.blob_shape))
        
            if thru == option:
                break
                
        # Load checkpoints 
        if self.is_restart:
            for i, loc in enumerate(self.indices):
                for j in range(len(self.buffers[i])):
                    buffers[j][loc] = self.buffers[i][j]
            
        # Make sure we track the extrema, be less verbose, etc.
        if 'progress_bar' not in pars:
            pars.update({'progress_bar': False})
        if 'verbose' not in pars:
            pars.update({'verbose': False})
        if hasattr(self, 'blob_names'): 
            pars.update({'inline_analysis': \
                (self.blob_names, self.blob_redshifts)})
        
        pars.update({'track_extrema': True})
                                        
        # Dictionary for hmf tables
        fcoll = {}

        # Initialize progressbar
        pb = ProgressBar(self.grid.size, 'grid')
        pb.start()

        start = time.time()
        
        # Open checkpoint file
        f = open('%s.grid.pkl' % prefix, 'ab')
        if not self.is_restart:
            pickle.dump(self.grid, f)
            
        # Loop over models use StellarPopulation.update routine 
        # to speed-up (don't have to re-load HMF spline as many times)
        for h, kwargs in enumerate(self.grid.all_kwargs):
            
            # Where does this model live in the grid?
            kvec = self.grid.locate_entry(kwargs)
            
            # Skip if it's a restart and we've already run this model
            if self.is_restart:
                if kvec in self.indices:
                    pb.update(h)
                    continue

            # See if this combination of parameter values are "allowed,"
            # meaning their values are consistent with our fiducial model.
            # If not, continue. Fill value in buffers should be -99999 or
            # something
            #if thru != 'B' and False:
            #    B_loc = self.grid_B.locate_entry(kwargs)
            #    if self.L[B_loc] < self.Lcut * self.Lmax:
            #        if self.verbose:
            #            print '############################################'
            #            print '######  Low-Likelihood Model (#%s)  ####' % (str(h).zfill(6))
            #            for kw in kwargs:
            #                print "######  %s = %g" % (kw, kwargs[kw])
            #            print '############################################\n'
            #        continue

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
            p = pars.copy()
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
            fail = False
            try:             
                sim.run()
            except:
                fail = True
                
                if self.verbose:
                    print '###########################################'
                    print '######  Simulation failed (#%s)  ####' % (str(h).zfill(8))
                    for kw in kwargs:
                        print "######  %s = %g" % (kw, kwargs[kw])
                    
                    print '###########################################'
                    
                    if hasattr(sim, 'turning_points'):
                        if sim.turning_points:
                            last = 1e3
                            TP = None
                            for feature in sim.turning_points:
                                if sim.turning_points[feature][0] < last:
                                    TP = feature
                            print '######  Failed at %s' % TP
                    else:
                        print '######  Failed prior to first extremum. ' 
                    
                    print '###########################################'

            # Save turning point position
            tmp = []
            for i, redshift in enumerate(self.blob_redshifts):
                if redshift == 'trans':
                    name = 'trans'
                else:
                    name = redshift

                if fail:
                    buffers[i][kvec] = np.array([-11111] * 2)
                else:
                    
                    try:
                        buffers[i][kvec] = np.array(sim.turning_points[name][0:2])
                    except KeyError:
                        buffers[i][kvec] = np.array([-77777] * 2) # B never happens
                    except IndexError:
                        buffers[i][kvec] = np.array([-88888] * 2) # B happens @z=zfl, i.e. zfl too small
                    except AttributeError:
                        buffers[i][kvec] = np.array([-99999] * 2) # track_extrema=False!
                    
                tmp.append(buffers[i][kvec])    
                    
                # Save other stuff    
                #for j, blob in enumerate(sim.history.keys()):    
                #    
                #    if fail:
                #        blobs[i][kvec][j] = -11111
                #    else:
                #    
                #        try:
                #            blobs[i][kvec][blob_num] = sim.blobs[i,j]                            
                #        except KeyError:
                #            blobs[i][kvec][blob_num] = -77777
                #        except IndexError:
                #            blobs[i][kvec][blob_num] = -88888
                #        except AttributeError:
                #            blobs[i][kvec][blob_num] = -99999
                
                if name == thru:
                    break
                
            extras[kvec] = {}                        
            if save_fields is not None:
                extras[kvec]['z'] = sim.history['z']
                for field in save_fields:
                    extras[kvec][field] = sim.history[field]
                                        
            # Figure out blobs if we haven't already
            if not hasattr(self, 'blob_names'):
                self.blob_names = sim.history.keys()
            else:
                if len(sim.history.keys()) > len(self.blob_names):
                    self.blob_names = sim.history.keys()
                        
            pb.update(h)
            
            # Checkpoint!
            pickle.dump((kvec, tmp, extras[kvec]), f)

            del p, sim, tmp

        pb.finish()

        f.close()        

        print "Processor %i done." % rank
        
        self.blobs = blobs
        
        if hasattr(self, 'extras'):
            for i, key in enumerate(self.indices):
                extras[key] = self.extras[i]
                        
        self.extras = extras
            
        # Collect results
        if size > 1:
            collected_buffs = []
            for buff in buffers:
                tmp = np.zeros_like(buff)
                nothing = MPI.COMM_WORLD.Allreduce(buff, tmp)
            
                if rank == 0:
                    collected_buffs.append(tmp)
            
            collected_blobs = []
            for blob in blobs:
                tmp = np.zeros_like(blob)
                nothing = MPI.COMM_WORLD.Allreduce(blob, tmp)
            
                if rank == 0:
                    collected_blobs.append(tmp)        
        
        else:
            collected_buffs = buffers
            collected_blobs = blobs
                        
        if rank == 0:
            for i, redshift in enumerate(self.blob_redshifts):
                if redshift == 'trans':
                    name = 'trans'
                else:
                    name = redshift
                    
                self.grid.add_dataset(name, collected_buffs[i])
                self.grid.higher_dimensions = ['z', 'dTb']
                
                #self.bgrid.add_dataset(name, collected_blobs[i])
                #self.bgrid.higher_dimensions = self.blob_names
                
                if name == thru:
                    break

            if prefix is not None:
                self.grid.to_hdf5(fn)
                print "Wrote %s." % fn

                #self.bgrid.to_hdf5('%s.blobs.hdf5' % prefix)
                #print "Wrote %s.blobs.hdf5" % prefix

            if save_fields is not None:
                self.save_extras(prefix)

        stop = time.time()

        if rank == 0:
            print 'Grid calculation complete in %.4g hours.' % ((stop - start) / 3600.)

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
    
        


