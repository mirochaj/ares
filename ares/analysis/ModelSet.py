"""

ModelFit.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Apr 28 11:19:03 MDT 2014

Description: For analysis of MCMC fitting.

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pl
from ..util import ProgressBar
import matplotlib._cntr as cntr
from ..physics import Cosmology
from .MultiPlot import MultiPanel
import re, os, string, time, glob
from .BlobFactory import BlobFactory
from matplotlib.patches import Rectangle
from ..physics.Constants import nu_0_mhz
from .MultiPhaseMedium import MultiPhaseMedium as aG21
from ..util import labels as default_labels
import matplotlib.patches as patches
from ..util.Aesthetics import Labeler
from ..util.PrintInfo import print_model_set
from .DerivedQuantities import DerivedQuantities as DQ
from ..util.ParameterFile import count_populations, par_info
from ..util.SetDefaultParameterValues import SetAllDefaults, TanhParameters
from ..util.Stats import Gauss1D, GaussND, error_2D, _error_2D_crude, \
    rebin, correlation_matrix
from ..util.ReadData import read_pickled_dict, read_pickle_file, \
    read_pickled_chain, read_pickled_logL, fcoll_gjah_to_ares, \
    tanh_gjah_to_ares

#try:
#    import dill as pickle
#except ImportError:
import pickle 

try:
    import shapely.geometry as geometry
    have_shapely = True
except ImportError, OSError:
    have_shapely = False
    
try:
    from descartes import PolygonPatch
    have_descartes = True
except ImportError:
    have_descartes = False    

try:
    import h5py
    have_h5py = True
except ImportError:
    have_h5py = False
    
try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1
    
default_mp_kwargs = \
{
 'diagonal': 'lower', 
 'keep_diagonal': True, 
 'panel_size': (0.5,0.5), 
 'padding': (0,0)
}    

numerical_types = [float, np.float64, np.float32, int, np.int32, np.int64]

# Machine precision
MP = np.finfo(float).eps

def patch_pinfo(pars):
    # This should be deprecated in future versions
    new_pars = []
    for par in pars:

        if par in tanh_gjah_to_ares:
            new_pars.append(tanh_gjah_to_ares[par])
        elif par in fcoll_gjah_to_ares:
            new_pars.append(fcoll_gjah_to_ares[par])
        else:
            new_pars.append(par)
    
    return new_pars

def err_str(label, mu, err, log, labels=None):
    s = undo_mathify(make_label(label, log, labels))

    s += '=%.3g^{+%.2g}_{-%.2g}' % (mu, err[1], err[0])
    
    return r'$%s$' % s

class ModelSubSet(object):
    def __init__(self):
        pass

class ModelSet(BlobFactory):
    def __init__(self, data, subset=None, verbose=True):
        """
        Parameters
        ----------
        data : instance, str
            prefix for a bunch of files ending in .chain.pkl, .pinfo.pkl, etc.,
            or a ModelSubSet instance.
        
        subset : list, str
            List of parameters / blobs to recover from individual files. Can
            also set subset='all', and we'll try to automatically track down
            all that are available.

        """
        
        self.subset = subset
                
        # Read in data from file (assumed to be pickled)
        if type(data) == str:
            
            # Check to see if perhaps this is just the chain
            if re.search('pkl', data):
                self._prefix_is_chain = True
                pre_pkl = data[0:data.rfind('.pkl')]
                self.prefix = prefix = pre_pkl
            else:
                self._prefix_is_chain = False
                self.prefix = prefix = data
            
            i = prefix.rfind('/') # forward slash index

            # This means we're sitting in the right directory already
            if i == - 1:
                self.path = './'
                self.fn = prefix
            else:
                self.path = prefix[0:i+1]
                self.fn = prefix[i+1:]

            if verbose:
                try:
                    print_model_set(self)
                except:
                    pass
                    
            #if not self.is_mcmc:
            #    
            #    self.grid = ModelGrid(**self.base_kwargs)
                
                #self.axes = {}
                #for i in range(self.chain.shape[1]):
                #    self.axes[self.parameters[i]] = np.unique(self.chain[:,i])
                #
                #self.grid.set_axes(**self.axes)
                #
            
        elif isinstance(data, ModelSubSet):
            self._chain = data.chain
            self._is_log = data.is_log
            self._base_kwargs = data.base_kwargs
            #self._fails = data.fails
            
            self.mask = np.zeros_like(data.blobs)    
            self.mask[np.isinf(data.blobs)] = 1
            self.mask[np.isnan(data.blobs)] = 1
            #self._blobs = np.ma.masked_array(data.blobs, mask=self.mask)

            #self._blob_names = data.blob_names
            #self._blob_redshifts = data.blob_redshifts
            #self._parameters = data.parameters
            #self._is_mcmc = data.is_mcmc
            
            #if self.is_mcmc:
            #    self.logL = data.logL
            #else:
            #    try:
            #        self.load = data.load
            #    except AttributeError:
            #        pass
            #    try:            
            #        self.axes = data.axes
            #    except AttributeError:
            #        pass
            #    try:
            #        self.grid = data.grid
            #    except AttributeError:
            #        pass

            #self.Nd = int(self.chain.shape[-1])       
                
        else:
            raise TypeError('Argument must be ModelSubSet instance or filename prefix')              
    
        self.have_all_blobs = os.path.exists('%s.blobs.pkl' % self.prefix)
    
        #self._pf = ModelSubSet()
        #self._pf.Npops = self.Npops
        
        self.derived_blobs = DQ(self)
    
        #try:
        #    self._fix_up()
        #except AttributeError:
        #    pass
            
    @property
    def mask(self):
        if not hasattr(self, '_mask'):
            if self.is_mcmc:
                self._mask = np.zeros_like(self.logL)
            else:
                self._mask = np.zeros(self.chain.shape[0])
        return self._mask
    
    @mask.setter
    def mask(self, value):
        if self.is_mcmc:
            assert len(value) == len(self.logL)
        self._mask = value

    @property
    def load(self):
        if not hasattr(self, '_load'):
            if os.path.exists('%s.load.pkl' % self.prefix):
                self._load = read_pickle_file('%s.load.pkl' % self.prefix)
            else:
                self._load = None

        return self._load

    @property
    def pf(self):
        return self.base_kwargs

    @property
    def base_kwargs(self):
        if not hasattr(self, '_base_kwargs'):
            if os.path.exists('%s.setup.pkl' % self.prefix):
                f = open('%s.setup.pkl' % self.prefix, 'rb')
                try:
                    self._base_kwargs = pickle.load(f)
                except:
                    self._base_kwargs = {}
                f.close()
                
            else:
                self._base_kwargs = None    
            
        return self._base_kwargs    

    @property
    def parameters(self):
        # Read parameter names and info
        if not hasattr(self, '_parameters'):
            if os.path.exists('%s.pinfo.pkl' % self.prefix):
                f = open('%s.pinfo.pkl' % self.prefix, 'rb')
                self._parameters, self._is_log = pickle.load(f)
                f.close()
                self._parameters = patch_pinfo(self._parameters)
            else:
                self._is_log = [False] * self.chain.shape[-1]
                self._parameters = ['p%i' % i \
                    for i in range(self.chain.shape[-1])]
        
            self._is_log = tuple(self._is_log)
            self._parameters = tuple(self._parameters)
        
        return self._parameters
        
    @property
    def nwalkers(self):
        # Read parameter names and info
        if not hasattr(self, '_nwalkers'):
            if os.path.exists('%s.rinfo.pkl' % self.prefix):
                f = open('%s.rinfo.pkl' % self.prefix, 'rb')
                self._nwalkers, self._save_freq, self._steps = \
                    map(int, pickle.load(f))
                f.close()
            else:
                self._nwalkers = self._save_freq = self._steps = None
    
        return self._nwalkers
    
    @property
    def save_freq(self):
        if not hasattr(self, '_save_freq'):
            nwalkers = self.nwalkers
        return self._save_freq
    
    @property
    def steps(self):
        if not hasattr(self, '_steps'):
            nwalkers = self.nwalkers
        return self._steps
    
    @property
    def priors(self):
        if not hasattr(self, '_priors'):   
            if os.path.exists('%s.priors.pkl' % self.prefix):
                f = open('%s.priors.pkl' % self.prefix, 'rb')
                self._priors = pickle.load(f)
                f.close() 
            else:
                self._priors = {}
                
        return self._priors    
        
    @property
    def is_log(self):
        if not hasattr(self, '_is_log'):
            pars = self.parameters
        
        return self._is_log
        
    @property
    def is_mcmc(self):
        if not hasattr(self, '_is_mcmc'):
            if os.path.exists('%s.logL.pkl' % self.prefix):
                self._is_mcmc = True
            else:
                self._is_mcmc = False

        return self._is_mcmc

    @property
    def facc(self):
        if not hasattr(self, '_facc'):
            if os.path.exists('%s.facc.pkl' % self.prefix):
                f = open('%s.facc.pkl' % self.prefix, 'rb')
                self._facc = []
                while True:
                    try:
                        self._facc.append(pickle.load(f))
                    except EOFError:
                        break
                f.close()
                self._facc = np.array(self._facc)
            else:
                self._facc = None
        
        return self._facc
                        
    def get_ax(self, ax=None, fig=1):
        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True
        
        return ax, gotax
            
    @property
    def timing(self):
        if not hasattr(self, '_timing'):
            self._timing = []
            
            i = 1
            fn = '%s.timing_%s.pkl' % (self.prefix, str(i).zfill(4))
            while os.path.exists(fn):
                f = open(fn, 'rb')
                while True:
                    try:
                        t, kw = pickle.load(f)
                        self._timing.append((t, kw))
                    except EOFError:
                        break
                        
                f.close()
                i += 1
                fn = '%s.timing_%s.pkl' % (self.prefix, str(i).zfill(4))  
                
                
        return self._timing
            
    def save_hdf5(self):
        if not have_h5py:
            return
        
        if rank > 0:
            return
            
        f = h5py.File('%s.blobs.hdf5' % self.prefix, 'w')
        f.create_dataset('blobs', data=self.blobs)
        f.create_dataset('mask', data=self._mask)
        f.close()
        
        print 'Saved %s.blobs.hdf5' % self.prefix
        
        # Save to disk!
        if not os.path.exists('%s.dblobs.hdf5') and have_h5py:
            f = h5py.File('%s.dblobs.hdf5' % self.prefix, 'w')
            f.create_dataset('mask', data=self._mask)
            f.create_dataset('derived_blobs', data=self._derived_blobs)
            f.create_dataset('derived_blob_names', data=self._derived_blob_names)
            f.close()
        
            if rank == 0:
                print 'Saved %s.dblobs.hdf5' % self.prefix

    @property
    def Nd(self):
        if not hasattr(self, '_Nd'):
            try:
                self._Nd = int(self.chain.shape[-1])       
            except TypeError:
                self._Nd = None
        
        return self._Nd

    @property
    def chain(self):
        # Read MCMC chain
        if not hasattr(self, '_chain'):
            have_chain_f = os.path.exists('%s.chain.pkl' % self.prefix)
            have_f = os.path.exists('%s.pkl' % self.prefix)
            
            if have_chain_f or have_f:
                if have_chain_f:
                    fn = '%s.chain.pkl' % self.prefix
                else:
                    fn = '%s.pkl' % self.prefix
                
                if rank == 0:
                    print "Loading %s..." % fn

                t1 = time.time()
                self._chain = read_pickled_chain(fn)
                t2 = time.time()

                if rank == 0:
                    print "Loaded %s in %.2g seconds.\n" % (fn, t2-t1)

                mask2d = np.array([self.mask] * self._chain.shape[1]).T
                self._chain = np.ma.array(self._chain, mask=mask2d)
                    
            else:
                self._chain = None            

        return self._chain        
    
    @property
    def logL(self):
        if not hasattr(self, '_logL'):            
            if os.path.exists('%s.logL.pkl' % self.prefix):
                self._logL = read_pickled_logL('%s.logL.pkl' % self.prefix)
                self._logL = np.ma.array(self._logL, mask=self.mask)
            else:
                self._logL = None
                
        return self._logL
    
    @logL.setter
    def logL(self, value):
        self._logL = value
        
    @property
    def L(self):
        if not hasattr(self, '_L'):
            self._L = np.exp(self.logL)
        
        return self._L    
        
    @property
    def betas(self):
        if not hasattr(self, '_betas'):
            if os.path.exists('%s.betas.pkl' % self.prefix):
                self._betas = read_pickled_logL('%s.betas.pkl' % self.prefix)
            else:
                self._betas = None
        
        return self._betas
                
    @property
    def fails(self):
        if not hasattr(self, '_fails'):
            if os.path.exists('%s.fails.pkl' % self.prefix):
                i = 0
                self._fails = []
                while os.path.exists('%s.fail_%s.pkl' % (self.prefix, str(i).zfill(3))):
                
                    data = read_pickled_dict('%s.fail_%s.pkl' % (prefix, str(i).zfill(3)))
                    self._fails.extend(data)                    
                    i += 1
                    
            else:
                self._fails = None
            
        return self._fails
        
    def get_walker(self, num):
        """
        Return chain elements corresponding to specific walker.
        
        Parameters
        ----------
        num : int
            ID # for walker of interest.
            
        Returns
        -------
        2-D array with shape (nsteps, nparameters).
        
        """
        
        sf = self.save_freq
        nw = self.nwalkers
        nchunks = int(self.steps / sf)
        schunk = nw * sf
        data = []
        for i in range(nchunks):
            chunk = self.chain[i*schunk + sf*num:i*schunk + sf*(num+1)]
            data.extend(chunk)
            
        return np.array(data)
                
    @property
    def Npops(self):
        if not hasattr(self, '_Npops') and self.base_kwargs is not None:
            self._Npops = count_populations(**self.base_kwargs)
        elif self.base_kwargs is None:
            self._Npops = 1
    
        return self._Npops
    
    def _fix_up(self):
        
        if not hasattr(self, 'blobs'):
            return
        
        if not hasattr(self, 'chain'):
            return
        
        if self.blobs.shape[0] == self.chain.shape[0]:
            return
            
        # Force them to be the same shape. The shapes might mismatch if
        # one processor fails to write to disk (or just hasn't quite yet),
        # or for more pathological reasons I haven't thought of yet.
                        
        if self.blobs.shape[0] > self.chain.shape[0]:
            tmp = self.blobs[0:self.chain.shape[0]]
            self.blobs = tmp
        else:
            tmp = self.chain[0:self.blobs.shape[0]]
            self.chain = tmp
    
    def _load(self, fn):
        if os.path.exists(fn):
            return read_pickle_file(fn)
    
    @property    
    def blob_redshifts_float(self):
        if not hasattr(self, '_blob_redshifts_float'):
            self._blob_redshifts_float = []
            for i, redshift in enumerate(self.blob_redshifts):
                if type(redshift) is str:
                    self._blob_redshifts_float.append(None)
                else:
                    self._blob_redshifts_float.append(round(redshift, 3))
            
        return self._blob_redshifts_float
    
    @property    
    def blob_redshifts_float(self):
        if not hasattr(self, '_blob_redshifts_float'):
            self._blob_redshifts_float = []
            for i, redshift in enumerate(self.blob_redshifts):
                if type(redshift) is str:
                    z = None
                else:
                    z = redshift
                    
                self._blob_redshifts_float.append(z)
            
        return self._blob_redshifts_float
    
    def SelectModels(self):
        """
        Draw a rectangle on supplied matplotlib.axes.Axes instance, return
        information about those models.
        """
                
        if not hasattr(self, '_ax'):
            raise AttributeError('No axis found.')
                
        self._op = self._ax.figure.canvas.mpl_connect('button_press_event', 
            self._on_press)
        self._or = self._ax.figure.canvas.mpl_connect('button_release_event', 
            self._on_release)
                            
    def _on_press(self, event):
        self.x0 = event.xdata
        self.y0 = event.ydata
        
    def _on_release(self, event):
        self.x1 = event.xdata
        self.y1 = event.ydata
        
        self._ax.figure.canvas.mpl_disconnect(self._op)
        self._ax.figure.canvas.mpl_disconnect(self._or)
        
        # Width and height of rectangle
        dx = abs(self.x1 - self.x0)
        dy = abs(self.y1 - self.y0)
        
        # Find lower left corner of rectangle
        lx = self.x0 if self.x0 < self.x1 else self.x1
        ly = self.y0 if self.y0 < self.y1 else self.y1
        
        # Lower-left
        ll = (lx, ly)
        
        # Upper right
        ur = (lx + dx, ly + dy)
    
        origin = (self.x0, self.y0)
        rect = Rectangle(ll, dx, dy, fc='none', ec='k')
        
        self._ax.add_patch(rect)
        self._ax.figure.canvas.draw()
        
        self.Slice((lx, lx+dx, ly, ly+dy), **self.plot_info)
                
    def Slice(self, constraints, pars, ivar=None, take_log=False, 
        un_log=False, multiplier=1.):
        """
        Return revised ("sliced") dataset given set of criteria.
    
        Parameters
        ----------
        constraints : list, tuple
            A rectangle (or line segment) bounding the region of interest. 
            For 2-D plane, supply (left, right, bottom, top), and then to
            `pars` supply list of datasets defining the plane. For 1-D, just
            supply (min, max).
        pars:
            Dictionary of constraints to use to calculate likelihood.
            Each entry should be a two-element list, with the first
            element being the redshift at which to apply the constraint,
            and second, a function for the posterior PDF for that quantity.s
    
        Examples
        --------
    
        Returns
        -------
        Object to be used to initialize a new ModelSet instance.
    
        """
        
        if len(constraints) == 4:
            Nd = 2
            x1, x2, y1, y2 = constraints
        else:
            Nd = 1
            x1, x2 = constraints
    
        # Figure out what these values translate to.
        data, is_log = self.ExtractData(pars, ivar, take_log, un_log, 
            multiplier)
                        
        # Figure out elements we want
        xok_ = np.logical_and(data[pars[0]] >= x1, data[pars[0]] <= x2)
        xok_MP = np.logical_or(np.abs(data[pars[0]] - x1) <= MP, 
            np.abs(data[pars[0]] - x2) <= MP)
        xok = np.logical_or(xok_, xok_MP)

        if Nd == 2:
            yok_ = np.logical_and(data[pars[1]] >= y1, data[pars[1]] <= y2)
            yok_MP = np.logical_or(np.abs(data[pars[1]] - y1) <= MP, 
                np.abs(data[pars[1]] - y2) <= MP)
            yok = np.logical_or(yok_, yok_MP)
            to_keep = np.logical_and(xok, yok)
        else:
            to_keep = xok

        mask = np.logical_not(to_keep)
        
        ##
        # CREATE NEW MODELSET INSTANCE
        ##
        model_set = ModelSet(self.prefix)
        
        # Set the mask! 
        model_set.mask = np.logical_or(mask, self.mask)
        
        i = 0
        while hasattr(self, 'slice_%i' % i):
            i += 1
    
        setattr(self, 'slice_%i' % i, model_set)
        
        print "Saved result to slice_%i attribute." % i
        
        return model_set
        
    @property
    def plot_info(self):
        if not hasattr(self, '_plot_info'):
            self._plot_info = None
    
        return self._plot_info
    
    @plot_info.setter
    def plot_info(self, value):
        self._plot_info = value

    def sort_by_Tmin(self):
        """
        If doing a multi-pop fit, re-assign population ID numbers in 
        order of increasing Tmin.
        
        Doesn't return anything. Replaces attribute 'chain' with new array.
        """

        # Determine number of populations
        tmp_pf = {key : None for key in self.parameters}
        Npops = count_populations(**tmp_pf)

        if Npops == 1:
            return
        
        # Check to see if Tmin is common among all populations or not    
    
    
        # Determine which indices correspond to Tmin, and population #
    
        i_Tmin = []
        
        # Determine which indices 
        pops = [[] for i in range(Npops)]
        for i, par in enumerate(self.parameters):

            # which pop?
            m = re.search(r"\{([0-9])\}", par)

            if m is None:
                continue

            num = int(m.group(1))
            prefix = par.split(m.group(0))[0]
            
            if prefix == 'Tmin':
                i_Tmin.append(i)

        self._unsorted_chain = self.chain.copy()

        # Otherwise, proceed to re-sort data
        tmp_chain = np.zeros_like(self.chain)
        for i in range(self.chain.shape[0]):

            # Pull out values of Tmin
            Tmin = [self.chain[i,j] for j in i_Tmin]
            
            # If ordering is OK, move on to next link in the chain
            if np.all(np.diff(Tmin) > 0):
                tmp_chain[i,:] = self.chain[i,:].copy()
                continue

            # Otherwise, we need to fix some stuff

            # Determine proper ordering of Tmin indices
            i_Tasc = np.argsort(Tmin)
            
            # Loop over populations, and correct parameter values
            tmp_pars = np.zeros(len(self.parameters))
            for k, par in enumerate(self.parameters):
                
                # which pop?
                m = re.search(r"\{([0-9])\}", par)

                if m is None:
                    tmp_pars.append()
                    continue

                pop_num = int(m.group(1))
                prefix = par.split(m.group(0))[0]
                
                new_pop_num = i_Tasc[pop_num]
                
                new_loc = self.parameters.index('%s{%i}' % (prefix, new_pop_num))
                
                tmp_pars[new_loc] = self.chain[i,k]

            tmp_chain[i,:] = tmp_pars.copy()
                        
        del self.chain
        self.chain = tmp_chain

    @property
    def cosm(self):
        if not hasattr(self, '_cosm'):
            self._cosm = Cosmology()
        
        return self._cosm

    @property
    def derived_blob_names(self):
        if hasattr(self, '_derived_blob_names'):
            return self._derived_blob_names
            
        dbs = self.derived_blobs
        
        return self._derived_blob_names
        
    def set_constraint(self, add_constraint=False, **constraints):
        """
        For ModelGrid calculations, the likelihood must be supplied 
        after the fact.

        Parameters
        ----------
        add_constraint: bool
            If True, operate with logical and when constructing likelihood.
            That is, this constraint will be applied in conjunction with
            previous constraints supplied.
        constraints : dict
            Constraints to use in calculating logL
            
        Example
        -------
        # Assume redshift of turning pt. D is 15 +/- 2 (1-sigma Gaussian)
        data = {'z': ['D', lambda x: np.exp(-(x - 15)**2 / 2. / 2.**2)]}
        self.set_constraint(**data)
            
        Returns
        -------
        Sets "logL" attribute, which is used by several routines.    
            
        """    

        if add_constraint and hasattr(self, 'logL'):
            pass
        else:    
            self.logL = np.zeros(self.chain.shape[0])

        if hasattr(self, '_weights'):
            del self._weights

        for i in range(self.chain.shape[0]):
            logL = 0.0
            
            if i >= self.blobs.shape[0]:
                break
            
            for element in constraints:

                z, func = constraints[element]
                
                try:
                    j = self.blob_redshifts.index(z)
                except ValueError:
                    ztmp = []
                    for redshift in self.blob_redshifts_float:
                        if redshift is None:
                            ztmp.append(None)
                        else:
                            ztmp.append(round(redshift, 1))    

                    j = ztmp.index(round(z, 1))
                
                if element in self.blob_names:
                    k = self.blob_names.index(element)
                    data = self.blobs[i,j,k]
                else:
                    k = self.derived_blob_names.index(element)
                    data = self.derived_blobs[i,j,k]                

                logL -= np.log(func(data))

            self.logL[i] -= logL

        mask = np.isnan(self.logL)

        self.logL[mask] = -np.inf

    def Scatter(self, pars, ivar=None, ax=None, fig=1, c=None,
        take_log=False, un_log=False, multiplier=1., use_colorbar=True, 
        **kwargs):
        """
        Plot samples as points in 2-d plane.
    
        Parameters
        ----------
        pars : list
            2-element list of parameter names. 
        ivar : float, list
            Independent variable(s) to be used for non-scalar blobs.
        z : str, float
            Redshift at which to plot x vs. y, if applicable.
        c : str
            Field for (optional) color axis.

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot instance.

        """
        
        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True

        # Make a new variable since pars might be self.parameters
        # (don't want to modify that)
        if c is not None:
            p = list(pars) + [c]
            if ivar is not None:
                if len(ivar) != 3:
                    iv = list(ivar) + [None]
                else:
                    iv = ivar
            else:
                iv = None
        else:
            p = pars
            iv = ivar
                    
        data, is_log = \
            self.ExtractData(p, iv, take_log, un_log, multiplier)

        xdata = data[p[0]]
        ydata = data[p[1]]
                
        if c is not None:
            cdata = data[p[2]].squeeze()
        else:
            cdata = None

        if hasattr(self, 'weights') and cdata is None:
            scat = ax.scatter(xdata, ydata, c=self.weights, **kwargs)
        elif cdata is not None:
            scat = ax.scatter(xdata, ydata, c=cdata, **kwargs)
        else:
            scat = ax.scatter(xdata, ydata, **kwargs)
                           
        if (cdata is not None) and use_colorbar:
            cb = self._cb = pl.colorbar(scat)
        else:
            cb = None
        
        self._scat = scat
            
        self.plot_info = {'pars': pars, 'ivar': ivar,
            'take_log': take_log, 'un_log':un_log, 'multiplier':multiplier}
            
        # Make labels
        self.set_axis_labels(ax, p, is_log, take_log, un_log, cb)
        
        pl.draw()        
        self._ax = ax
        return ax
    
    def BoundingPolygon(self, pars, ivar=None, ax=None, fig=1,
        take_log=False, un_log=False, multiplier=1., 
        **kwargs):
        """
        Basically a scatterplot but instead of plotting individual points,
        we draw lines bounding the locations of all those points.
        
        Parameters
        ----------
        pars : list, tuple
            List of parameters that defines 2-D plane.
            
        """
        
        assert have_shapely, "Need shapely installed for this to work."
        assert have_descartes, "Need descartes installed for this to work."
        
        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True

        data, is_log = \
            self.ExtractData(pars, ivar, take_log, un_log, multiplier)
        
        xdata = data[pars[0]].compressed()
        ydata = data[pars[1]].compressed()
        
        # Organize into (x, y) pairs
        points = zip(xdata, ydata)
                
        # Create polygon object
        point_collection = geometry.MultiPoint(list(points))
        polygon = point_collection.convex_hull
                
        # Plot a Polygon using descartes
        x_min, y_min, x_max, y_max = polygon.bounds
        patch = PolygonPatch(polygon, **kwargs)
        ax.add_patch(patch)
        
        pl.draw()
        
        return ax
        
    def get_par_prefix(self, par):
        m = re.search(r"\{([0-9])\}", par)

        if m is None:
            return par

        # Population ID number
        num = int(m.group(1))

        # Pop ID including curly braces
        prefix = par.split(m.group(0))[0]
    
        return prefix
    
    @property
    def weights(self):        
        if (not self.is_mcmc) and hasattr(self, 'logL') \
            and (not hasattr(self, '_weights')):
            self._weights = np.exp(self.logL)

        return self._weights

    def get_levels(self, L, nu=[0.95, 0.68]):
        """
        Return levels corresponding to input nu-values, and assign
        colors to each element of the likelihood.
        """
    
        nu, levels = _error_2D_crude(L, nu=nu)
                                                                      
        return nu, levels
    
    def get_1d_error(self, par, ivar=None, nu=0.68, take_log=False,
        limit=None, un_log=False, multiplier=1., peak='median'):
        """
        Compute 1-D error bar for input parameter.
        
        Parameters
        ----------
        par : str
            Name of parameter. 
        nu : float
            Percent likelihood enclosed by this 1-D error
        peak : str
            Determines whether the 'best' value is the median, mode, or
            maximum likelihood point.
            
        Returns
        -------
        if peak is None:
            Returns x-values corresponding to desired quartile range, i.e.,
            not really an error-bar.
        else:
            tuple: (maximum likelihood value, negative error, positive error).
        """

        #pars, take_log, multiplier, un_log, ivar = \
        #    self._listify_common_inputs([par], take_log, multiplier, un_log, ivar)

        to_hist, is_log = self.ExtractData(par, ivar=ivar, take_log=take_log, 
            multiplier=multiplier)

        # Need to weight results of non-MCMC runs explicitly
        if not hasattr(self, 'weights'):
            weights = None
        else:
            weights = self.weights

        # Apply mask to weights
        if weights is not None and to_hist[par].shape != weights.shape:
            weights = weights[np.logical_not(mask)]

        if to_hist[par] == []:
            print "WARNING: error w/ %s" % par
            print "@ z=" % z
            return

        if peak == 'median':
            N = len(self.logL)
            psorted = np.sort(to_hist[par])
            mu = psorted[int(N / 2.)]
        elif peak == 'mode':
            mu = to_hist[par][np.argmax(self.logL)]
        else:
            mu = None
            
        if hasattr(to_hist[par], 'compressed'):
            to_hist[par] = to_hist[par].compressed()
        
        q1 = 0.5 * 100 * (1. - nu)    
        q2 = 100 * nu + q1
        lo, hi = np.percentile(to_hist[par], (q1, q2))
        
        if mu is not None:
            sigma = (mu - lo, hi - mu)
        else:
            sigma = (lo, hi)

        return mu, np.array(sigma)
        
    def _get_1d_kwargs(self, **kw):
        
        for key in ['labels', 'colors', 'linestyles']:
        
            if key in kw:
                kw.pop(key)

        return kw
        
    def _slice_by_nu(self, pars, z=None, take_log=False, bins=20, like=0.68,
        **constraints):
        """
        Return points in dataset satisfying given confidence contour.
        """
        
        binvec, to_hist, is_log = self._prep_plot(pars, z=z, bins=bins, 
            take_log=take_log)
        
        if not self.is_mcmc:
            self.set_constraint(**constraints)
        
        if not hasattr(self, 'weights'):
            weights = None
        else:
            weights = self.weights
        
        hist, xedges, yedges = \
            np.histogram2d(to_hist[0], to_hist[1], bins=binvec, 
            weights=weights)

        # Recover bin centers
        bc = []
        for i, edges in enumerate([xedges, yedges]):
            bc.append(rebin(edges))
                
        # Determine mapping between likelihood and confidence contours

        # Get likelihood contours (relative to peak) that enclose
        # nu-% of the area
        like, levels = self.get_levels(hist, nu=like)
        
        # Grab data within this contour.
        to_keep = np.zeros(to_hist[0].size)
        
        for i in range(hist.shape[0]):
            for j in range(hist.shape[1]):
                if hist[i,j] < levels[0]:
                    continue
                    
                # This point is good
                iok = np.logical_and(xedges[i] <= to_hist[0], 
                    to_hist[0] <= xedges[i+1])
                jok = np.logical_and(yedges[j] <= to_hist[1], 
                    to_hist[1] <= yedges[j+1])
                                    
                ok = iok * jok                    
                                                            
                to_keep[ok == 1] = 1
                
        model_set = ModelSubSet()
        model_set.chain = np.array(self.chain[to_keep == 1])
        model_set.base_kwargs = self.base_kwargs.copy()
        model_set.fails = []
        model_set.blobs = np.array(self.blobs[to_keep == 1,:,:])
        model_set.blob_names = self.blob_names
        model_set.blob_redshifts = self.blob_redshifts
        model_set.is_log = self.is_log
        model_set.parameters = self.parameters
        
        model_set.is_mcmc = self.is_mcmc
        
        if self.is_mcmc:
            model_set.logL = logL[to_keep == 1]
        else:
            model_set.axes = self.axes
        
        return ModelSet(model_set)
        
    def _prep_plot(self, pars, z=None, take_log=False, multiplier=1.,
        skip=0, skim=1, bins=20):
        """
        Given parameter names as strings, return data, bins, and log info.
        
        Returns
        -------
        Tuple : (bin vectors, data to histogram, is_log)
        """
        
        if type(pars) not in [list, tuple]:
            pars = [pars]
        if type(take_log) == bool:
            take_log = [take_log] * len(pars)
        if type(multiplier) in [int, float]:
            multiplier = [multiplier] * len(pars)    
        
        if type(z) is list:
            if len(z) != len(pars):
                raise ValueError('Length of z must be = length of pars!')
        else:
            z = [z] * len(pars)
        
        binvec = []
        to_hist = []
        is_log = []
        for k, par in enumerate(pars):

            if par in self.parameters:        
                j = self.parameters.index(par)
                is_log.append(self.is_log[j])
                
                val = self.chain[skip:,j].ravel()[::skim]
                                
                if self.is_log[j]:
                    val += np.log10(multiplier[k])
                else:
                    val *= multiplier[k]
                                
                if take_log[k] and not self.is_log[j]:
                    to_hist.append(np.log10(val))
                else:
                    to_hist.append(val)
                
            elif (par in self.blob_names) or (par in self.derived_blob_names):
                
                if z is None:
                    raise ValueError('Must supply redshift!')
                    
                i = self.blob_redshifts.index(z[k])
                
                if par in self.blob_names:
                    j = list(self.blob_names).index(par)
                else:
                    j = list(self.derived_blob_names).index(par)
                
                is_log.append(False)
                
                if par in self.blob_names:
                    val = self.blobs[skip:,i,j][::skim]
                else:
                    val = self.derived_blobs[skip:,i,j][::skim]
                
                if take_log[k]:
                    val += np.log10(multiplier[k])
                else:
                    val *= multiplier[k]
                
                if take_log[k]:
                    to_hist.append(np.log10(val))
                else:
                    to_hist.append(val)

            else:
                raise ValueError('Unrecognized parameter %s' % str(par))

            if not bins:
                continue
            
            # Set bins
            if self.is_mcmc or (par not in self.parameters):
                if type(bins) == int:
                    valc = to_hist[k]
                    binvec.append(np.linspace(valc.min(), valc.max(), bins))
                elif type(bins[k]) == int:
                    valc = to_hist[k]
                    binvec.append(np.linspace(valc.min(), valc.max(), bins[k]))
                else:
                    if take_log[k]:
                        binvec.append(np.log10(bins[k]))
                    else:
                        binvec.append(bins[k])
            else:
                if take_log[k]:
                    binvec.append(np.log10(self.axes[par]))
                else:
                    binvec.append(self.axes[par])
        
        return pars, to_hist, is_log, binvec
      
    def ExtractData(self, pars, ivar=None, take_log=False, un_log=False, 
        multiplier=1., remove_nas=False):
        """
        Extract data for subsequent analysis.
        
        This means a few things:
         (1) Go retrieve data from native format without having to worry about
          all the indexing yourself.
         (2) [optionally] take the logarithm.
         (3) [optionally] apply multiplicative factors.
         (4) Create a mask that excludes all nans / infs.
         
        Parameters
        ----------
        pars : list
            List of quantities to return. These can be parameters or the names
            of meta-data blobs.
        ivars : list
            List of independent variables at which to compute values of pars.
        take_log single bool or list of bools determining whether data should
                 be presented after its log is taken
        un_log single bool or list of bools determining whether data should be
               presented after its log is untaken (i.e. it is exponentiated)
        multiplier list of numbers to multiply the parameters by before they
                   are presented
        remove_nas bool determining whether rows with nan's or inf's should be
                   removed or not. This must be set to True when the user
                   is using numpy newer than version 1.9.x if the user wants
                   to histogram the data because numpy gave up support for
                   masked arrays in histograms.
        
        Returns
        -------
        Tuple with two entries:
         (i) Dictionary containing 1-D arrays of samples for each quantity.
         (ii) Dictionary telling us which of the datasets are actually the
          log10 values of the associated parameters.
         
        """

        pars, take_log, multiplier, un_log, ivar = \
            self._listify_common_inputs(pars, take_log, multiplier, un_log, 
            ivar)

        data = {}
        is_log = {}
        for k, par in enumerate(pars):
                    
            # If one of our free parameters, things are easy.
            if par in self.parameters:
                
                j = self.parameters.index(par)

                if self.is_log[j] and un_log[k]:
                    val = 10**self.chain[:,j].copy()
                else:
                    val = self.chain[:,j].copy()
        
                if self.is_log[j] and (not un_log[k]):
                    val += np.log10(multiplier[k])
                else:
                    val *= multiplier[k]
                    
                # Take log, unless the parameter is already in log10
                if take_log[k] and (not self.is_log[j]):
                    val = np.log10(val)
                                        
            # Blobs are a little harder, might need new mask later.
            elif par in self.all_blob_names:
                
                i, j, nd, dims = self.blob_info(par)

                if nd == 0:
                    val = self.get_blob(par, ivar=None).copy()
                else:
                    val = self.get_blob(par, ivar=ivar[k]).copy()

                # Blobs are never stored as log10 of their true values
                val *= multiplier[k]
                
            # Only derived blobs in this else block, yes?                        
            else:
                
                cand = glob.glob('%s*.%s.pkl' % (self.prefix, par))
                
                if len(cand) == 1:
                    f = open(cand[0], 'rb')     
                    dat = pickle.load(f)
                    f.close()
                    
                    # What follows is real cludgey...sorry, future Jordan
                    nd = len(dat.shape) #- 1
                    assert nd == 1, "Help!"
                    
                    if ivar[k] is not None:
                        for hh in range(len(self.blob_dims)):
                            if len(self.blob_dims[hh]) != 1:
                                continue
                            if len(dat[0]) == self.blob_dims[hh][0]:
                                loc = np.argmin(np.abs(self.blob_ivars[hh][0] - ivar[k]))
                                break
                            
                        val = dat[:,loc]
                    else:
                        val = dat

            # must handle log-ifying blobs separately
            if par not in self.parameters:
                if take_log[k]:
                    val = np.log10(val)
                                   
            ##
            # OK, at this stage, 'val' is just an array. If it corresponds to
            # a parameter, it's 1-D, if a blob, it's dimensionality could
            # be different. So, we have to be a little careful with the mask.
            ##                   
              
            if not np.all(np.array(val.shape) == np.array(self.mask.shape)):
                
                # If no masked elements, don't worry any more. Just set -> 0.
                if not np.any(self.mask == 1):
                    mask = 0
                # Otherwise, we might need to reshape the mask.
                # If, for example, certain links in the MCMC chain are masked,
                # we need to make sure that every blob element corresponding
                # to those links are masked.
                else:
                    mask = np.zeros_like(val)
                    for j, element in enumerate(self.mask):
                        if element == 1:
                            mask[j].fill(1)
            else:
                mask = self.mask

            if self.is_mcmc:
                data[par] = np.ma.array(val, mask=mask)
            else:
                try:
                    data[par] = np.ma.array(val, mask=mask)
                except np.ma.MaskError:
                    print "MaskError encountered. Assuming mask=0."
                        
                    data[par] = np.ma.array(val, mask=0)

        if remove_nas:
            to_remove = []
            length = len(data[data.keys()[0]])
            for ilink in range(length):
                for par in data:
                    elem = data[par][ilink]
                    if type(elem) is np.ma.core.MaskedConstant:
                        to_remove.append(ilink)
                        break
                    elif type(elem) in numerical_types:
                        if np.isinf(elem) or np.isnan(elem):
                            to_remove.append(ilink)
                            break
                    else: # elem is array (because par is a non-0d blob)
                        is_inf_or_nan = (np.isinf(elem) | np.isnan(elem))
                        if hasattr(elem, 'mask'): # ignore rows affected by mask
                            is_inf_or_nan = (is_inf_or_nan | elem.mask)
                        if not np.all(~is_inf_or_nan):
                            to_remove.append(ilink)
                            break
            for par in data:
                data[par] = np.delete(data[par], to_remove, axis=0)
            print ("%i of %i " % (len(to_remove),length,)) +\
                  "chain elements ignored because of chain links with " +\
                  "inf's/nan's."

        return data, is_log

    def _set_bins(self, pars, to_hist, take_log=False, bins=20):
        """
        Create a vector of bins to be used when plotting PDFs.
        """
        
        if type(to_hist) is dict:
            binvec = {}
        else:
            binvec = []
            
        for k, par in enumerate(pars):
            
            if type(to_hist) is dict:
                tohist = to_hist[par]
            else:
                tohist = to_hist[k]
        
            if self.is_mcmc or (par not in self.parameters) or \
                not hasattr(self, 'axes'):
                if type(bins) == int:
                    valc = tohist
                    bvp = np.linspace(valc.min(), valc.max(), bins)
                elif type(bins[k]) == int:
                    valc = tohist
                    bvp = np.linspace(valc.min(), valc.max(), bins[k])
                else:
                    bvp = bins[k]
                    #if take_log[k]:
                    #    binvec.append(np.log10(bins[k]))
                    #else:
                    #    binvec.append(bins[k])
            else:
                if take_log[k]:
                    bvp = np.log10(self.axes[par])
                else:
                    bvp = self.axes[par]
        
            if type(to_hist) is dict:
                binvec[par] = bvp
            else:
                binvec.append(bvp)
        
        return binvec
        
    def _set_inputs(self, pars, inputs, is_log, take_log, multiplier):
        """
        Figure out input values for x and y parameters for each panel.
        
        Returns
        -------
        Dictionary, elements sorted by 
        """

        if inputs is None:
            return None
        
        if type(inputs) is list:
            if inputs == []:
                return None
        
        if type(inputs) is dict:
            if not inputs:
                return None
        else:
            inputs = list(inputs)
                        
        if type(is_log) is dict:
            if is_log != {}:
                tmp = [is_log[par] for par in pars]    
                is_log = tmp
            else:
                is_log = [False] * len(pars)
        
        if type(multiplier) in [int, float]:
            multiplier = [multiplier] * len(pars)    
            
        if len(np.unique(pars)) < len(pars):
            input_output = []
        else:
            input_output = {}
        
        Nd = len(pars)
                                
        for i, par in enumerate(pars):
            if type(inputs) is list:
                val = inputs[i]
            elif par in inputs:
                val = inputs[par]
            else:
                dq = DQ(data=inputs)
                try:
                    val = dq[par]
                except:
                    val = None
                    
            # Take log [optional]    
            if val is None:
                vin = None
            elif is_log[i] or take_log[i]:
                vin = np.log10(val * multiplier[i])                            
            else:
                vin = val * multiplier[i]    
                
            if type(input_output) is dict:
                input_output[par] = vin
            else:
                input_output.append(vin)
            
        return input_output
        
    def _listify_common_inputs(self, pars, take_log, multiplier, un_log, 
        ivar=None):
        """
        Make everything lists.
        """
        
        if type(pars) not in [list, tuple]:
            pars = [pars]
        if type(take_log) == bool:
            take_log = [take_log] * len(pars)
        if type(un_log) == bool:
            un_log = [un_log] * len(pars)    
        if type(multiplier) in [int, float]:
            multiplier = [multiplier] * len(pars)

        if ivar is not None:
            if type(ivar) is list:
                if len(pars) == 1:
                    i, j, nd, dims = self.blob_info(pars[0])
                    
                    if nd == 2:
                        ivar = list(np.atleast_2d(ivar))
                
                assert len(ivar) == len(pars)
            else:
                if len(pars) == 1:
                    ivar = [ivar]
                else:
                    raise ValueError('ivar must be same length as pars')    
                
        else:
            ivar = [None] * len(pars)
            
        return pars, take_log, multiplier, un_log, ivar

    def PosteriorCDF(self, pars, bins=500, **kwargs):
        return self.PosteriorPDF(pars, bins=bins, cdf=True, **kwargs)
               
    def PosteriorPDF(self, pars, to_hist=None, is_log=None, ivar=None, 
        ax=None, fig=1, 
        multiplier=1., like=[0.95, 0.68], cdf=False,
        color_by_like=False, filled=True, take_log=False, un_log=False,
        bins=20, skip=0, skim=1, 
        contour_method='raw', excluded=False, stop=None, **kwargs):
        """
        Compute posterior PDF for supplied parameters. 
    
        If len(pars) == 2, plot 2-D posterior PDFs. If len(pars) == 1, plot
        1-D marginalized PDF.
    
        Parameters
        ----------
        pars : str, list
            Name of parameter or list of parameters to analyze.
        ivar : float
            Redshift, if any element of pars is a "blob" quantity.
        plot : bool
            Plot PDF?
        nu : float, list
            If plot == False, return the nu-sigma error-bar.
            If color_by_like == True, list of confidence contours to plot.
        color_by_like : bool
            If True, color points based on what confidence contour they lie
            within.
        multiplier : list
            Two-element list of multiplicative factors to apply to elements of
            pars.
        take_log : list
            Two-element list saying whether to histogram the base-10 log of
            each parameter or not.
        skip : int
            Number of steps at beginning of chain to exclude. This is a nice
            way of doing a burn-in after the fact.
        skim : int
            Only take every skim'th step from the chain.
        excluded : bool
            If True, and filled == True, fill the area *beyond* the given contour with
            cross-hatching, rather than the area interior to it.

        Returns
        -------
        Either a matplotlib.Axes.axis object or a nu-sigma error-bar, 
        depending on whether we're doing a 2-D posterior PDF (former) or
        1-D marginalized posterior PDF (latter).
    
        """

        cs = None
        
        kw = kwargs

        if 'labels' in kw:
            labels = kwargs['labels']
        else:
            labels = {}
            
        # Only make a new plot window if there isn't already one
        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True

        # Grab all the data we need
        if (to_hist is None) or (is_log is None):
            to_hist, is_log = self.ExtractData(pars, ivar=ivar, 
                take_log=take_log, un_log=un_log, multiplier=multiplier)

        pars, take_log, multiplier, un_log, ivar = \
            self._listify_common_inputs(pars, take_log, multiplier, un_log, 
            ivar)
            
        # Modify bins to account for log-taking, multipliers, etc.
        binvec = self._set_bins(pars, to_hist, take_log, bins)

        # We might supply weights by-hand for ModelGrid calculations
        if not hasattr(self, 'weights'):
            weights = None
        else:
            weights = self.weights

        ##
        ### Histogramming and plotting starts here
        ##

        if stop is not None:
            stop = -int(stop)
                    
        # Marginalized 1-D PDFs 
        if len(pars) == 1:
                        
            if type(to_hist) is dict:
                tohist = to_hist[pars[0]][skip:stop]
                b = binvec[pars[0]]
            elif type(to_hist) is list:
                tohist = to_hist[0][skip:stop]
                b = binvec[0]
            else:
                tohist = to_hist[skip:stop]
                b = bins
            
            hist, bin_edges = \
                np.histogram(tohist, density=True, bins=b, weights=weights)

            bc = rebin(bin_edges)
            
            # Take CDF
            if cdf:
                hist = np.cumsum(hist)
                        
            tmp = self._get_1d_kwargs(**kw)
            
            ax.plot(bc, hist / hist.max(), drawstyle='steps-mid', **tmp)
            
            ax.set_ylim(0, 1.05)
            
        # Marginalized 2-D PDFs
        else:
            
            #if to_hist[0].size != to_hist[1].size:
            #    print 'Looks like calculation was terminated after chain',
            #    print 'was written to disk, but before blobs. How unlucky!'
            #    print 'Applying cludge to ensure shape match...'
            #    
            #    if to_hist[0].size > to_hist[1].size:
            #        to_hist[0] = to_hist[0][0:to_hist[1].size]
            #    else:
            #        to_hist[1] = to_hist[1][0:to_hist[0].size]

            if type(to_hist) is dict:
                tohist1 = to_hist[pars[0]][skip:stop]
                tohist2 = to_hist[pars[1]][skip:stop]
                b = [binvec[pars[0]], binvec[pars[1]]]
            else:
                tohist1 = to_hist[0][skip:stop]
                tohist2 = to_hist[1][skip:stop]
                b = [binvec[0], binvec[1]]

            # Compute 2-D histogram
            hist, xedges, yedges = \
                np.histogram2d(tohist1, tohist2, bins=b, weights=weights)

            hist = hist.T

            # Recover bin centers
            bc = []
            for i, edges in enumerate([xedges, yedges]):
                bc.append(rebin(edges))

            # Determine mapping between likelihood and confidence contours
            if color_by_like:

                # Get likelihood contours (relative to peak) that enclose
                # nu-% of the area

                if contour_method == 'raw':
                    nu, levels = error_2D(None, None, hist, None, nu=like, 
                        method='raw')
                else:
                    nu, levels = error_2D(to_hist[0], to_hist[1], self.L / self.L.max(), 
                        bins=[binvec[0], binvec[1]], nu=nu, method=contour_method)
        
                if filled:
                    if excluded and len(nu) == 1:
                        # Fill the entire window with cross-hatching
                        x1, x2 = ax.get_xlim()
                        y1, y2 = ax.get_ylim()

                        x_polygon = [x1, x2, x2, x1]
                        y_polygon = [y1, y1, y2, y2]

                        ax.fill(x_polygon, y_polygon, color="none", hatch='X', 
                            edgecolor=kwargs['color'])
                            
                        # Now, fill the enclosed area with white
                        ax.contourf(bc[0], bc[1], hist / hist.max(), 
                            levels, color='w', colors='w', zorder=2)
                        # Draw an outline too   
                        ax.contour(bc[0], bc[1], hist / hist.max(), 
                            levels, colors=kwargs['color'], linewidths=1, 
                            zorder=2)
                        
                    else:
                        ax.contourf(bc[0], bc[1], hist / hist.max(), 
                            levels, zorder=3, **kwargs)
                    
                else:
                    ax.contour(bc[0], bc[1], hist / hist.max(),
                        levels, zorder=4, **kwargs)
                
            else:
                if filled:
                    cs = ax.contourf(bc[0], bc[1], hist / hist.max(), 
                        zorder=3, **kw)
                else:
                    cs = ax.contour(bc[0], bc[1], hist / hist.max(), 
                        zorder=4, **kw)

            # Force linear
            if not gotax:
                ax.set_xscale('linear')
                ax.set_yscale('linear')
            
        # Add nice labels (or try to)
        self.set_axis_labels(ax, pars, is_log, take_log, un_log, None, labels)

        # Rotate ticks?
        for tick in ax.get_xticklabels():
            tick.set_rotation(45.)
        for tick in ax.get_yticklabels():
            tick.set_rotation(45.)
        
        pl.draw()
        
        return ax
              
    def Contour(self, pars, c, levels, leveltol=1e-6, ivar=None, take_log=False,
        un_log=False, multiplier=1., ax=None, fig=1, filled=False, **kwargs):         
        """
        Draw contours that are NOT associated with confidence levels.
        
        ..note:: To draw many contours in same plane, just call this 
            function repeatedly.
        
        Parameters
        ----------
        pars : list 
            List of parameters defining the plane on which to draw contours.
        c : str
            Name of parameter or blob that we're to draw contours of.
        levels : list
            [Optional] list of levels for 
                        
        """
        
        # Only make a new plot window if there isn't already one
        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True

        p = list(pars) + [c]

        # Grab all the data we need
        data, is_log = self.ExtractData(p, ivar=ivar, 
            take_log=take_log, un_log=un_log, multiplier=multiplier)

        xdata = data[p[0]]
        ydata = data[p[1]]    
        zdata = data[p[2]]
        
        for level in levels:
            # Find indices of appropriate elements
            cond = np.abs(zdata - level) < leveltol
            elements = np.argwhere(cond).squeeze()
            
            order = np.argsort(xdata[elements])
            ax.plot(xdata[elements][order], ydata[elements][order], **kwargs)
            
        pl.draw()    
            
        return ax, xdata, ydata, zdata
              
    def ContourScatter(self, x, y, c, z=None, Nscat=1e4, take_log=False, 
        cmap='jet', alpha=1.0, bins=20, vmin=None, vmax=None, zbins=None, 
        labels=None, **kwargs):
        """
        Show contour plot in 2-D plane, and add colored points for third axis.
        
        Parameters
        ----------
        x : str
            Fields for the x-axis.
        y : str
            Fields for the y-axis.
        c : str
            Name of parameter to represent with colored points.
        z : int, float, str
            Redshift (if investigating blobs)
        Nscat : int
            Number of samples plot.
            
        Returns
        -------
        Three objects: the main Axis instance, the scatter plot instance,
        and the colorbar object.
        
        """

        if type(take_log) == bool:
            take_log = [take_log] * 3

        if labels is None:
            labels = default_labels
        else:
            labels_tmp = default_labels.copy()
            labels_tmp.update(labels)
            labels = labels_tmp

        if type(z) is not list:
            z = [z] * 3

        pars = [x, y]

        axes = []
        for i, par in enumerate(pars):
            if par in self.parameters:
                axes.append(self.chain[:,self.parameters.index(par)])
            elif par in self.blob_names:
                axes.append(self.blobs[:,self.blob_redshifts.index(z[i]),
                    self.blob_names.index(par)])
            elif par in self.derived_blob_names:
                axes.append(self.derived_blobs[:,self.blob_redshifts.index(z[i]),
                    self.derived_blob_names.index(par)])        

        for i in range(2):
            if take_log[i]:
                axes[i] = np.log10(axes[i])

        xax, yax = axes

        if c in self.parameters:        
            zax = self.chain[:,self.parameters.index(c)].ravel()
        elif c in self.blob_names:   
            zax = self.blobs[:,self.blob_redshifts.index(z[-1]),
                self.blob_names.index(c)]
        elif c in self.derived_blob_names:   
            zax = self.derived_blobs[:,self.blob_redshifts.index(z[-1]),
                self.derived_blob_names.index(c)]
                
        if zax.shape[0] != self.chain.shape[0]:
            if self.chain.shape[0] > zax.shape[0]:
                xax = xax[0:self.blobs.shape[0]]
                yax = yax[0:self.blobs.shape[0]]
                print 'Looks like calculation was terminated after chain',
                print 'was written to disk, but before blobs. How unlucky!'
                print 'Applying cludge to ensure shape match...'
            else:                
                raise ValueError('Shape mismatch between blobs and chain!')    
                
        if take_log[2]:
            zax = np.log10(zax)    
            
        z.pop(-1)
        ax = self.PosteriorPDF(pars, z=z, take_log=take_log, filled=False, 
            bins=bins, **kwargs)
        
        # Pick out Nscat random points to plot
        mask = np.zeros_like(xax, dtype=bool)
        rand = np.arange(len(xax))
        np.random.shuffle(rand)
        mask[rand < Nscat] = True
        
        if zbins is not None:
            cmap_obj = eval('mpl.colorbar.cm.%s' % cmap)
            #if take_log[2]:
            #    norm = mpl.colors.LogNorm(zbins, cmap_obj.N)
            #else:    
            if take_log[2]:
                norm = mpl.colors.BoundaryNorm(np.log10(zbins), cmap_obj.N)
            else:    
                norm = mpl.colors.BoundaryNorm(zbins, cmap_obj.N)
        else:
            norm = None
        
        scat = ax.scatter(xax[mask], yax[mask], c=zax[mask], cmap=cmap,
            zorder=1, edgecolors='none', alpha=alpha, vmin=vmin, vmax=vmax,
            norm=norm)
        cb = pl.colorbar(scat)

        cb.set_alpha(1)
        cb.draw_all()

        if c in labels:
            cblab = labels[c]
        elif '{' in c:
            cblab = labels[c[0:c.find('{')]]
        else:
            cblab = c 
            
        if take_log[2]:
            cb.set_label(logify_str(cblab))
        else:
            cb.set_label(cblab)    
            
        cb.update_ticks()
            
        pl.draw()
        
        return ax, scat, cb    
        
    def ExtractPanel(self, panel, mp, ax=None, fig=99):
        """
        Save panel of a triangle plot as separate file.
        
        panel : int, str
            Integer or letter corresponding to plot panel you want.
        mp : MultiPlot instance
            Object representation of the triangle plot
        fig : int
            Figure number.
        
        """    
        
        letters = list(string.ascii_lowercase)
        letters.extend([let*2 for let in list(string.ascii_lowercase)])
        
        
        if type(panel) is str:
            panel = letters.index(panel)
        
        info = self.plot_info[panel]
        kw = self.plot_info['kwargs']
        
        ax = self.PosteriorPDF(info['axes'], z=info['z'], bins=info['bins'],
            multiplier=info['multiplier'], take_log=info['take_log'],
            fig=fig, ax=ax, **kw)
        
        ax.set_xticks(mp.grid[panel].get_xticks())
        ax.set_yticks(mp.grid[panel].get_yticks())
        
        xt = []
        for i, x in enumerate(mp.grid[panel].get_xticklabels()):
            xt.append(x.get_text())
        
        ax.set_xticklabels(xt, rotation=45.)
        
        yt = []
        for i, x in enumerate(mp.grid[panel].get_yticklabels()):
            yt.append(x.get_text())
            
        ax.set_yticklabels(yt, rotation=rotate_y)
        
        ax.set_xlim(mp.grid[panel].get_xlim())
        ax.set_ylim(mp.grid[panel].get_ylim())
        
        pl.draw()
        
        return ax
                
    def TrianglePlot(self, pars=None, ivar=None, take_log=False, un_log=False, 
        multiplier=1, fig=1, mp=None, inputs={}, tighten_up=0.0, ticks=5, 
        bins=20, skip=0, scatter=False,
        skim=1, oned=True, twod=True, filled=True, show_errors=False, 
        label_panels='upper right', 
        fix=True, skip_panels=[], stop=None, mp_kwargs={},
        **kwargs):
        """
        Make an NxN panel plot showing 1-D and 2-D posterior PDFs.

        Parameters
        ----------
        pars : list
            Parameters to include in triangle plot.
            1-D PDFs along diagonal will follow provided order of parameters
            from left to right. This list can contain the names of parameters,
            so long as the file prefix.pinfo.pkl exists, otherwise it should
            be the indices where the desired parameters live in the second
            dimension of the MCMC chain.

            NOTE: These can alternatively be the names of arbitrary meta-data
            blobs.

            If None, this will plot *all* parameters, so be careful!
        fig : int
            ID number for plot window.
        bins : int, np.ndarray
            Number of bins in each dimension. Or, array of bins to use
            for each parameter. If the latter, the bins should be in the 
            *final* units of the quantities of interest. For example, if
            you apply a multiplier or take_log, the bins should be in the
            native units times the multiplier or in the log10 of the native
            units (or both).
        ivar : int, float, str, list
            If plotting arbitrary meta-data blobs, must choose a redshift.
            Can be 'B', 'C', or 'D' to extract blobs at 21-cm turning points,
            or simply a number. If it's a list, it must have the same
            length as pars. This is how one can make a triangle plot 
            comparing the same quantities at different redshifts.
        input : dict
            Dictionary of parameter:value pairs representing the input
            values for all model parameters being fit. If supplied, lines
            will be drawn on each panel denoting these values.
        skip : int
            Number of steps at beginning of chain to exclude.
        stop: int
            Number of steps to exclude from the end of the chain.
        skim : int
            Only take every skim'th step from the chain.
        oned : bool    
            Include the 1-D marginalized PDFs?
        filled : bool
            Use filled contours? If False, will use open contours instead.
        color_by_like : bool
            If True, set contour levels by confidence regions enclosing nu-%
            of the likelihood. Set parameter `nu` to modify these levels.
        like : list
            List of levels, default is 1,2, and 3 sigma contours (i.e., 
            like=[0.68, 0.95])
        skip_panels : list
            List of panel numbers to skip over.
        mp_kwargs : dict 
            panel_size : list, tuple (2 elements)
                Multiplicative factor in (x, y) to be applied to the default 
                window size as defined in your matplotlibrc file. 
            
        ..note:: If you set take_log = True AND supply bins by hand, use the
            log10 values of the bins you want.
        
            
        Returns
        -------
        ares.analysis.MultiPlot.MultiPanel instance. Also saves a bunch of 
        information to the `plot_info` attribute.
        
        """    
        
        # Grab data that will be histogrammed
        np_version = np.__version__.split('.')
        newer_than_one = (int(np_version[0]) > 1)
        newer_than_one_pt_nine =\
            ((int(np_version[0]) == 1) and (int(np_version[1])>9))
        remove_nas = (newer_than_one or newer_than_one_pt_nine)
        to_hist, is_log = self.ExtractData(pars, ivar=ivar, take_log=take_log,
            un_log=un_log, multiplier=multiplier, remove_nas=remove_nas)
            
        # Make sure all inputs are lists of the same length!
        pars, take_log, multiplier, un_log, ivar = \
            self._listify_common_inputs(pars, take_log, multiplier, un_log, 
            ivar)        
            
        # Modify bins to account for log-taking, multipliers, etc.
        binvec = self._set_bins(pars, to_hist, take_log, bins)      
                            
        if type(binvec) is not list:
            bins = [binvec[par] for par in pars]      
        else:
            bins = binvec    
                 
        # Can opt to exclude 1-D panels along diagonal                
        if oned:
            Nd = len(pars)
        else:
            Nd = len(pars) - 1
                           
        # Setup MultiPanel instance
        had_mp = True
        if mp is None:
            had_mp = False
            
            mp_kw = default_mp_kwargs.copy()
            mp_kw['dims'] = [Nd] * 2    
            mp_kw.update(mp_kwargs)
            if 'keep_diagonal' in mp_kwargs:
                oned = False
            
            mp = MultiPanel(fig=fig, **mp_kw)
        
        # Apply multipliers etc. to inputs
        inputs = self._set_inputs(pars, inputs, is_log, take_log, multiplier)
        
        # Save some plot info for [optional] later tinkering
        self.plot_info = {}
        self.plot_info['kwargs'] = kwargs
        
        # Loop over parameters
        # p1 is the y-value, p2 is the x-value
        for i, p1 in enumerate(pars[-1::-1]):
            for j, p2 in enumerate(pars):

                # Row number is i
                # Column number is self.Nd-j-1

                if mp.diagonal == 'upper':
                    k = mp.axis_number(mp.N - i, mp.N - j)
                else:    
                    k = mp.axis_number(i, j)

                if k is None:
                    continue
                    
                if k in skip_panels:
                    continue

                if mp.grid[k] is None:
                    continue

                col, row = mp.axis_position(k)   
                
                # Read-in inputs values
                if inputs is not None:
                    if type(inputs) is dict:
                        xin = inputs[p2]
                        yin = inputs[p1]
                    else:
                        xin = inputs[j]
                        yin = inputs[-1::-1][i]
                else:
                    xin = yin = None
                    
                # 1-D PDFs on the diagonal    
                if k in mp.diag and oned:

                    # Grab array to be histogrammed
                    try:
                        tohist = [to_hist[j]]
                    except KeyError:
                        tohist = [to_hist[p2]]

                    # Plot the PDF
                    ax = self.PosteriorPDF(p1, ax=mp.grid[k], 
                        to_hist=tohist, is_log=is_log, 
                        take_log=take_log[-1::-1][i], ivar=ivar[-1::-1][i],
                        un_log=un_log[-1::-1][i], 
                        multiplier=[multiplier[-1::-1][i]], 
                        bins=[bins[-1::-1][i]], 
                        skip=skip, skim=skim, stop=stop, **kwargs)

                    # Stick this stuff in fix_ticks?
                    if col != 0:
                        mp.grid[k].set_ylabel('')
                    if row != 0:
                        mp.grid[k].set_xlabel('')

                    if show_errors:
                        mu, err = self.get_1d_error(p1)
                                                 
                        mp.grid[k].set_title(err_str(p1, mu, err, 
                            self.is_log[i], labels), va='bottom', fontsize=18) 
                     
                    self.plot_info[k] = {}
                    self.plot_info[k]['axes'] = [p1]
                    self.plot_info[k]['data'] = tohist
                    self.plot_info[k]['ivar'] = ivar[-1::-1][i]
                    self.plot_info[k]['bins'] = [bins[-1::-1][i]]
                    self.plot_info[k]['multplier'] = [multiplier[-1::-1][i]]
                    self.plot_info[k]['take_log'] = take_log[-1::-1][i]
                                          
                    if not inputs:
                        continue
                        
                    self.plot_info[k]['input'] = xin
                        
                    if xin is not None:
                        mp.grid[k].plot([xin]*2, [0, 1.05], 
                            color='k', ls=':', lw=2, zorder=20)
                            
                    continue

                if ivar is not None:
                    iv = [ivar[j], ivar[-1::-1][i]]
                else:
                    iv = None

                # If not oned, may end up with some x vs. x plots if we're not careful
                if p1 == p2 and (iv[0] == iv[1]):
                    continue
                    
                try:
                    tohist = [to_hist[j], to_hist[-1::-1][i]]
                except KeyError:
                    tohist = [to_hist[p2], to_hist[p1]]
                                                    
                # 2-D PDFs elsewhere
                if scatter:
                    ax = self.Scatter([p2, p1], ax=mp.grid[k], 
                        to_hist=tohist, is_log=is_log, z=red, 
                        take_log=[take_log[j], take_log[-1::-1][i]],
                        multiplier=[multiplier[j], multiplier[-1::-1][i]], 
                        bins=[bins[j], bins[-1::-1][i]], filled=filled, 
                        skip=skip, stop=stop, **kwargs)
                else:
                    ax = self.PosteriorPDF([p2, p1], ax=mp.grid[k], 
                        to_hist=tohist, is_log=is_log, ivar=iv, 
                        take_log=[take_log[j], take_log[-1::-1][i]],
                        un_log=[un_log[j], un_log[-1::-1][i]],
                        multiplier=[multiplier[j], multiplier[-1::-1][i]], 
                        bins=[bins[j], bins[-1::-1][i]], filled=filled, 
                        skip=skip, stop=stop, **kwargs)

                if row != 0:
                    mp.grid[k].set_xlabel('')
                if col != 0:
                    mp.grid[k].set_ylabel('')
                    
                self.plot_info[k] = {}
                self.plot_info[k]['axes'] = [p2, p1]
                self.plot_info[k]['data'] = tohist
                self.plot_info[k]['ivar'] = iv
                self.plot_info[k]['bins'] = [bins[j], bins[-1::-1][i]]
                self.plot_info[k]['multiplier'] = [multiplier[j], multiplier[-1::-1][i]]
                self.plot_info[k]['take_log'] = [take_log[j], take_log[-1::-1][i]] 
                
                # Input values
                if not inputs:
                    continue
                                
                self.plot_info[k]['input'] = (xin, yin)
                                                                    
                # Plot as dotted lines
                if xin is not None:
                    mp.grid[k].plot([xin]*2, mp.grid[k].get_ylim(), color='k',
                        ls=':', zorder=20)
                if yin is not None:
                    mp.grid[k].plot(mp.grid[k].get_xlim(), [yin]*2, color='k',
                        ls=':', zorder=20)

                    
        if oned:
            mp.grid[np.intersect1d(mp.left, mp.top)[0]].set_yticklabels([])
        
        if fix:
            mp.fix_ticks(oned=oned, N=ticks, rotate_x=45, rotate_y=45)
        
        if not had_mp:
            mp.rescale_axes(tighten_up=tighten_up)
    
        if label_panels is not None and (not had_mp):
            mp = self._label_panels(mp, label_panels)
    
        return mp
        
    def _label_panels(self, mp, label_panels):
        letters = list(string.ascii_lowercase)
        letters.extend([let*2 for let in list(string.ascii_lowercase)])
        
        ct = 0
        for ax in mp.grid:
            if ax is None:
                continue
        
            if label_panels == 'upper left':
                ax.annotate('(%s)' % letters[ct], (0.05, 0.95),
                    xycoords='axes fraction', ha='left', va='top')
            elif label_panels == 'upper right':
                ax.annotate('(%s)' % letters[ct], (0.95, 0.95),
                    xycoords='axes fraction', ha='right', va='top')
            elif label_panels == 'upper center':
                ax.annotate('(%s)' % letters[ct], (0.5, 0.95),
                    xycoords='axes fraction', ha='center', va='top')
            elif label_panels == 'lower right':
                ax.annotate('(%s)' % letters[ct], (0.95, 0.95),
                    xycoords='axes fraction', ha='right', va='top')                
            else:
                print "WARNING: Uncrecognized label_panels option."
                break
        
            ct += 1    
        
        pl.draw()    
        
        return mp
        
    def ReconstructedFunction(self, name, ivar=None, fig=1, ax=None,
        use_best=False, percentile=0.68, take_log=False, un_log=False, 
        multiplier=1, skip=0, stop=None, return_data=False, z_to_freq=False,
        best='maxL', fill=True, **kwargs):    
        """
        Reconstructed evolution in whatever the independent variable is.
        
        Parameters
        ----------
        name : str
            Name of blob you're interested in.
        ivar : list, np.ndarray
            List of values (or nested list) of independent variables. If 
            blob is 2-D, only need to provide the independent variable for
            one of the dimensions, e.g., 
                
                # If LF data, plot LF at z=3.8
                ivar = [3.8, None]
                
            or 
            
                # If LF data, plot z evolution of phi(MUV=-20)
                ivar = [None, -20]
        
        percentile : bool, float    
            If not False, should be the confidence interval to plot, e.g, 0.68.
        use_best : bool
            If True, will plot the maximum likelihood reconstructed
            function. Otherwise, will use `percentile` and plot shaded region.
            
        """
        
        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True
        
        if percentile:    
            q1 = 0.5 * 100 * (1. - percentile)    
            q2 = 100 * percentile + q1    
        
        info = self.blob_info(name)
        nd = info[2]
        
        if nd == 1:
            ivars = np.atleast_2d(self.blob_ivars[info[0]])
        else:
            ivars = self.blob_ivars[info[0]]
        
        if nd != 1 and (ivar is None):
            raise NotImplemented('If not 1-D blob, must supply one ivar!')
                
        # Grab the maximum likelihood point 'cuz why not
        if self.is_mcmc:
            if best == 'median':
                N = len(self.logL)
                psorted = np.argsort(self.logL)
                loc = psorted[int(N / 2.)]
            else:
                loc = np.argmax(self.logL[skip:stop])
        
        # 1-D case 
        if nd == 1:
            
            # Read in the independent variable(s)
            xarr = ivars[0]
            
            tmp, is_log = self.ExtractData(name, 
                take_log=take_log, un_log=un_log, multiplier=multiplier)
            
            data = tmp[name].squeeze()
            
            y = []
            for i, x in enumerate(xarr):
                if (use_best and self.is_mcmc):
                    y.append(data[:,i][skip:stop][loc])
                elif percentile:
                    lo, hi = np.percentile(data[:,i][skip:stop].compressed(), 
                        (q1, q2))
                    y.append((lo, hi))
                else:
                    dat = data[:,i][skip:stop].compressed()
                    lo, hi = dat.min(), dat.max()
                    y.append((lo, hi))
        elif nd == 2:
            if ivar[0] is None:
                scalar = ivar[1]
                vector = xarr = ivars[0]
                slc = slice(-1, None, -1)

            else:
                scalar = ivar[0]
                vector = xarr = ivars[1]
                slc = slice(0, None, 1)
                                                
            y = []
            for i, value in enumerate(vector):
                iv = [scalar, value][slc]
                data, is_log = self.ExtractData(name, ivar=iv,
                    take_log=take_log, un_log=un_log, multiplier=multiplier)
                        
                if (use_best and self.is_mcmc):
                    y.append(data[name][skip:stop][loc])        
                elif percentile:
                    lo, hi = np.percentile(data[name][skip:stop].compressed(),
                        (q1, q2))
                    y.append((lo, hi))    
                else:
                    dat = data[name][skip:stop].compressed()
                    lo, hi = dat.min(), dat.max()
                    y.append((lo, hi))
                    
        # Convert redshifts to frequencies    
        if z_to_freq:
            xarr = nu_0_mhz / (1. + xarr)
                        
        # Where y is zero, set to small number?                
                        
        if use_best and self.is_mcmc:
            if take_log:
                y = 10**y
        
            ax.plot(xarr, y, **kwargs)
        else:
            y = np.array(y).T
        
            if take_log:
                y = 10**y
            else:
                zeros = np.argwhere(y == 0)
                for element in zeros:
                    y[element[0],element[1]] = 1e-15
            
            if fill:
                ax.fill_between(xarr, y[0], y[1], **kwargs)
            else:
                ax.plot(xarr, y[0], **kwargs)
                ax.plot(xarr, y[1], **kwargs)
                
        
        pl.draw()
                 
        if return_data:
            return ax, xarr, y
        else:            
            return ax
        
    def RedshiftEvolution(self, blob, ax=None, redshifts=None, fig=1,
        like=0.68, take_log=False, bins=20, label=None,
        plot_bands=False, limit=None, **kwargs):
        """
        Plot constraints on the redshift evolution of given quantity.
        
        Parameters
        ----------
        blob : str
            
        Note
        ----
        If you get a "ValueError: attempt to get argmin of an empty sequence"
        you might consider setting take_log=True.    
            
        """    
        
        if plot_bands and (limit is not None):
            raise ValueError('Choose bands or a limit, not both!')
        
        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True      
        
        try:
            ylabel = default_labels[blob]
        except KeyError:
            ylabel = blob
        
        if redshifts is None:
            redshifts = self.blob_redshifts
            
        if plot_bands or (limit is not None):
            x = []; ymin = []; ymax = []
            
        for i, z in enumerate(redshifts):
            
            # Skip turning points for banded plots
            if type(z) == str and plot_bands:
                continue
            
            # Only plot label once
            if i == 0:
                l = label
            else:
                l = None
            
            try:
                value, (blob_err1, blob_err2) = \
                    self.get_1d_error(blob, ivar=z, like=like, take_log=take_log,
                    bins=bins, limit=limit)
            except TypeError:
                continue
            
            if value is None:
                continue    
            
            # Error on redshift
            if type(z) == str and not plot_bands:
                if blob == 'dTb':
                    mu_z, (z_err1, z_err2) = \
                        self.get_1d_error('nu', ivar=z, nu=like, bins=bins)
                else:
                    mu_z, (z_err1, z_err2) = \
                        self.get_1d_error('z', ivar=z, nu=like, bins=bins)

                xerr = np.array(z_err1, z_err2).T
            else:
                mu_z = z
                xerr = None
            
            if plot_bands:
                if blob == 'dTb':
                    x.append(nu_0_mhz / (1. + mu_z))
                else:
                    x.append(z)
                ymin.append(value - blob_err1)
                ymax.append(value + blob_err2)
            elif limit is not None:
                if blob == 'dTb':
                    x.append(nu_0_mhz / (1. + mu_z))
                else:
                    x.append(z)
                ymin.append(value)
            else:                                    
                ax.errorbar(mu_z, value, 
                    xerr=xerr, 
                    yerr=np.array(blob_err1, blob_err2).T, 
                    lw=2, elinewidth=2, capsize=3, capthick=1, label=l,
                    **kwargs)        
        
        if plot_bands:
            ax.fill_between(x, ymin, ymax, **kwargs)
        elif limit is not None:
            ax.plot(x, ymin, **kwargs)
        
        # Look for populations
        m = re.search(r"\{([0-9])\}", blob)
        
        if m is None:
            prefix = blob
        else:
            # Population ID number
            num = int(m.group(1))
            
            # Pop ID excluding curly braces
            prefix = blob.split(m.group(0))[0]
        
        if blob == 'dTb':
            ax.set_xlabel(r'$\nu \ (\mathrm{MHz})$')
        else:
            ax.set_xlabel(r'$z$')
            
        ax.set_ylabel(ylabel)

        pl.draw()
        
        return ax
        
    def CovarianceMatrix(self, pars, ivar=None):
        """
        Compute covariance matrix for input parameters.

        Parameters
        ----------
        pars : list
            List of parameter names to include in covariance estimate.

        Returns
        -------
        Returns vector of mean, and the covariance matrix itself.
        
        """
                
        data, is_log = self.ExtractData(pars, ivar=ivar)
        
        blob_vec = []
        for i in range(len(pars)):
            blob_vec.append(data[pars[i]])    
        
        mu  = np.ma.mean(blob_vec, axis=1)
        cov = np.ma.cov(blob_vec)

        return mu, cov    
        
    def AssembleParametersList(self, N=None, loc=None, include_bkw=False):
        """
        Return dictionaries that can be used to initialize an ares 
        simulation. 
        
        N : int
            Maximum number of models to return.
        include_bkw : bool  
            Include base_kwargs?
            
        Returns
        -------
        List of dictionaries. Maximum length: `N`.
            
        """ 
        
        all_kwargs = []
        for i, element in enumerate(self.chain):
            
            if self.mask[i]:
                continue
            
            if loc is not None:
                if i != loc:
                    continue
            elif N is not None:
                if i >= N:
                    break
                
            if include_bkw:
                kwargs = self.base_kwargs.copy()    
            else:
                kwargs = {}
                
            for j, parameter in enumerate(self.parameters):
                if self.is_log[j]:
                    kwargs[parameter] = 10**self.chain[i,j]
                else:
                    kwargs[parameter] = self.chain[i,j]
                    
            all_kwargs.append(kwargs.copy())

        if loc is not None:
            return all_kwargs[0]
        else:
            return all_kwargs

    def CorrelationMatrix(self, pars, ivar=None, fig=1, ax=None):
        """
        Plot correlation matrix.
        """
    
        mu, cov = self.CovarianceMatrix(pars, ivar=ivar)
    
        corr = correlation_matrix(cov)
    
        if ax is None:
            fig = pl.figure(fig); ax = fig.add_subplot(111)
    
        cax = ax.imshow(corr, interpolation='none', cmap='RdBu_r', 
            vmin=-1, vmax=1)
        cb = pl.colorbar(cax)
    
        return ax
    
    def get_blob(self, name, ivar=None):
        """
        Extract an array of values for a given quantity.
        
        ..note:: If ivar is not supplied, this is equivalent to just reading
            all data from disk.
        
        Parameters
        ----------
        name : str
            Name of quantity
        ivar : list, tuple, array
            Independent variables a given blob may depend on.
            
        """
                        
        i, j, nd, dims = self.blob_info(name)
        blob = self.get_blob_from_disk(name)
        
        if nd == 0:
            return blob
        elif nd == 1:
            if ivar is None:
                return blob
            else:
                k = np.argmin(np.abs(self.blob_ivars[i] - ivar))
                return blob[:,k]
        elif nd == 2:
            if ivar is None:
                return blob
                    
            assert len(ivar) == 2, "Must supply 2-D coordinate for blob!"
            k1 = np.argmin(np.abs(self.blob_ivars[i][0] - ivar[0]))
            k2 = np.argmin(np.abs(self.blob_ivars[i][1] - ivar[1]))
            return blob[:,k1,k2]    
    
    def max_likelihood_parameters(self, method='median'):
        """
        Return parameter values at maximum likelihood point.
        """
                    
        if method == 'median':
            N = len(self.logL)
            psorted = np.sort(self.logL)
            iML = psorted[int(N / 2.)]
        else:
            iML = np.argmax(self.logL)
        
        self._max_like_pars = {}
        for i, par in enumerate(self.parameters):
            if self.is_log[i]:
                self._max_like_pars[par] = 10**self.chain[iML,i]
            else:
                self._max_like_pars[par] = self.chain[iML,i]
        
        return self._max_like_pars
        
    def DeriveBlob(self, func=None, fields=None, expr=None, varmap=None, 
        save=True, name=None, clobber=False):
        """
        Derive new blob from pre-existing ones.
        
        Parameters
        ----------
        Either supply the first two arguments:
        func : function!
            A function of two variables: ``data`` (a dictionary containing the 
            data) and ``ivars``, which contain the independent variables for
            each field in ``data``.
        fields : list, tuple
            List of quantities required by ``func``.
            
        OR the second two:    
            
        expr : str
            For example, 'x - y'
        varmap : dict
            Relates variables in `expr` to blobs. For example, 
            
            varmap = {'x': 'nu_D', 'y': 'nu_C'}
        
        The remaining parameters are:
        
        save : bool
            Save to disk? If not, just returns array.
        name : str
            If save==True, this is a name for this new blob that we can use
            to call it up later.
        clobber : bool
            If file with same ``name`` exists, overwrite it?
        
        """    
        
        if func is not None:
            data, is_log = self.ExtractData(fields)
            
            # Grab ivars
            ivars = {}
            for key in data:
                i, j, nd, size = self.blob_info(key)
                ivars[key] = self.blob_ivars[i]
                
            result = func(data, ivars)
        
        else:
        
            blobs = varmap.values()
            
            data, is_log = self.ExtractData(blobs)
            
            # Assign data to variable names
            for var in varmap.keys():
                exec('%s = data[\'%s\']' % (var, varmap[var]))
            
            result = eval(expr)
        
        if save:
            assert name is not None, "Must supply name for new blob!"
            
            # First dimension is # of samples
            nd = len(result.shape) - 1
            
            fn = '%s.blob_%id.%s.pkl' % (self.prefix, nd, name)
            
            if os.path.exists(fn) and (not clobber):
                print '%s exists! Set clobber=True or remove by hand.' % fn
                data, is_log = self.ExtractData(name)
                return data[name]
        
            f = open(fn, 'wb')
            pickle.dump(result, f)
            f.close()
        
        return result
        
    def save(self, pars, z=None, path='.', fmt='hdf5', clobber=False):
        """
        Extract data from chain or blobs and output to separate file(s).
        
        Parameters
        ----------
        pars : str, list, tuple
            Name of parameter (or list of parameters) or blob(s) to extract.
        z : int, float, str, list, tuple
            Redshift(s) of interest.
        fmt : str
            Options: 'hdf5' or 'pkl'
        path : str
            By default, will save files to CWD. Can modify this if you'd like.
                
        """
        
        if type(pars) not in [list, tuple]:
            pars = [pars]
        if type(z) not in [list, tuple]:
            z = [z] * len(pars)

        # Output to HDF5
        if fmt == 'hdf5':

            # Loop over parameters and save to disk
            for i, par in enumerate(pars):
                fn = '%s/%s.subset.%s.%s' % (path,self.fn, par, fmt)

                # If the file exists, see if it already contains data for the
                # redshifts of interest. If not append. If so, raise error.
                if os.path.exists(fn) and (not clobber):

                    if z[i] is None:
                        raise IOError('%s exists!' % fn)

                    # Check for redshift data    
                    f = h5py.File(fn, 'r')
                    ids, redshifts = zip(*f.attrs.items())
                    f.close()

                    ids_ints = map(int, ids)
                    id_start = max(ids_ints) + 1

                    if z[i] in redshifts:
                        print '%s exists! As does this dataset. Continuing...' % fn
                        continue
                else:
                    id_start = 0
                    
                idnum = str(id_start).zfill(5)
                
                if z is None:
                    attr = -99999
                else:
                    attr = z[i]
            
                # Stick requested fields in a dictionary
                data, is_log = self.ExtractData(par, z=z[i])
            
                f = h5py.File(fn, 'a')
                f.attrs.create(idnum, attr)

                ds = f.create_dataset(idnum, data=data[par])
                ds.attrs.create('mask', data[par].mask)
                    
                f.close()
                print "Wrote %s." % fn    
                
        else:
            raise NotImplemented('Only support for hdf5 so far. Sorry!')
        
    def set_axis_labels(self, ax, pars, is_log, take_log=False, un_log=False,
        cb=None, labels={}):
        """
        Make nice axis labels.
        """
                
        pars, take_log, multiplier, un_log, ivar = \
            self._listify_common_inputs(pars, take_log, 1.0, un_log, None)

        if type(is_log) != dict:
            tmp = {par:is_log[i] for i, par in enumerate(pars)}
            is_log = tmp
        if type(take_log) != dict:
            tmp = {par:take_log[i] for i, par in enumerate(pars)}
            take_log = tmp        
            
        # Prep for label making
        labeler = self.labeler = Labeler(pars, is_log, extra_labels=labels,
            **self.base_kwargs)

        # x-axis first
        ax.set_xlabel(labeler.label(pars[0], take_log=take_log[pars[0]], 
            un_log=un_log[0]))
    
        if len(pars) == 1:
            ax.set_ylabel('PDF')
            pl.draw()
            return
    
        ax.set_ylabel(labeler.label(pars[1], take_log=take_log[pars[1]], 
            un_log=un_log[1]))
            
        pl.draw()
                        
        xt = []
        for i, x in enumerate(ax.get_xticklabels()):
            xt.append(x.get_text())
        
        ax.set_xticklabels(xt, rotation=45.)
        
        yt = []
        for i, x in enumerate(ax.get_yticklabels()):
            yt.append(x.get_text())
        
        ax.set_yticklabels(yt, rotation=45.)
            
        # colorbar
        if cb is not None and len(pars) > 2:
            cb.set_label(labeler.label(pars[2], take_log=take_log[pars[2]], 
                un_log=un_log[2]))
        
        pl.draw()

    
