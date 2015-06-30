"""

ModelFit.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Apr 28 11:19:03 MDT 2014

Description: For analysis of MCMC fitting.

"""

import numpy as np
import matplotlib as mpl
import re, os, string, time
import matplotlib.pyplot as pl
from ..util import ProgressBar
import matplotlib._cntr as cntr
from ..physics import Cosmology
from .MultiPlot import MultiPanel
from ..inference import ModelGrid
from matplotlib.patches import Rectangle
from ..physics.Constants import nu_0_mhz
from .Global21cm import Global21cm as aG21
from ..util import labels as default_labels
from ..util.PrintInfo import print_model_set
from ..util.ParameterFile import count_populations
from .DerivedQuantities import DerivedQuantities as DQ
from .DerivedQuantities import registry_special_Q
from ..simulations.Global21cm import Global21cm as sG21
from ..util.SetDefaultParameterValues import SetAllDefaults, TanhParameters
from ..util.Stats import Gauss1D, GaussND, error_1D, error_2D, _error_2D_crude, \
    rebin, correlation_matrix
from ..util.ReadData import read_pickled_dict, read_pickle_file, \
    read_pickled_chain, read_pickled_logL, fcoll_gjah_to_ares, \
    tanh_gjah_to_ares

try:
    from scipy.optimize import fmin
    from scipy.integrate import dblquad
except ImportError:
    pass

import pickle    

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

tanh_pars = TanhParameters()

def_kwargs = {}

def patch_pinfo(pars):
    new_pars = []
    for par in pars:

        if par in tanh_gjah_to_ares:
            new_pars.append(tanh_gjah_to_ares[par])
        elif par in fcoll_gjah_to_ares:
            new_pars.append(fcoll_gjah_to_ares[par])
        else:
            new_pars.append(par)
    
    return new_pars

def parse_blobs(name):
    nsplit = name.split('_')
    
    if len(nsplit) == 2:
        pre, post = nsplit
    elif len(nsplit) == 3:
        pre, mid, post = nsplit

        pre = pre + mid
    
    if pre in default_labels:
        pass
        
    return None 
        
def logify_str(s, sup=None):
    s_no_dollar = str(s.replace('$', ''))
    
    new_s = s_no_dollar
    
    if sup is not None:
        new_s += sup_scriptify_str(s)
        
    return r'$\mathrm{log}_{10}' + new_s + '$'
    
def undo_mathify(s):
    return str(s.replace('$', ''))
    
def mathify_str(s):
    return r'$%s$' % s    
    
def make_label(name, take_log=False, labels=None):
    """
    Take a string and make it a nice (LaTeX compatible) axis label. 
    """
    
    if labels is None:
        labels = default_labels
        
    # Check to see if it has a population ID # tagged on the end
    m = re.search(r"\{([0-9])\}", name)
    
    if m is None:
        num = None
        prefix = name
        if prefix in labels:
            label = labels[prefix]
        else:
            label = r'$%s$' % prefix
    else:
        num = int(m.group(1))
        prefix = name.split(m.group(0))[0]
        if prefix in labels:
            label = r'$%s$' % (undo_mathify(labels[prefix].split(m.group(0))[0]) + '^{%i}' % num)
        else:
            label = r'$%s$' % prefix
        
    if take_log:        
        return mathify_str('\mathrm{log}_{10}' + undo_mathify(label))
    else:
        return label
        
def err_str(label, mu, err, log, labels=None):
    s = undo_mathify(make_label(label, log, labels))

    s += '=%.3g^{+%.2g}_{-%.2g}' % (mu, err[1], err[0])
    
    return r'$%s$' % s

def def_par_names(N):
    return [i for i in np.arange(N)]

def def_par_labels(i):
    return 'parameter # %i' % i
                
class ModelSubSet(object):
    def __init__(self):
        pass

class ModelSet(object):
    def __init__(self, data):
        """
        Parameters
        ----------
        data : instance, str
            prefix for a bunch of files ending in .chain.pkl, .pinfo.pkl, etc.,
            or a ModelSubSet instance.

        """

        # Read in data from file (assumed to be pickled)
        if type(data) == str:
            self.prefix = prefix = data
            
            i = prefix.rfind('/') # forward slash index

            # This means we're sitting in the right directory already
            if i == - 1:
                self.path = './'
                self.fn = prefix
            else:
                self.path = prefix[0:i+1]
                self.fn = prefix[i+1:]

            print_model_set(self)
                    
            if not self.is_mcmc:
                
                self.grid = ModelGrid(**self.base_kwargs)
                
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
            self._fails = data.fails
            
            self.mask = np.zeros_like(data.blobs)    
            self.mask[np.isinf(data.blobs)] = 1
            self.mask[np.isnan(data.blobs)] = 1
            self._blobs = np.ma.masked_array(data.blobs, mask=self.mask)

            self._blob_names = data.blob_names
            self._blob_redshifts = data.blob_redshifts
            self._parameters = data.parameters
            self._is_mcmc = data.is_mcmc
            
            if self.is_mcmc:
                self.logL = data.logL
            else:
                try:
                    self.load = data.load
                except AttributeError:
                    pass
                try:            
                    self.axes = data.axes
                except AttributeError:
                    pass
                try:
                    self.grid = data.grid
                except AttributeError:
                    pass

            self.Nd = int(self.chain.shape[-1])       
                
        else:
            raise TypeError('Argument must be ModelSubSet instance or filename prefix')              
    
        #try:
        #    self._fix_up()
        #except AttributeError:
        #    pass
        
    @property
    def load(self):
        if not hasattr(self, '_load'):
            if os.path.exists('%s.load.pkl' % self.prefix):
                self._load = read_pickle_file('%s.load.pkl' % self.prefix)
            else:
                self._load = None
        
        return self._load
            
    @property
    def base_kwargs(self):
        if not hasattr(self, '_base_kwargs'):
            if os.path.exists('%s.setup.pkl' % self.prefix):
                f = open('%s.setup.pkl' % self.prefix, 'rb')
                self._base_kwargs = pickle.load(f)
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
                self._parameters = self._is_log = None
        
        return self._parameters
        
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
            
    @property
    def blob_names(self):
        if not hasattr(self, '_blob_names'):
            if os.path.exists('%s.binfo.pkl' % self.prefix):
                f = open('%s.binfo.pkl' % self.prefix, 'rb')
                self._blob_names, self._blob_redshifts = \
                    map(list, pickle.load(f))
                f.close()
                
        return self._blob_names
    
    @property
    def blob_redshifts(self):
        if not hasattr(self, '_blob_redshifts'):
            names = self.blob_names
        
        return self._blob_redshifts

    @property
    def blobs(self):
        if not hasattr(self, '_blobs'):
            if os.path.exists('%s.blobs.hdf5' % self.prefix) and have_h5py:
                t1 = time.time()
                f = h5py.File('%s.blobs.hdf5' % self.prefix, 'r')
                bs = f['blobs'].value
                self._blobs = np.ma.masked_array(bs, mask=f['mask'].value)
                f.close()
                t2 = time.time()    
                
                if rank == 0:
                    print "Loaded %s.blobs.hdf5 in %.2g seconds.\n" \
                     % (self.prefix, t2 - t1)
            
            elif os.path.exists('%s.blobs.pkl' % self.prefix):
                
                try:
                    if rank == 0:
                        print "Loading %s.blobs.pkl..." % self.prefix
                    
                    t1 = time.time()
                    blobs = read_pickle_file('%s.blobs.pkl' % self.prefix)
                    t2 = time.time()    
                        
                    if rank == 0:
                        print "Loaded %s.blobs.pkl in %.2g seconds.\n" \
                         % (self.prefix, t2 - t1)
                        
                    self._mask = np.zeros_like(blobs)    
                    self._mask[np.isinf(blobs)] = 1
                    self._mask[np.isnan(blobs)] = 1
                    self._blobs = np.ma.masked_array(blobs, mask=self._mask)
                    
                except:
                    if rank == 0:
                        print "WARNING: Error loading blobs."    

                if have_h5py:
                    f = h5py.File('%s.blobs.hdf5' % self.prefix, 'w')
                    f.create_dataset('blobs', data=self._blobs)
                    f.create_dataset('mask', data=self._mask)
                    f.close()

                    if rank == 0:
                        print 'Saved %s.blobs.hdf5' % self.prefix            

            else:
                self._blobs = None            

        return self._blobs
        
    #def _load_blob(self, blob):
    #    
    #    fn = '%s'
    #    
    #    f = h5py.File(fn, 'r')
    #    
    #    results = {}
    #    for key in f:
    #        mask = f[key].attrs.get('mask')
    #        results[key] = np.ma.array(f[key].value, mask=mask)
            
        

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
            if os.path.exists('%s.chain.pkl' % self.prefix):
                if rank == 0:
                    print "Loading %s.chain.pkl..." % self.prefix

                t1 = time.time()
                self._chain = read_pickled_chain('%s.chain.pkl' % self.prefix)
                t2 = time.time()

                if rank == 0:
                    print "Loaded %s.chain.pkl in %.2g seconds.\n" \
                        % (self.prefix, t2-t1)
            else:
                self._chain = None

        return self._chain        
    
    @property
    def logL(self):
        if not hasattr(self, '_logL'):            
            if os.path.exists('%s.logL.pkl' % self.prefix):
                self._logL = read_pickled_logL('%s.logL.pkl' % self.prefix)
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
        
        # Figure out what these values translate to.
        
        # Get basic info about plot
        x = self.plot_info['x']
        y = self.plot_info['y']
        z = self.plot_info['z']
        take_log = self.plot_info['log']
        multiplier = self.plot_info['multiplier']
        
        pars = [x, y]
        
        # Index corresponding to this redshift
        iz = self.blob_redshifts.index(z)
        
        # Use slice routine!
        constraints = \
        {
         x: [z, lambda xx: 1 if lx <= xx <= (lx + dx) else 0],
         y: [z, lambda yy: 1 if ly <= yy <= (ly + dy) else 0],
        }
        
        i = 0
        while hasattr(self, 'slice_%i' % i):
            i += 1
        
        tmp = self._slice_by_nu(pars=pars, z=z, like=0.95, 
            take_log=take_log, **constraints)
        
        setattr(self, 'slice_%i' % i, tmp)        
        
        print "Saved result to slice_%i attribute." % i
        
    def ReRunModels(self, prefix=None, N=None, random=False, last=False, 
        clobber=False, save_freq=10, **kwargs):
        """
        Take list of dictionaries and re-run each as a Global21cm model.

        Parameters
        ----------
        N : int
            Draw N samples from chain, and re-run those models.
            If None, will re-run all models.
        random : bool
            If True, draw N *random* samples, rather than just the first N.
        last : bool
            If True, take last ``N`` samples from chain, rather than first.
            
        prefix : str
            Prefix of files to be saved. There will be three:
                i) prefix.chain.pkl
                ii) prefix.blobs.pkl
                iii) prefix.pinfo.pkl

        Returns
        -------
        Nothing, just saves stuff to disk.
        
        """
                
        had_N = True
        if N is None:
            had_N = False
            N = self.chain.shape[0]
                    
        model_num = np.arange(N)
        
        if had_N:
            if random:
                if size > 1:
                    raise ValueError('This will cause each processor to run different models!')
                np.random.shuffle(model_num)
            
            if last:
                model_ids = model_num[-N:]
            else:    
                model_ids = model_num[0:N]            
        else:
            model_ids = model_num    
                        
        # Create list of models to run
        models = []
        for i, model in enumerate(model_ids):
            tmp = {}
            for j, par in enumerate(self.parameters):
                tmp[par] = self.chain[i,j]
            
            models.append(tmp.copy())
                                
        # Take advantage of pre-existing machinery to run them
        mg = self.mg = ModelGrid(**kwargs)
        mg.set_models(models)
        mg.is_log = self.is_log
        mg.LoadBalance(0)

        mg.run(prefix, clobber=clobber, save_freq=save_freq)

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

    @property
    def derived_blobs(self):
        """
        Total rates, convert to rate coefficients.
        """

        if hasattr(self, '_derived_blobs'):
            return self._derived_blobs
            
        if os.path.exists('%s.dblobs.hdf5' % self.prefix) and have_h5py:
            f = h5py.File('%s.dblobs.hdf5' % self.prefix, 'r')
            dbs = f['derived_blobs'].value
            self._derived_blobs = np.ma.masked_array(dbs, mask=f['mask'].value)
            self._derived_blob_names = list(f['derived_blob_names'].value)
            f.close()
            return self._derived_blobs

        #if self.blobs is None:
        #    return   

        # Just a dummy class
        pf = ModelSubSet()
        pf.Npops = self.Npops

        # Create data container to mimic that of a single run,
        dqs = []
        self._derived_blob_names = []
        for i, redshift in enumerate(self.blob_redshifts):

            if type(redshift) is str:
                z = self.extract_blob('z', redshift)[i] 
            else:
                z = redshift

            data = {}
            data['z'] = np.array([z])
            
            for j, key in enumerate(self.blob_names):
                data[key] = self.blobs[:,i,j]
                            
            _dq = DQ(data, pf)
            
            # SFRD
            _dq.build(**registry_special_Q)
            
            dqs.append(_dq.derived_quantities.copy())

            for key in _dq.derived_quantities:
                if key in self._derived_blob_names:
                    continue
                
                self._derived_blob_names.append(key)

        # (Nlinks, Nz, Nblobs)
        shape = list(self.blobs.shape[:-1])
        shape.append(len(self._derived_blob_names))

        self._derived_blobs = np.ones(shape) * np.inf
        
        for i, redshift in enumerate(self.blob_redshifts):
            
            data = dqs[i]
            for key in data:
                j = self._derived_blob_names.index(key)
                self._derived_blobs[:,i,j] = data[key]

                
        mask = np.ones_like(self._derived_blobs)    
        #mask[np.isinf(self._derived_blobs)] = 1
        #mask[np.isnan(self._derived_blobs)] = 1
        mask[np.isfinite(self._derived_blobs)] = 0
        
        self.dmask = mask
        
        self._derived_blobs = np.ma.masked_array(self._derived_blobs, 
            mask=mask)
            
        # Save to disk!
        if not os.path.exists('%s.dblobs.hdf5') and have_h5py:
            f = h5py.File('%s.dblobs.hdf5' % self.prefix, 'w')
            f.create_dataset('mask', data=mask)
            f.create_dataset('derived_blobs', data=self._derived_blobs)
            f.create_dataset('derived_blob_names', data=self._derived_blob_names)
            f.close()
            
            if rank == 0:
                print 'Saved %s.dblobs.hdf5' % self.prefix

        return self._derived_blobs

    def set_constraint(self, add_constraint=False, **constraints):
        """
        For ModelGrid calculations, the likelihood must be supplied 
        after-the-fact.

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

                j = self.blob_redshifts.index(z)
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

    def Scatter(self, x, y, z=None, c=None, ax=None, fig=1, 
        take_log=False, multiplier=1., **kwargs):
        """
        Show occurrences of turning points B, C, and D for all models in
        (z, dTb) space, with points color-coded to likelihood.
    
        Parameters
        ----------
        x : str
            Fields for the x-axis.
        y : str
            Fields for the y-axis.        
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

        if 'labels' in kwargs:
            labels_tmp = default_labels.copy()
            labels_tmp.update(kwargs['labels'])
            labels = labels_tmp
        else:
            labels = default_labels

        if type(take_log) == bool:
            take_log = [take_log] * 2
        if type(multiplier) in [int, float]:
            multiplier = [multiplier] * 2

        if z is not None:
            j = self.blob_redshifts.index(z)

        if x in self.parameters:
            xdat = self.chain[:,self.parameters.index(x)]
        else:
            if z is None:
                raise ValueError('Must supply redshift!')
            if x in self.blob_names:
                xdat = self.blobs[:,j,self.blob_names.index(x)]
            else:
                xdat = self.derived_blobs[:,j,self.derived_blob_names.index(x)]
        
        if y in self.parameters:
            ydat = self.chain[:,self.parameters.index(y)]
        else:
            if z is None:
                raise ValueError('Must supply redshift!')
            if y in self.blob_names:
                ydat = self.blobs[:,j,self.blob_names.index(y)]
            else:
                ydat = self.derived_blobs[:,j,self.derived_blob_names.index(y)]
        
        cdat = None
        if c in self.parameters:
            cdat = self.chain[:,self.parameters.index(c)]
        elif c == 'load':
            cdat = self.load
        elif c is None:
            pass
        else:
            if z is None:
                raise ValueError('Must supply redshift!')
            if c in self.blob_names:
                cdat = self.blobs[:,j,self.blob_names.index(c)]
            else:
                cdat = self.derived_blobs[:,j,self.derived_blob_names.index(c)]

        if take_log[0]:
            xdat = np.log10(xdat)
        if take_log[1]:
            ydat = np.log10(ydat)
        if len(take_log) == 3 and cdat is not None:
            if take_log[2]:
                cdat = np.log10(cdat)  

        if hasattr(self, 'weights') and cdat is None:
            scat = ax.scatter(xdat, ydat, c=self.weights, **kwargs)
        elif cdat is not None:
            scat = ax.scatter(xdat, ydat, c=cdat, **kwargs)
        else:
            scat = ax.scatter(xdat, ydat, **kwargs)

        ax.set_xlabel(make_label(x, take_log=take_log[0]))
        ax.set_ylabel(make_label(y, take_log=take_log[1]))
        #if take_log[0]:
        #    ax.set_xlabel(logify_str(labels[self.get_par_prefix(x)]))
        #else:
        #    ax.set_xlabel(labels[self.get_par_prefix(x)])

        #if take_log[1]: 
        #    ax.set_ylabel(logify_str(labels[self.get_par_prefix(y)]))
        #else:
        #    ax.set_ylabel(labels[self.get_par_prefix(y)])
                            
        if c is not None:
            cb = pl.colorbar(scat)
            try:
                if take_log[2]: 
                    cb.set_label(logify_str(labels[self.get_par_prefix(c)]))
                else:
                    cb.set_label(labels[self.get_par_prefix(c)])
            except IndexError:
                cb.set_label(labels[self.get_par_prefix(c)])
        
            self._cb = cb
        
        pl.draw()
        
        self._ax = ax
        self.plot_info = {'x': x, 'y': y, 'log': take_log, 
            'multiplier': multiplier, 'z': z}
    
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
    
    def get_1d_error(self, par, z=None, bins=500, nu=0.68, take_log=False,
        limit=None, multiplier=1.):
        """
        Compute 1-D error bar for input parameter.
        
        Parameters
        ----------
        par : str
            Name of parameter. 
        bins : int
            Number of bins to use in histogram
        nu : float
            Percent likelihood enclosed by this 1-D error
        limit : str
            Valid options: 'lower' and 'upper', if not None.
        Returns
        -------
        Tuple, (maximum likelihood value, negative error, positive error).

        """

        pars, take_log, multiplier, z = \
            self._listify_common_inputs([par], take_log, multiplier, z)

        to_hist, is_log = self.ExtractData(par, z=z, take_log=take_log, 
            multiplier=multiplier)

        # Need to weight results of non-MCMC runs explicitly
        if not hasattr(self, 'weights'):
            weights = None
        else:
            weights = self.weights

        # Apply mask to weights
        if weights is not None and to_hist.shape != weights.shape:
            weights = weights[np.logical_not(mask)]
        
        if hasattr(to_hist[par], 'compressed'):
            to_hist[par] = to_hist[par].compressed()
        
        hist, bin_edges = \
            np.histogram(to_hist[par], density=True, bins=bins, 
            weights=weights)

        bc = rebin(bin_edges)

        if to_hist is []:
            return None, (None, None)

        mu, sigma = error_1D(bc, hist, nu=nu, limit=limit)
                
        try:
            mu, sigma = error_1D(bc, hist, nu=nu, limit=limit)
        except ValueError:
            return None, (None, None)

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
        
    def Slice(self, pars=None, z=None, like=0.68, take_log=False, bins=20, 
        **constraints):
        """
        Return revised ("sliced") dataset given set of criteria.
                
        Parameters
        ----------
        like : float
            If supplied, return a new ModelSet instance containing only the
            models with likelihood's exceeding this value.
        constraints : dict
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

        return self._slice_by_nu(pars, z=z, take_log=take_log, bins=bins, 
            like=like, **constraints)
        
    def ExamineFailures(self, N=1):
        """
        Try to figure out what went wrong with failed models.
        
        Picks a random subset of failed models, plots them, and returns
        the analysis instances associated with each.
        
        Parameters
        ----------
        N : int
            Number of failed models to plot.
            
        """    
        
        kw = self.base_kwargs.copy()
                
        Nf = len(self.fails)
        
        r = np.arange(Nf)
        np.random.shuffle(r)
        
        ax = None
        objects = {}
        for i in range(N):
            
            idnum = r[i]
            
            p = self.base_kwargs.copy()
            p.update(self.fails[idnum])
            
            sim = sG21(**p)
            sim.run()
            
            anl = aG21(sim)
            ax = anl.GlobalSignature(label='fail i=%i' % idnum)
            
            objects[idnum] = anl
            
        return ax, objects
        
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
      
    def ExtractData(self, pars, z=None, take_log=False, multiplier=1.):
        """
        Extract data for subsequent analysis.
        
        This means a few things:
         (1) Go retrieve data from native format without having to worry about
          all the indexing yourself.
         (2) [optionally] take the logarithm.
         (3) [optionally] apply multiplicative factors.
         (4) If bins are s
         
        Parameters
        ----------
        pars : list
            List of quantities to return. These can be parameters or the names
            of meta-data blobs.
         
        Returns
        -------
        Tuple with three entries:
         (i) Dictionary containing 1-D arrays of samples for each quantity.
         (ii) 
         
         
        """
        
        pars, take_log, multiplier, z = \
            self._listify_common_inputs(pars, take_log, multiplier, z)        
                        
        to_hist = []
        is_log = []
        for k, par in enumerate(pars):
        
            # If one of our free parameters, return right away
            if par in self.parameters:
                j = self.parameters.index(par)
                is_log.append(self.is_log[j])
                val = self.chain[:,j].copy()
        
                if self.is_log[j]:
                    val += np.log10(multiplier[k])
                else:
                    val *= multiplier[k]
        
                if take_log[k] and not self.is_log[j]:
                    to_hist.append(np.log10(val))
                else:
                    to_hist.append(val)
        
            else:
                val = self.extract_blob(par, z[k]).copy()
        
                val *= multiplier[k]
        
                if take_log[k]:
                    is_log.append(True)
                    to_hist.append(np.log10(val))
                else:
                    is_log.append(False)
                    to_hist.append(val)
        
        # Re-organize
        if len(np.unique(pars)) < len(pars):
            data = to_hist
        else:    
            data = {par:to_hist[i] for i, par in enumerate(pars)}
            is_log = {par:is_log[i] for i, par in enumerate(pars)}
                    
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
        if not inputs:
            return None
            
        if type(is_log) is dict:
            tmp = [is_log[par] for par in pars]    
            is_log = tmp
        
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
        
        # Loop over parameters
        #for i, p1 in enumerate(pars[-1::-1]):
        #    for j, p2 in enumerate(pars):
        #        
        #        # y-values first
        #        if type(inputs) is list:
        #            val = inputs[-1::-1][i]
        #        elif p1 in inputs:
        #            val = inputs[p1]
        #        else:
        #            val = None
        #            
        #        # Take log [optional]    
        #        if val is None:
        #            yin = None
        #        elif is_log[i] or take_log[i]:
        #            yin = np.log10(val * multiplier[-1::-1][i])                            
        #        else:
        #            yin = val * multiplier[-1::-1][i]                        
        #                                    
        #        # x-values next
        #        if type(inputs) is list:
        #            val = inputs[j]        
        #        elif p2 in inputs:
        #            val = inputs[p2]
        #        else:
        #            val = None
        #                        
        #        # Take log [optional]
        #        if val is None:
        #            xin = None  
        #        elif is_log[Nd-j-1] or take_log[Nd-j-1]:
        #            xin = np.log10(val * multiplier[-1::-1][Nd-j-1])
        #        else:
        #            xin = val * multiplier[-1::-1][Nd-j-1]
                
            
        return input_output
        
    def _listify_common_inputs(self, pars, take_log, multiplier, z):
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
            
        return pars, take_log, multiplier, z

    def PosteriorCDF(self, pars, bins=500, **kwargs):
        return self.PosteriorPDF(pars, bins=bins, cdf=True, **kwargs)
               
    def PosteriorPDF(self, pars, to_hist=None, is_log=None, z=None, ax=None, fig=1, 
        multiplier=1., nu=[0.95, 0.68], overplot_nu=False, density=True, cdf=False,
        color_by_like=False, filled=True, take_log=False, bins=20, skip=0, skim=1, 
        contour_method='raw', excluded=False, **kwargs):
        """
        Compute posterior PDF for supplied parameters. 
    
        If len(pars) == 2, plot 2-D posterior PDFs. If len(pars) == 1, plot
        1-D marginalized PDF.
    
        Parameters
        ----------
        pars : str, list
            Name of parameter or list of parameters to analyze.
        z : float
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
        
        kw = def_kwargs.copy()
        kw.update(kwargs)

        if 'labels' in kw:
            labels_tmp = default_labels.copy()
            labels_tmp.update(kwargs['labels'])
            labels = labels_tmp

        else:
            labels = default_labels
            
        pars, take_log, multiplier, z = \
            self._listify_common_inputs(pars, take_log, multiplier, z)

        # Only make a new plot window if there isn't already one
        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True

        # Grab all the data we need
        if (to_hist is None) or (is_log is None):
            to_hist, is_log = self.ExtractData(pars, z=z, take_log=take_log, 
                multiplier=multiplier)

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
        
        # Marginalized 1-D PDFs 
        if len(pars) == 1:
                        
            if type(to_hist) is dict:
                tohist = to_hist[pars[0]][skip:]
                b = binvec[pars[0]]
            elif type(to_hist) is list:
                tohist = to_hist[0][skip:]
                b = binvec[0]
            else:
                tohist = to_hist
                b = bins
            
            hist, bin_edges = \
                np.histogram(tohist, density=density, 
                    bins=b, weights=weights)

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
                tohist1 = to_hist[pars[0]][skip:]
                tohist2 = to_hist[pars[1]][skip:]
                b = [binvec[pars[0]], binvec[pars[1]]]
            else:
                tohist1 = to_hist[0][skip:]
                tohist2 = to_hist[1][skip:]
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
                #nu, levels = self.get_levels(hist, nu=nu)
                #
                #print nu, levels
                
                if contour_method == 'raw':
                    nu, levels = error_2D(None, None, hist, None, nu=nu, 
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
        self.set_axis_labels(ax, pars, is_log, take_log, labels)

        # Rotate ticks?

        pl.draw()
        
        return ax
      
    def FindContour(self, pars, z=None, take_log=False, multiplier=1., bins=20, 
        nu=0.68, contour_method='raw', weights=None):
        """
        Find a contour in the plane defined by `pars`, and return its x-y trajectory.
        
        Parameters
        ----------
        pars : list
            2-element list of parameters that defined plane of interest.
        
        """
        
        pars, z, take_log, multiplier = \
            self._listify_common_inputs(pars, z, take_log, multiplier)
        
        if type(nu) not in [list]:
            nu = [nu]
        
        to_hist, is_log = self.ExtractData(pars, z=z, take_log=take_log,
            multiplier=multiplier)
        
        binvec = self._set_bins(pars, to_hist, take_log, bins)
        
        # Compute 2-D histogram
        hist, xedges, yedges = \
            np.histogram2d(to_hist[pars[0]], to_hist[pars[1]], 
                bins=[binvec[pars[0]], binvec[pars[1]]], weights=weights)
        
        # Recover bin centers
        bc = []
        for i, edges in enumerate([xedges, yedges]):
            bc.append(rebin(edges))
        
        # ContourSet needs 2-D arrays
        x, y = np.meshgrid(bc[0], bc[1])
        
        # Create contour object
        c = cntr.Cntr(x, y, hist.T / hist.max())     
                
        # Find levels corresponding to nu-sigma inputs
        nu, levels = error_2D(to_hist[pars[0]], to_hist[pars[1]], 
            self.L / self.L.max(), bins=[binvec[pars[0]], binvec[pars[1]]], 
            nu=nu, method=contour_method)
        
        # Find x-y trajectory of contour
        all_contours = []
        for level in levels:
            nlist = c.trace(level, level, 0)
            segs = nlist[:len(nlist)/2]
            all_contours.append(segs) 
            
        return all_contours
        
    def get_2d_error(self, plane, nu=0.68):
        pass
        
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

        pars = [x, y]

        axes = []
        for par in pars:
            if par in self.parameters:
                axes.append(self.chain[:,self.parameters.index(par)])
            elif par in self.blob_names:
                axes.append(self.blobs[:,self.blob_redshifts.index(z),
                    self.blob_names.index(par)])
            elif par in self.derived_blob_names:
                axes.append(self.derived_blobs[:,self.blob_redshifts.index(z),
                    self.derived_blob_names.index(par)])        

        for i in range(2):
            if take_log[i]:
                axes[i] = np.log10(axes[i])

        xax, yax = axes

        if c in self.parameters:        
            zax = self.chain[:,self.parameters.index(c)].ravel()
        elif c in self.blob_names:   
            zax = self.blobs[:,self.blob_redshifts.index(z),
                self.blob_names.index(c)]
        elif c in self.derived_blob_names:   
            zax = self.derived_blobs[:,self.blob_redshifts.index(z),
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
            
        cb.set_label(logify_str(cblab))
            
        cb.update_ticks()
            
        pl.draw()
        
        return ax, scat, cb    
        
    def TrianglePlot(self, pars=None, z=None, panel_size=(0.5,0.5), 
        padding=(0,0), show_errors=False, take_log=False, multiplier=1,
        fig=1, inputs={}, tighten_up=0.0, ticks=5, bins=20, mp=None, skip=0, 
        skim=1, top=None, oned=True, twod=True, filled=True, box=None, rotate_x=45, 
        rotate_y=45, add_cov=False, label_panels='upper right', fix=True,
        skip_panels=[], **kwargs):
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
        z : int, float, str, list
            If plotting arbitrary meta-data blobs, must choose a redshift.
            Can be 'B', 'C', or 'D' to extract blobs at 21-cm turning points,
            or simply a number. If it's a list, it must have the same
            length as pars. This is how one can make a triangle plot 
            comparing the same quantities at different redshifts.
        input : dict
            Dictionary of parameter:value pairs representing the input
            values for all model parameters being fit. If supplied, lines
            will be drawn on each panel denoting these values.
        panel_size : list, tuple (2 elements)
            Multiplicative factor in (x, y) to be applied to the default 
            window size as defined in your matplotlibrc file. 
        skip : int
            Number of steps at beginning of chain to exclude. This is a nice
            way of doing a burn-in after the fact.
        skim : int
            Only take every skim'th step from the chain.
        oned : bool    
            Include the 1-D marginalized PDFs?
        filled : bool
            Use filled contours? If False, will use open contours instead.
        color_by_like : bool
            If True, set contour levels by confidence regions enclosing nu-%
            of the likelihood. Set parameter `nu` to modify these levels.
        nu : list
            List of levels, default is 1,2, and 3 sigma contours (i.e., 
            nu=[0.68, 0.95])
        rotate_x : bool
            Rotate xtick labels 90 degrees.
        add_cov : bool, list
            Overplot 1-sigma contours, computed using the covariance matrix.
            If it's a list, will assume it has [mu, cov] (i.e., not necessarily
            the extracted covariance matrix but the one used in the fit).
        skip_panels : list
            List of panel numbers to skip over.
        
        Returns
        -------
        ares.analysis.MultiPlot.MultiPanel instance.
        
        """    
                
        pars, take_log, multiplier, z = \
            self._listify_common_inputs(pars, take_log, multiplier, z)
                
        kw = def_kwargs.copy()
        kw.update(kwargs)
                        
        if 'labels' in kwargs:
            labels_tmp = default_labels.copy()
            labels_tmp.update(kwargs['labels'])
            labels = kwargs['labels']
            kw['labels'] = kwargs['labels']
        else:
            labels = default_labels
        
        to_hist, is_log = self.ExtractData(pars, z=z, take_log=take_log, 
            multiplier=multiplier)
            
        # Modify bins to account for log-taking, multipliers, etc.
        binvec = self._set_bins(pars, to_hist, take_log, bins)      
                            
        if type(binvec) is not list:
            bins = [binvec[par] for par in pars]      
        else:
            bins = binvec    
                                
        if oned:
            Nd = len(pars)
        else:
            Nd = len(pars) - 1
                           
        # Multipanel instance
        had_mp = True
        if mp is None:
            had_mp = False
            
            if 'diagonal' not in kwargs:
                kw['diagonal'] = 'lower'
            if 'dims' not in kwargs:    
                kw['dims'] = [Nd] * 2
            if 'keep_diagonal' not in kwargs:    
                kw['keep_diagonal'] = True
            else:
                oned = False

            mp = MultiPanel(padding=padding,
                panel_size=panel_size, fig=fig, top=top, **kw)
        
        for key in ['bottom', 'top', 'left', 'right', 'figsize', 'diagonal', 'dims', 'keep_diagonal']:
            if key in kw:
                del kw[key]
        
        # Apply multipliers etc. to inputs
        inputs = self._set_inputs(pars, inputs, is_log, take_log, multiplier)

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
                    ax = self.PosteriorPDF(p1, to_hist=tohist, is_log=is_log,
                        ax=mp.grid[k], take_log=take_log[-1::-1][i], z=z[-1::-1][i],
                        multiplier=[multiplier[-1::-1][i]], 
                        bins=[bins[-1::-1][i]], skip=skip, skim=skim, 
                        **kw)

                    if add_cov:
                        if type(add_cov) is list:
                            self._PosteriorIdealized(ax=mp.grid[k], 
                                mu=add_cov[k], cov=add_cov[k])                            
                        else:
                            self._PosteriorIdealized(pars=p1, ax=mp.grid[k], 
                                z=z[-1::-1][i])

                    # Stick this stuff in fix_ticks?
                    if col != 0:
                        mp.grid[k].set_ylabel('')
                    if row != 0:
                        mp.grid[k].set_xlabel('')

                    if show_errors:
                        mu, err = self.get_1d_error(p1)
                                                 
                        mp.grid[k].set_title(err_str(p1, mu, err, 
                            self.is_log[i], labels), va='bottom', fontsize=18) 
                     
                    if not inputs:
                        continue
                        
                    if xin is not None:
                        mp.grid[k].plot([xin]*2, [0, 1.05], 
                            color='k', ls=':', lw=2, zorder=20)
                            
                    continue

                red = [z[j], z[-1::-1][i]]

                # If not oned, may end up with some x vs. x plots if we're not careful
                if p1 == p2 and (red[0] == red[1]):
                    continue

                try:
                    tohist = [to_hist[j], to_hist[-1::-1][i]]
                except KeyError:
                    tohist = [to_hist[p2], to_hist[p1]]
                
                # 2-D PDFs elsewhere
                ax = self.PosteriorPDF([p2, p1], to_hist=tohist, is_log=is_log,
                    ax=mp.grid[k], z=red, take_log=[take_log[j], take_log[-1::-1][i]],
                    multiplier=[multiplier[j], multiplier[-1::-1][i]], 
                    bins=[bins[j], bins[-1::-1][i]], filled=filled, 
                    skip=skip, **kw)
                
                if add_cov:
                    if type(add_cov) is list:
                        pass
                        self._PosteriorIdealized(ax=mp.grid[k], 
                            mu=add_cov[k], cov=add_cov[k])
                    else:
                        self._PosteriorIdealized(pars=[p2, p1], 
                            ax=mp.grid[k], z=red)

                if row != 0:
                    mp.grid[k].set_xlabel('')
                if col != 0:
                    mp.grid[k].set_ylabel('')

                # Input values
                if not inputs:
                    continue
                                                                    
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
            mp.fix_ticks(oned=oned, N=ticks, rotate_x=rotate_x, rotate_y=rotate_y)
        
        if not had_mp:
            mp.rescale_axes(tighten_up=tighten_up)
    
        if label_panels is not None and (not had_mp):
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
        
    def RedshiftEvolution(self, blob, ax=None, redshifts=None, fig=1,
        nu=0.68, take_log=False, bins=20, label=None,
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
                    self.get_1d_error(blob, z=z, nu=nu, take_log=take_log,
                    bins=bins, limit=limit)
            except TypeError:
                continue
            
            if value is None:
                continue    
            
            # Error on redshift
            if type(z) == str and not plot_bands:
                if blob == 'dTb':
                    mu_z, (z_err1, z_err2) = \
                        self.get_1d_error('nu', z=z, nu=nu, bins=bins)
                else:
                    mu_z, (z_err1, z_err2) = \
                        self.get_1d_error('z', z=z, nu=nu, bins=bins)
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
        
    def CovarianceMatrix(self, pars, z=None):
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
        
        if z is None:
            z = [None] * len(pars)
        elif type(z) is not list:
            z = [z]
        
        masks = []
        blob_vec = []
        for i in range(len(pars)):
            blob = self.extract_blob(pars[i], z=z[i])
            masks.append(blob.mask)
            blob_vec.append(blob)    
        
        master_mask = np.zeros_like(masks[0])
        for mask in masks:
            master_mask += mask
        
        master_mask[master_mask > 0] = 1
            
        blob_vec_mast = self.blob_vec_mast = np.ma.array(blob_vec, 
            mask=[master_mask] * len(blob_vec))
        
        blobs_compr = np.array([vec.compressed() for vec in blob_vec_mast])

        mu = np.mean(blobs_compr, axis=1)
        cov = np.cov(blobs_compr)

        return mu, cov     

    def _PosteriorIdealized(self, ax=None, pars=None, mu=None, cov=None, z=None, 
        nu=[0.68, 0.95]):
        """
        Draw posterior distribution using covariance matrix.
        """

        # Handle 1-D PDFs separately
        if pars is not None:
            if type(pars) is str:
            
                # Set bounds using current axes limits
                mi, ma = ax.get_xlim()
                x = np.linspace(mi, ma, 100)
                
                # Extract the data, and calculate the mean and variance
                blob = self.extract_blob(pars, z=z)
                mu = np.mean(blob)
                cov = np.std(blob)**2
                
                # Plot the PDF!
                ax.plot(x, np.exp(-(x - mu)**2 / 2. / cov), color='k', ls='-')
                
                pl.draw()
                return

        if pars is not None:
            if len(pars) > 2: 
                raise ValueError('Only meant for 2x2 chunks of covariance.')

        # Compute the covariance matrix
        if cov is None:
            mu, cov = self.CovarianceMatrix(pars, z)
        
        var = np.sqrt(np.diag(cov))

        # When integrating over constant likelihood contours, must hold
        # the eccentricity constant
        eccen = np.sqrt(1. - (min(var) / max(var))**2)    

        # For elliptical integration paths, must make sure the semi-major
        # axis is longer than the semi-minor axis
        if var[1] > var[0]:
            new_mu = mu[-1::-1]

            diag = np.diag(cov)[-1::-1]
            new_cov = np.zeros([len(diag)] * 2)
            new_cov[0,0] = diag[0]
            new_cov[1,1] = diag[1]
        else:
            new_mu = mu
            
            diag = np.diag(cov)
            new_cov = np.zeros([len(diag)] * 2)
            new_cov[0,0] = diag[0]
            new_cov[1,1] = diag[1]
            
        # Convenience variable for variances    
        new_var = diag
        
        # Likelihood functions                      
        likelihood = lambda yy, xx: GaussND(np.array([xx, yy]), mu, cov) 
            
        # For integration, consider 2-D Gaussian centered at zero for simplicity
        _like = lambda yy, xx: GaussND(np.array([xx, yy]), 
            np.array([0.]*2), new_cov)
            
        xmin, xmax = np.array(ax.get_xlim())
        ymin, ymax = np.array(ax.get_ylim())
        
        x = np.linspace(xmin, xmax, 100)                    
        y = np.linspace(ymin, ymax, 100)   

        # Construct likelihood surface                        
        surf = np.zeros([x.size, y.size])
        for j, xx in enumerate(x):
            for k, yy in enumerate(y):
                surf[j,k] = likelihood(yy, xx)
            
        # Find levels corresponding to 1 and 2 sigma
        levels = []
        for k, level in enumerate(nu):

            # Minimize difference between area(DX) and requested level
            #to_min = lambda DX: \
            #    np.abs(self._elliptical_integral(DX, eccen, _like) - level)
            #
            #guess = np.array([np.sqrt(new_var[0])])
        
            # Solve for the standard deviation in each dimension
            #dx = fmin(to_min, guess, disp=False, ftol=1e-4)[0]
            #dy = dx * np.sqrt(1. - eccen**2)
                            
            #contour = _like(dy, dx) 
            contour_1s = _like(np.sqrt(new_var[1]), np.sqrt(new_var[0]))
                        
            #contour_1s_2 = self._elliptical_integral(np.sqrt(new_var[0]),
            #    eccen, _like)  
            
            #print level, eccen, np.sqrt(new_var[0]), contour_1s_2
                        
            #print level, dx, dy, np.sqrt(new_var), self._elliptical_integral(dx, eccen, _like)
                    
            # Save and move on
            levels.append(contour_1s)

        ax.contour(x, y, surf.T, levels=levels, linestyles=['-', '--'], 
            colors='k')

        pl.draw()

    def _elliptical_integral(self, dx, ecc, like):
    
        # Integration is over an ellipse!
        x1 = - dx
        x2 = + dx

        # Figure out elliptical surface
        dy = dx * np.sqrt(1. - ecc**2)

        # Arcs in +/- y
        y1 = lambda x: - dy * np.sqrt(1. - (x / dx)**2)
        y2 = lambda x: + dy * np.sqrt(1. - (x / dx)**2)

        fig = pl.figure(10); ax = fig.add_subplot(111)

        xm = np.linspace(x1, x2)
        xp = np.linspace(x1, x2)

        ym = map(y1, xm)
        yp = map(y2, xp)

        pl.plot(xm, ym, color='b')
        pl.plot(xp, yp, color='r')
        
        raw_input('<enter>')
        
        pl.close()
        
        return dblquad(like, x1, x2, y1, y2, epsrel=1e-2, epsabs=1e-4)[0]            
        
    def CorrelationMatrix(self, pars, z=None, fig=1, ax=None):
        """
        Plot correlation matrix.
        """
    
        mu, cov = self.CovarianceMatrix(pars, z=z)
    
        corr = correlation_matrix(cov)
    
        if ax is None:
            fig = pl.figure(fig); ax = fig.add_subplot(111)
    
        cax = ax.imshow(corr, interpolation='none', cmap='RdBu_r', 
            vmin=-1, vmax=1)
        cb = pl.colorbar(cax)
    
        return corr
    
    def add_boxes(self, ax=None, val=None, width=None, **kwargs):
        """
        Add boxes to 2-D PDFs.
        
        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot instance
        
        val : int, float
            Center of box (probably maximum likelihood value)
        width : int, float
            Size of box (above/below maximum likelihood value)
        
        """
        
        if width is None:
            return
        
        iwidth = 1. / width
            
        # Vertical lines
        ax.plot(np.log10([val[0] * iwidth, val[0] * width]), 
            np.log10([val[1] * width, val[1] * width]), **kwargs)
        ax.plot(np.log10([val[0] * iwidth, val[0] * width]), 
            np.log10([val[1] * iwidth, val[1] * iwidth]), **kwargs)
            
        # Horizontal lines
        ax.plot(np.log10([val[0] * iwidth, val[0] * iwidth]), 
            np.log10([val[1] * iwidth, val[1] * width]), **kwargs)
        ax.plot(np.log10([val[0] * width, val[0] * width]), 
            np.log10([val[1] * iwidth, val[1] * width]), **kwargs)

    def extract_blob(self, name, z):
        """
        Extract a 1-D array of values for a given quantity at a given redshift.
        """

        # Otherwise, we've got a meta-data blob
        try:
            i = self.blob_redshifts.index(z)
        except ValueError:
            
            ztmp = []
            for redshift in self.blob_redshifts_float:
                if redshift is None:
                    ztmp.append(None)
                else:
                    ztmp.append(round(redshift, 1))    
                
            i = ztmp.index(round(z, 1))
    
        if name in self.blob_names:
            j = self.blob_names.index(name)            
            return self.blobs[:,i,j]
        else:
            j = self.derived_blob_names.index(name)
            return self.derived_blobs[:,i,j]
    
    def max_likelihood_parameters(self):
        """
        Return parameter values at maximum likelihood point.
        """
    
        iML = np.argmax(self.logL)
    
        p = {}
        for i, par in enumerate(self.parameters):
            if self.is_log[i]:
                p[par] = 10**self.chain[iML,i]
            else:
                p[par] = self.chain[iML,i]
    
        return p
        
    def save(self, pars, z=None, fmt='hdf5', clobber=False):
        """
        Output a particular quantity to its own file.
        
        Parameters
        ----------
        pars : str, list, tuple
            Name of parameter (or list of parameters) or blob(s) to extract.
        fmt : str
            Options: 'hdf5' or 'pkl'

        .. note :: Cannot handle multiple fields of the same name (but 
            different redshifts) yet.
            
        .. note :: Will write to current working directory.
        
        """
        
        if len(np.unique(pars)) < len(pars):
            raise NotImplemented('Cannot handle multiple parameters of same name!')
        
        if type(pars) not in [list, tuple]:
            pars = [pars]
        if type(z) not in [list, tuple]:
            z = [z] * len(pars)

        # Output to HDF5
        if fmt == 'hdf5':

            # Loop over parameters and save to disk
            for i, par in enumerate(pars):
                fn = '%s.%s.%s' % (self.fn, par, fmt)

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
                        raise IOError('%s exists! As does this dataset.' % fn)
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
        
    def set_axis_labels(self, ax, pars, is_log, take_log=False, labels=None):
        """
        Make nice axis labels.
        """
        
        pars, take_log, multiplier, z = \
            self._listify_common_inputs(pars, take_log, 1.0, z=None)
        
        if type(is_log) != dict:
            tmp = {par:is_log[i] for i, par in enumerate(pars)}
            is_log = tmp
        if type(take_log) != dict:
            tmp = {par:take_log[i] for i, par in enumerate(pars)}
            take_log = tmp        
            
        # [optionally] modify default labels
        if labels is None:
            labels = default_labels
        else:
            labels_tmp = default_labels.copy()
            labels_tmp.update(labels)
            labels = labels_tmp

        p = []
        sup = []
        for par in pars:

            if type(par) is int:
                sup.append(None)
                p.append(par)
                continue

            # If we want special labels for population specific parameters
            #if par in labels:
            sup.append(None)
            p.append(par)
            continue    

            #p.append(par)
            #
            #m = re.search(r"\{([0-9])\}", par)
            #
            #if m is None:
            #    sup.append(None)
            #    p.append(par)
            #    continue
            #
            ## Split parameter prefix from population ID in braces
            #p.append(par.split(m.group(0))[0])
            #sup.append(int(m.group(1)))
        
        del pars
        pars = p
    
        log_it = is_log[pars[0]] or take_log[pars[0]]
            
        ax.set_xlabel(make_label(pars[0], log_it, labels))
    
        if len(pars) == 1:
            ax.set_ylabel('PDF')
    
            pl.draw()
            return
    
        log_it = is_log[pars[1]] or take_log[pars[1]]
        ax.set_ylabel(make_label(pars[1], log_it, labels))

        pl.draw()
                
    def confidence_regions(self, L, nu=[0.95, 0.68]):
        """
        Integrate outward at "constant water level" to determine proper
        2-D marginalized confidence regions.
    
        ..note:: This is fairly crude -- the "coarse-ness" of the resulting
            PDFs will depend a lot on the binning.
    
        Parameters
        ----------
        L : np.ndarray
            Grid of likelihoods.
        nu : float, list
            Confidence intervals of interest.
    
        Returns
        -------
        List of contour values (relative to maximum likelihood) corresponding 
        to the confidence region bounds specified in the "nu" parameter, 
        in order of decreasing nu.
        """
    
        if type(nu) in [int, float]:
            nu = np.array([nu])
    
        # Put nu-values in ascending order
        if not np.all(np.diff(nu) > 0):
            nu = nu[-1::-1]
    
        peak = float(L.max())
        tot = float(L.sum())
    
        # Counts per bin in descending order
        Ldesc = np.sort(L.ravel())[-1::-1]
    
        j = 0  # corresponds to whatever contour we're on
    
        Lprev = 1.0
        Lencl_prev = 0.0
        contours = [1.0]
        for i in range(1, Ldesc.size):
    
            # How much area (fractional) is enclosed within the current contour?
            Lencl_now = L[L >= Ldesc[i]].sum() / tot
    
            Lnow = Ldesc[i]
    
            # Haven't hit next contour yet
            if Lencl_now < nu[j]:
                pass
    
            # Just passed a contour
            else:
                # Interpolate to find contour more precisely
                Linterp = np.interp(nu[j], [Lencl_prev, Lencl_now],
                    [Ldesc[i-1], Ldesc[i]])
                # Save relative to peak
                contours.append(Linterp / peak)
    
                j += 1
    
            Lprev = Lnow
            Lencl_prev = Lencl_now
    
            if j == len(nu):
                break
    
        # Return values that match up to inputs    
        return nu[-1::-1], contours[-1::-1]
    
    def errors_to_latex(self, pars, nu=0.68, in_units=None, out_units=None):
        """
        Output maximum-likelihood values and nu-sigma errors ~nicely.
        """
                
        if type(nu) != list:
            nu = [nu]
            
        hdr = 'parameter    '
        for conf in nu:
            hdr += '%.1f' % (conf * 100)
            hdr += '%    '
        
        print hdr
        print '-' * len(hdr)    
        
        for i, par in enumerate(pars):
            
            s = str(par)
            
            for j, conf in enumerate(nu):
                
                
                mu, sigma = \
                    map(np.array, self.get_1d_error(par, bins=100, nu=conf))

                if in_units and out_units != None:
                    mu, sigma = self.convert_units(mu, sigma,
                        in_units=in_units, out_units=out_units)

                s += r" & $%5.3g_{-%5.3g}^{+%5.3g}$   " % (mu, sigma[0], sigma[1])
        
            s += '\\\\'
            
            print s
    
    def convert_units(self, mu, sigma, in_units, out_units):
        """
        Convert units on common parameters of interest.
        
        So far, just equipped to handle frequency -> redshift and Kelvin
        to milli-Kelvin conversions. 
        
        Parameters
        ----------
        mu : float
            Maximum likelihood value of some parameter.
        sigma : np.ndarray
            Two-element array containing asymmetric error bar.
        in_units : str
            Units of input mu and sigma values.
        out_units : str
            Desired units for output.
        
        Options
        -------
        in_units and out_units can be one of :
        
            MHz
            redshift
            K
            mK
            
        Returns
        -------
        Tuple, (mu, sigma). Remember that sigma is itself a two-element array.
            
        """
        
        if in_units == 'MHz' and out_units == 'redshift':
            new_mu = nu_0_mhz / mu - 1.
            new_sigma = abs(new_mu - (nu_0_mhz / (mu + sigma[1]) - 1.)), \
                abs(new_mu - (nu_0_mhz / (mu - sigma[0]) - 1.))
                        
        elif in_units == 'redshift' and out_units == 'MHz':
            new_mu = nu_0_mhz / (1. + mu)
            new_sigma = abs(new_mu - (nu_0_mhz / (1. + mu - sigma[0]))), \
                        abs(new_mu - (nu_0_mhz / (1. + mu - sigma[1])))
        elif in_units == 'K' and out_units == 'mK':
            new_mu = mu * 1e3
            new_sigma = np.array(sigma) * 1e3
        elif in_units == 'mK' and out_units == 'K':
            new_mu = mu / 1e3
            new_sigma = np.array(sigma) / 1e3
        else:
            raise ValueError('Unrecognized unit combination')
        
        return new_mu, new_sigma
    