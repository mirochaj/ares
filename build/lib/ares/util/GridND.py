"""

GridND.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Dec  3 11:08:54 2012

Description: Keeps track of all axes of an N-D dataset, and provides 
routines for slicing through it and applying functions to it. 

"""

import numpy as np
import os, itertools, copy
from collections import defaultdict

try: 
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1
    
try:
    import h5py
except ImportError:
    pass    
    
clobber_msg = "Run with clobber=True to overwrite."

class GridAxis(object):
    def __init__(self, name=None, num=None, values=None):
        """
        Initialize GridAxis object.
        
        Parameters
        ----------
        
        
        """
        self.num = num
        self.name = name
        self.values = values
        self.size = len(self.values)
        self.limits = (values.min(), values.max())
        self.min, self.max = self.limits
        self.range = self.max - self.min
        self.dx = np.diff(values)[0]
        
        split_name = self.name.split('_')
        
        if split_name[0] == 'spectrum':
            self.basename = split_name[1]
        else:
            self.basename = split_name[0]
            if len(split_name) == 1:
                self.id = ''    
            else:
                if split_name[-1].isdigit():
                    self.id = ',{!s}'.format(split_name[-1])
                else:
                    self.id = ''
                    self.basename = self.name

    def locate(self, val, tol=0.0):
        """ Return index of axis value closest to supplied value. """
        
        if tol == 0:
            if val in self.values:
                return np.argmin(np.abs(val - self.values))
            else:
                return None
        else:
            
            if val in self.values:
                return np.argmin(np.abs(val - self.values))
            elif val == 0:
                diff = np.abs((val - self.values))
            else:    
                diff = np.abs((val - self.values) / val)

            amin = np.argmin(diff)

            if diff[amin] <= tol:
                return amin
            else:
                return None
            

class GridND(defaultdict):
    def __init__(self, fn=None, verbose=False):
        """
        Initialize ModelGrid object.
        
        Parameters
        ----------
        fn : string, optional
            Filename of pre-existing ModelGrid (in hdf5 or pickle format).
        verbose : bool
            Print stuff to the screen without reservation.
        
        """
        self.fn = fn
        self.verbose = verbose
        
    def __missing__(self, key):
        return None        
        
    @property
    def structured(self):
        if not hasattr(self, '_structured'):
            if hasattr(self, '_axes'):
                if self._axes:
                    self._structured = True
                else:
                    self._structured = False
            else:
                self._structured = False

        return self._structured

    @structured.setter
    def structured(self, value):
        self._structured = value

    @property
    def axes(self):
        """ List of all GridAxis instances. """
        if not hasattr(self, '_axes'):
            self._axes = [None] * self.Nd
            for key in self:
                if isinstance(self[key], GridAxis):
                    self._axes[self[key].num] = self[key]

        return self._axes    

    @property    
    def axes_names(self):    
        """ Names of all axes. """
        
        if not hasattr(self, '_axes_names'):
            self._axes_names = [None] * self.Nd
            for key in self:
                if isinstance(self[key], GridAxis):
                    self._axes_names[self[key].num] = self[key].name
        
        return self._axes_names
        
    @axes_names.setter
    def axes_names(self, value):
        self._axes_names = value    
        
    @property
    def dims(self):
        """ Dimensions of grid. """
        if not hasattr(self, '_dims'):
            self._dims = [None] * self.Nd
            for axis in self.axes:
                self._dims[axis.num] = axis.size
        
        return self._dims
        
    @property
    def shape(self):
        """ Equivalent to self.dims. """
        return self.dims
        
    @property
    def size(self):    
        """ Total number of elements in grid. """
        
        if self.structured and not hasattr(self, '_size'):
            self._size = int(np.prod(self.dims))
        elif not self.structured:
            self._size = len(self.all_kwargs)
        
        return self._size
        
    @property
    def Nd(self):
        """ Number of dimensions in grid. """
        if not hasattr(self, '_Nd'):
            self._Nd = 0
            for key in self.keys():
                if isinstance(self[key], GridAxis):
                    self._Nd += 1

        return self._Nd
        
    @Nd.setter
    def Nd(self, value):
        self._Nd = value
        
    @property
    def coords(self):
        """
        List of Nd-element tuples corresponding to all_kwargs parameter sets.
        """    
        
        if not hasattr(self, '_coords'):
            self._coords = []
            for kwargs in self.all_kwargs:
                self._coords.append(self.locate_entry(kwargs))
                
        return self._coords
        
    def new_axis(self, name, values):
        """
        Create a new axis, perhaps a product of two pre-existing axes.
        """    
        
        if name not in self:
            self[name] = GridAxis(name=name, num=np.nan, values=values)
        else:
            print('{!s} exists!'.format(name))
            
    def new_field(self, name, clobber=False):
        if name in self.keys():
            if clobber:
                del self[name]
            else:
                print('{0!s} already exists! {1!s}'.format(name, clobber_msg))
                return
            
        self[name] = np.zeros(self.dims)        
                   
    def axis(self, ax):
        """
        Return instance of axis via its name or index number.
        """
        if type(ax) is int:
            return self[self.axisname(ax)]
        else:
            return self[ax]
    
    def axisname(self, ax):
        """
        Return name of axis corresponding to given index.
        """
        for axis in self.axes:
            if axis.num == ax:
                return axis.name

    def axisnum(self, ax):
        """
        Return index of axis corresponding to given name.
        """
        return self[ax].num

    def locate_entry(self, model, tol=0):
        """
        Find location (tuple of indices) of given model (dictionary) in an array
        of shape self.data.shape.
        """

        ijkl = [Ellipsis] * self.Nd
        for axis in self.axes:
            ijkl[axis.num] = axis.locate(model[axis.name], tol=tol)

        return tuple(ijkl)

    def locate_kwargs(self, model):
        """
        Find location of given model (dictionary) in self.all_kwargs.
        """
        
        loc = self.locate_entry(model)
        return self.coords.index(loc)
        
    @property
    def higher_dimensions(self):
        if not hasattr(self, '_higher_dimensions'):
            return None
        
        return self._higher_dimensions
        
    @higher_dimensions.setter
    def higher_dimensions(self, value):
        self._higher_dimensions = value
        
    def meshgrid(self, axis):
        """
        Effectively do a numpy.meshgrid, but for any single parameter.  
        
        The axes of our model grid are 1D, but often we'll perform 
        operations on an N-D grid, so mapping the 1D axes values to N-D 
        makes grabbing their values more convenient.
        
        Parameters
        ----------
        axis : str
            Name of axis to meshgrid.
            
        Is there a pre-existing vectorized method of doing this?
        
        Returns
        -------
        "Meshed," or N-D, version of input axis.

        """
        
        axmesh = np.zeros(self.shape)
        for i, coords in enumerate(self.coords):
            axmesh[coords] = self.all_kwargs[i][axis]
        
        return axmesh
        
    def reshape(self, data):
        """
        Take 1D data (like self.all_kwargs) and shape to grid.
        """    
        
        new_data = np.zeros(self.dims)
        for i, coords in enumerate(self.coords):
            new_data[coords] = data[i]                
        
        return new_data
        
    def slice(self, data, slices):
        """
        Slice through N dimensions.  Must provide dictionary
        describing higher dimensions we're slicing through ('slices').
        """    
               
        # First, determine which index we'll slice through in each axis       
        slc = [Ellipsis] * self.Nd
        for axis in self.axes:
            if axis.name not in slices.keys():
                continue

            slc[axis.num] = axis.locate(slices[axis.name])
            
            if self.verbose:
                print('Slice through {0!s} = {1:g}'.format(axis.name,\
                    axis.values[slc[axis.num]]))
        
        # Retrieve non-sliced axes information
        xyax = []
        for axis in self.axes:
            if axis.name in slices.keys():
                continue
                
            xyax.append(axis)
                  
        return xyax, data[tuple(slc)].squeeze()
        
    def prior(self, data, priors):
        """
        Compute prior over entire grid.
        """
        
        if self.verbose:
            print("Computing prior in {}-D...".format(self.Nd))
        
        if hasattr(self, 'NDprior'):
            return self.NDprior
            
        self.NDprior = np.ones_like(data)
        if priors is not None:
            for p in priors:
                axm = self.meshgrid(p, data)
                self.NDprior *= priors[p](axm)
                
        return self.NDprior
        
    def marginalize(self, space, likelihood, priors=None):
        """
        Marginalize over all axes except those in list 'space'.
        Currently only supporting 1D priors.
        
        Parameters
        ----------
        space : list
            List of axes (provide their names as strings) to include for the
            resulting N-D PDF. That is, marginalize over all axes *NOT* 
            supplied via this argument.
        likelihood : np.ndarray
            Array, same shape as grid, denoting likelihood of every model.
        priors : NOT IMPLEMENTED
        
        """

        if type(space) is not list:
            space = [space]

        # Axes to marginalize over
        axes = copy.copy(self.axes_names)
        for axis in space:
            axes.remove(axis)
        
        # Setup N-D prior
        #pdf = data.copy() * self.prior(data, priors)
        
        pdf = likelihood.copy()
        
        # List of axes numbers
        axes_num = [axis.num for axis in self.axes]
        
        shape, maxes = self._marginal_pdf_info(axes)
        for i, axis in enumerate(axes):
            num = self.axis(axis).num
            pdf = np.trapz(pdf, axis=axes_num.index(num))
            axes_num.pop(axes_num.index(num))
            
        xyax = [self.axis(ax) for ax in space]                        
        return xyax, pdf / np.sum(pdf)
        
    def marginal_properties(self, x, L):
        """
        Compute mean and standard deviation of 1-D marginalized PDF.
        """
        
        i_mode = np.argmin(np.abs(L - np.max(L)))
        mode = x[i_mode]
    
        # Allow for asymmetric error
        
    
        return mode
        
    def _marginal_pdf_info(self, axes):
        """
        Compute shape of PDF array after marginalizing over axes, and
        also return list of axes names themselves (in order).
        """
        
        shape, names = [], []
        for axis in self.axes:
            if axis.name not in axes:
                continue
            shape.append(axis.size)
            names.append(axis.name)
            
        return shape, names
        
    def likelihood_filter(self, data, Lth=0.05):
        """
        Return list of parameter sets that have an unmarginalized 
        likelihood greater than some threshold.
        """    
        
        print('Filtering by likelihood threshold...')
        
        Lmax = np.max(data)
        filtered = []
        for i, coords in enumerate(self.coords):
            if data[coords] < (Lmax * Lth):
                continue
            
            filtered.append(self.all_kwargs[i].copy())
        
        return filtered
        
    def compute(self, func, name=None, **kwargs):
        """
        Apply given function to single model and store it in its rightful
        place in self[name] or return.
        """
        
        if not name:
            return func(**kwargs)
        
        if not (name in self):
            self.new_field(name)
        
        loc = self.locate_entry(kwargs)
        pars = np.array([kwargs[ax] for ax in self.axes_names])        
        self[name][loc] = func(pars)
                    
    def update(self, name, loc, value):
        """
        Update dataset in self.
        """        
        
        self[name][loc] = value
        
    def add_dataset(self, name, data, clobber=False):
        """
        Store data in self.
        """    
        
        if name in self:
            if clobber:
                del self[name]
            else:
                if self.verbose:
                    print('{0!s} already exists! {1!s}'.format(name,\
                        clobber_msg))
                return
        
        self[name] = data    
    
    def build(self, **kwargs):
        """
        Build N-dimensional parameter space via inner-product.
        """
        
        self.structured = True
        
        names = []
        self._Nd = 0
        self.kwargs = kwargs
        self.const_kwargs = {}
        
        # Figure out how many axes we have, and store other (constant) kwargs.
        for param in kwargs.keys():
            if hasattr(kwargs[param], 'dot'): # Will correctly identify arrays
                if len(kwargs[param]) > 1 and type(kwargs[param]) is np.ndarray:
                    names.append(param)
                    self._Nd += 1
                    continue
                else:
                    kwargs[param] = kwargs[param][0]                        
            
            self.const_kwargs[param] = kwargs[param]
                       
        # Figure out all combinations of parameters
        pranges = [None] * self.Nd
        for i in range(self.Nd):
            if names[i] in self:
                ax = self.axis(names[i])
                pranges[ax.num] = ax.values
                continue
            
            pranges[i] = kwargs[names[i]]
            self[names[i]] = GridAxis(names[i], i, kwargs[names[i]])

        combos = itertools.product(*pranges)
        
        # Create list of dictionaries - each one containing params for unique model
        self.all_kwargs = []
        for combo in combos:
            tmp = self.const_kwargs.copy()
            for i, param in enumerate(self.axes_names):
                tmp[param] = combo[i]
                
            self.all_kwargs.append(tmp)   
                            
    def recover(self, all_kwargs):
        """
        From list of models (each a dictionary), recover the original
        kwargs, and build model grid.
        
        Parameters
        ----------
        all_kwargs : list
            Each of element is a dictionary, describing a single model.
            
        """
        
        # Determine all parameters used in this set of models
        allnames = list(all_kwargs[0].keys())
        
        # Figure out all values of all parameters
        allvals = []
        for kwargs in all_kwargs:
            allvals.append(list(kwargs.values()))
                    
        # Figure out which parameters vary from model to model            
        axes_values = []
        const_kwargs = {}
        for i, par in enumerate(allnames):
            parvals = list(zip(*allvals))[i] 
            
            if not np.all(np.diff(parvals) == 0):
                unique_vals = np.unique(parvals)
                axes_values.append(np.array(np.sort(unique_vals)))
            else:
                if par not in const_kwargs:
                    const_kwargs[par] = all_kwargs[0][par]
                    
        axes_names = []
        for name in allnames:
            if name in const_kwargs:
                continue
                
            axes_names.append(name)            
                    
        kwargs = const_kwargs.copy()
        for i, name in enumerate(axes_names):
            kwargs[name] = axes_values[i]
                        
        self.build(**kwargs)           
        
