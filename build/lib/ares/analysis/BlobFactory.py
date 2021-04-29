"""

BlobFactory.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Dec 11 14:24:53 PST 2015

Description: 

"""

import os
import re
import glob
import numpy as np
from inspect import ismethod
from types import FunctionType
from scipy.interpolate import RectBivariateSpline, interp1d
from ..util.Pickling import read_pickle_file, write_pickle_file
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str
    
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
        
def get_k(s):
    m = re.search(r"\[(\d+(\.\d*)?)\]", s)
    return int(m.group(1))
    
def parse_attribute(blob_name, obj_base):
    """
    Find the attribute nested somewhere in an object that we need to compute
    the value of blob `blob_name`.
    
    ..note:: This is the only place I ever use eval (I think). It's because
        using __getattribute__ conflicts with the __getattr__ method used
        in analysis.Global21cm.
        
    Parameters
    ----------
    blob_name : str
        Full name of blob (might be nested)
    obj_base : instance
        Usually just an instance of BlobFactory, which is inherited by
        simulation classes, so think of this as an instance of an
        ares.simulation class.
        
    """
                
    # Check for decimals
    decimals = []
    for i in range(1, len(blob_name) - 1):
        if blob_name[i-1].isdigit() and blob_name[i] == '.' \
           and blob_name[i+1].isdigit():
            decimals.append(i)

    marker = 'x&x&'

    s = ''
    for i, char in enumerate(blob_name):
        if i in decimals:
            s += marker
        else:
            s += blob_name[i]
                
    attr_split = []
    for element in s.split('.'):
        attr_split.append(element.replace(marker, '.'))
              
    if len(attr_split) == 1: 
        s = attr_split[0]
        return eval('obj_base.{!s}'.format(s))

    # Nested attribute
    blob_attr = None
    obj_list = [obj_base]
    for i in range(len(attr_split)):
        
        # One particular chunk of the attribute name
        s = attr_split[i]
        
        new_obj = eval('obj_base.{!s}'.format(s))
        obj_list.append(new_obj)

        obj_base = obj_list[-1]

        # Need to stop once we see parentheses
        
    #if blob is None:
    #    blob = new_obj
        
    return new_obj

class BlobFactory(object):
    """
    This class must be inherited by another class, which need only have the
    ``pf`` attribute.
    
    The three most (only) important parameters are:
        blob_names
        blob_ivars
        blob_funcs
        
    """
    
    #def __del__(self):
    #    print("Killing blobs! Processor={}".format(rank))

    def _parse_blobs(self):
        
        hdf5_situation = False                                        
        try:
            names = self.pf['blob_names']
        except KeyError:
            names = None
        except TypeError:
            hdf5_situation = True
            f = h5py.File('{!s}.hdf5'.format(self.prefix), 'r')
            names = list(f['blobs'].keys())
            f.close()

        if names is None:
            self._blob_names = self._blob_ivars = None
            self._blob_dims = self._blob_nd = None
            self._blob_funcs = None
            self._blob_kwargs = None
            return None
        else:
            # Otherwise, figure out how many different kinds (shapes) of
            # blobs we have
            assert type(names) in [list, tuple], \
                "Must supply blob_names as list or tuple!"

            if hdf5_situation:
                f = h5py.File('{!s}.hdf5'.format(self.prefix), 'r')

                _blob_ivars = []
                _blob_ivarn = []
                _blob_names = names
                for name in names:
                    ivar = f['blobs'][name].attrs.get('ivar')

                    if ivar is None:
                        _blob_ivars.append(ivar)
                    else:
                        _blob_ivarn.append('unknown')
                        _blob_ivars.append(ivar.squeeze())

                f.close()

                # Re-organize...maybe eventually
                self._blob_ivars = _blob_ivars
                self._blob_ivarn = _blob_ivarn
                self._blob_names = _blob_names
                
            elif 'blob_ivars' in self.pf:
                self._blob_names = names
                if self.pf['blob_ivars'] is None:
                    self._blob_ivars = [None] * len(names)
                else:
                    self._blob_ivarn = []
                    self._blob_ivars = []
                    raw = self.pf['blob_ivars']
                    
                    # k corresponds to ivar group
                    for k, element in enumerate(raw):
                        
                        if element is None:
                            self._blob_ivarn.append(None)
                            self._blob_ivars.append(None)
                            continue
                            
                        # Must make list because could be multi-dimensional
                        # blob, i.e., just appending to blob_ivars won't 
                        # cut it.
                        self._blob_ivarn.append([])
                        self._blob_ivars.append([])
                        
                        for l, pair in enumerate(element):
                                                        
                            assert type(pair) in [list, tuple], \
                                "Must supply blob_ivars as (variable, values)!"        
                            
                            self._blob_ivarn[k].append(pair[0])
                            self._blob_ivars[k].append(pair[1])
                                        
            else:
                self._blob_names = names
                self._blob_ivars = [None] * len(names)


            self._blob_nd = []
            self._blob_dims = []
            self._blob_funcs = []
            self._blob_kwargs = []
            for i, element in enumerate(self._blob_names):
                
                # Scalar blobs handled first
                if self._blob_ivars[i] is None:
                    self._blob_nd.append(0)
                    self._blob_dims.append(0)
                    
                    if hdf5_situation:
                        continue
                    
                    if self.pf['blob_funcs'] is None:
                        self._blob_funcs.append([None] * len(element))
                        self._blob_kwargs.append([None] * len(element))
                    elif self.pf['blob_funcs'][i] is None:
                        self._blob_funcs.append([None] * len(element))
                        self._blob_kwargs.append([None] * len(element))
                    else:
                        self._blob_funcs.append(self.pf['blob_funcs'][i])
                        # For backward compatibility
                        if 'blob_kwargs' in self.pf:
                            self._blob_kwargs.append(self.pf['blob_kwargs'][i])
                        else:
                            self._blob_kwargs.append([None] * len(element))
                        
                    continue
                # Everything else
                else:
                    
                    # Be careful with 1-D
                    if type(self._blob_ivars[i]) is np.ndarray:
                        lenarr = len(self._blob_ivars[i].shape)
                        assert lenarr == 1

                        self._blob_nd.append(1)
                        dims = len(self._blob_ivars[i]),
                        self._blob_dims.append(dims)
                    else:

                        self._blob_nd.append(len(self._blob_ivars[i]))

                        dims = tuple([len(element2) \
                            for element2 in self._blob_ivars[i]])
                        self._blob_dims.append(dims)
                
                # Handle functions
                
                try:
                    no_blob_funcs = self.pf['blob_funcs'] is None or \
                        self.pf['blob_funcs'][i] is None
                except (KeyError, TypeError, IndexError):
                    no_blob_funcs = True
                
                if no_blob_funcs:                     
                    self._blob_funcs.append([None] * len(element))
                    self._blob_kwargs.append([None] * len(element))
                    continue

                assert len(element) == len(self.pf['blob_funcs'][i]), \
                    "blob_names must have same length as blob_funcs!"
                self._blob_funcs.append(self.pf['blob_funcs'][i])
                
                if 'blob_kwargs' in self.pf:
                    self._blob_kwargs.append(self.pf['blob_kwargs'][i])
                else:
                    self._blob_kwargs.append(None)

        self._blob_nd = tuple(self._blob_nd)
        self._blob_dims = tuple(self._blob_dims)
        self._blob_names = tuple(self._blob_names)
        self._blob_ivars = tuple(self._blob_ivars)
        self._blob_ivarn = tuple(self._blob_ivarn)
        self._blob_funcs = tuple(self._blob_funcs)
        self._blob_kwargs = tuple(self._blob_kwargs)

    @property
    def blob_nbytes(self):
        """
        Estimate for the size of each blob (per walker per step).
        """    
                
        if not hasattr(self, '_blob_nbytes'):
            nvalues = 0.
            for i in range(self.blob_groups):
                if self.blob_nd[i] == 0:
                    nvalues += len(self.blob_names[i])
                else:
                    nvalues += len(self.blob_names[i]) \
                        * np.product(self.blob_dims[i])
    
            self._blob_nbytes = nvalues * 8.
        
        return self._blob_nbytes    
        
    @property 
    def all_blob_names(self):
        if not hasattr(self, '_all_blob_names'):
            
            if not self.blob_names:
                self._all_blob_names = []
                return []
            
            nested = any(isinstance(i, list) for i in self.blob_names)
            
            if nested:
                self._all_blob_names = []
                for i in range(self.blob_groups):
                    self._all_blob_names.extend(self.blob_names[i])    
            
            else:
                self._all_blob_names = self._blob_names
                
            if len(set(self._all_blob_names)) != len(self._all_blob_names):
                raise ValueError('Blobs must be unique!')
        
        return self._all_blob_names
        
    @property
    def blob_groups(self):
        if not hasattr(self, '_blob_groups'):
            
            nested = any(isinstance(i, list) for i in self.blob_names)
            
            if nested:
                if self.blob_nd is not None:
                    self._blob_groups = len(self.blob_nd)
                else:
                    self._blob_groups = 0
            else:
                self._blob_groups = None
                
        return self._blob_groups
                
    @property
    def blob_nd(self):    
        if not hasattr(self, '_blob_nd'):
            self._parse_blobs()
        return self._blob_nd
    
    @property
    def blob_dims(self):    
        if not hasattr(self, '_blob_dims'):
            self._parse_blobs()
        return self._blob_dims    
        
    @property
    def blob_names(self):
        if not hasattr(self, '_blob_names'):
            self._parse_blobs()
        return self._blob_names    
            
    @property
    def blob_ivars(self):
        if not hasattr(self, '_blob_ivars'):
            self._parse_blobs()
        return self._blob_ivars
    
    @property
    def blob_ivarn(self):
        if not hasattr(self, '_blob_ivarn'):
            self._parse_blobs()
        return self._blob_ivarn

    @property
    def blob_funcs(self):
        if not hasattr(self, '_blob_funcs'):
            self._parse_blobs()
        return self._blob_funcs
    
    @property
    def blob_kwargs(self):
        if not hasattr(self, '_blob_kwargs'):
            self._parse_blobs()
        return self._blob_kwargs    
    
    @property
    def blobs(self):
        if not hasattr(self, '_blobs'):
            if not self.blob_names:
                self._blobs = []
            else:
                try:
                    self._generate_blobs()
                except AttributeError as e:
                    if hasattr(self, 'prefix'):
                        self._blobs =\
                        read_pickle_file('{!s}.blobs.pkl'.format(self.prefix),\
                        nloads=1, verbose=False)
                    else:
                        raise AttributeError(e)
                        
        return self._blobs
        
    def get_ivars(self, name):
        
        if self.blob_groups is None:
            return self.blob_ivars[self.blob_names.index(name)]
        
        found_blob = False
        for i in range(self.blob_groups):
            for j, blob in enumerate(self.blob_names[i]):
                if blob == name:
                    found_blob = True
                    break
            
            if blob == name:
                break
                
        if not found_blob:
            print("WARNING: ivars for blob {} not found.".format(name))
            
            if name in self.derived_blob_names:
                print("CORRECTION: found {} in derived blobs!".format(name))    
            
                return self.derived_blob_ivars[name]
            
            return None
                
        return self.blob_ivars[i]
                
    def get_blob(self, name, ivar=None, tol=1e-2):
        """
        This is meant to recover a blob from a single simulation, i.e.,
        NOT a whole slew of them from an MCMC.
        """
        
        found = True
        #for i in range(self.blob_groups):
        #    for j, blob in enumerate(self.blob_names[i]):
        #        if blob == name:
        #            found = True
        #            break
        #
        #    if blob == name:
        #        break        
        
        try:
            i, j, dims, shape = self.blob_info(name)
        except KeyError:
            found = False

        if not found:
            print("WARNING: blob={} not found. This should NOT happen!".format(name))
            return np.inf

        if self.blob_nd[i] == 0:
            return float(self.blobs[i][j])
        elif self.blob_nd[i] == 1:
            if ivar is None:
                                
                try:
                    # When would this NOT be the case?
                    return self.blobs[i][j]
                except:
                    return self.blobs[i]
            elif len(self.blob_ivars[i]) == 1:
                iv = self.blob_ivars[i][1]
            else:
                iv = self.blob_ivars[i]

            # This is subject to rounding errors
            if ivar in iv:
                k = list(iv).index(ivar)
            elif np.any(np.abs(iv - ivar) < tol):
                k = np.argmin(np.abs(iv - ivar))
            else:
                raise IndexError("ivar={0:.2g} not in listed ivars!".format(\
                    ivar))
            
            return float(self.blobs[i][j][k])

        elif self.blob_nd[i] == 2:
            
            if ivar is None:
                return self.blobs[i][j]
            
            assert len(ivar) == 2
            # also assert that both values are in self.blob_ivars!
            # Actually, we don't have to abide by that. As long as a function
            # is provided we can evaluate the blob anywhere (with interp)
            
            kl = []
            for n in range(2):
                
                #if ivar[n] is None:
                #    kl.append(slice(0,None))
                #    continue
                #    
                assert ivar[n] in self.blob_ivars[i][n], \
                    "{} not in ivars for blob={}".format(ivar[n], name)

                #val = list(self.blob_ivars[i][n]).index(ivar[n])
                #    
                #kl.append(val)
                
                
            k = list(self.blob_ivars[i][0]).index(ivar[0])
            l = list(self.blob_ivars[i][1]).index(ivar[1])

            #k, l = kl

            #print(i,j,k,l)
            return float(self.blobs[i][j][k][l])
                        
    def _generate_blobs(self):
        """
        Create a list of blobs, one per blob group.
        
        ..note:: This should only be run for individual simulations,
            not in the analysis of MCMC data.
        
        Returns
        -------
        List, where each element has shape (ivar x blobs). Each element of 
        this corresponds to the blobs for one blob group, which is defined by
        either its dimensionality, its independent variables, or both.
        
        For example, for 1-D blobs, self.blobs[i][j][k] would mean
            i = blob group
            j = index corresponding to elements of self.blob_names
            k = index corresponding to elements of self.blob_ivars[i]
        """
                
        self._blobs = []
        for i, element in enumerate(self.blob_names):
                                                
            this_group = []
            for j, key in enumerate(element):
                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
                # 0-D blobs. Need to know name of attribute where stored!
                if self.blob_nd[i] == 0:
                    if self.blob_funcs[i][j] is None:
                        # Assume blob name is the attribute
                        #blob = self.__getattribute__(key)
                        blob = parse_attribute(key, self)
                                                
                    else:
                        fname = self.blob_funcs[i][j]
                                                
                        # In this case, the return of parse_attribute is
                        # a value, not a function to be applied to ivars.
                        blob = parse_attribute(fname, self)
                        
                # 1-D blobs. Assume the independent variable is redshift 
                # unless a function is provided
                elif self.blob_nd[i] == 1:
                    # The 0 index is because ivars are kept in a list no
                    # matter what
                    x = np.array(self.blob_ivars[i][0]).squeeze()
                    if (self.blob_funcs[i][j] is None) and (key in self.history):
                        blob = np.interp(x, self.history['z'][-1::-1],
                            self.history[key][-1::-1])
                    elif self.blob_funcs[i][j] is None:
                        raise KeyError('Blob {!s} not in history!'.format(key))
                    else:
                        fname = self.blob_funcs[i][j]

                        # Name of independent variable
                        xn = self.blob_ivarn[i][0]

                        if isinstance(fname, basestring):
                            func = parse_attribute(fname, self)
                        else:

                            print('hey {!s}'.format(fname))
                            raise ValueError('pretty sure this is broken!')

                            # fname is a slice, like ('igm_k_heat', 0)
                            # to retrieve heating rate from H ionizations
                            _xx = self.history['z'][-1::-1]
                            _yy = self.history[fname[0]][-1::-1,fname[1]]

                            func = (_xx, _yy)

                        if ismethod(func) or isinstance(func, interp1d) or \
                            (type(func) == FunctionType) \
                            or hasattr(func, '__call__'):

                            try:
                                if self.blob_kwargs[i] is not None:
                                    kw = self.blob_kwargs[i][j]
                                else:
                                    kw = {}

                                def func_kw(xx):
                                    _kw = kw.copy()
                                    _kw.update({xn:xx})
                                    return func(**_kw)
                                    
                                blob = np.array([func_kw(xx) for xx in x])
                                                                
                            except TypeError:
                                blob = np.array(list(map(func, x)))
                            
                        else:
                            blob = np.interp(x, func[0], func[1])
                                                                                                                                  
                else:
                                        
                    # Must have blob_funcs for this case
                    fname = self.blob_funcs[i][j]
                    tmp_f = parse_attribute(fname, self)
                    
                    xarr, yarr = list(map(np.array, self.blob_ivars[i]))
                    
                    if (type(tmp_f) is FunctionType) or ismethod(tmp_f) \
                        or hasattr(func, '__call__'):
                        func = tmp_f
                    elif type(tmp_f) is tuple:
                        z, E, flux = tmp_f
                        func = RectBivariateSpline(z, E, flux)
                    else:
                        raise TypeError('Sorry: don\'t understand blob {!s}'.format(key))
                                      
                    xn, yn = self.blob_ivarn[i]
                                                            
                    blob = []
                    # We're assuming that the functions are vectorized.
                    # Didn't used to, but it speeds things up (a lot).
                    for x in xarr:
                        tmp = []
                        
                        if self.blob_kwargs[i] is not None:
                            kw = self.blob_kwargs[i][j]
                        else:
                            kw = {}
                                                                
                        kw.update({xn:x, yn:yarr})
                        result = func(**kw)
                                
                        # Happens when we save a blob that isn't actually
                        # a PQ (i.e., just a constant). Need to kludge so it
                        # doesn't crash.
                        if type(result) in [int, float, np.float64]:
                            result = result * np.ones_like(yarr)
                        
                        tmp.extend(result)
                        blob.append(tmp)
                                                                                                                        
                this_group.append(np.array(blob))
                                
            self._blobs.append(np.array(this_group))
            
    @property 
    def blob_data(self):
        if not hasattr(self, '_blob_data'):
            self._blob_data = {}
        return self._blob_data
    
    @blob_data.setter
    def blob_data(self, value):
        self._blob_data.update(value)    
    
    def get_blob_from_disk(self, name):
        return self.__getitem__(name)
    
    def __getitem__(self, name):
        if name in self.blob_data:
            return self.blob_data[name]
        
        return self._get_item(name)
    
    def blob_info(self, name):
        """
        Returns
        -------
        index of blob group, index of element within group, dimensionality, 
        and exact dimensions of blob.
        """
        
        if hasattr(self, 'derived_blob_names'):
            # This is bad practice since this is an attribute of ModelSet,
            # i.e., the child class (sometimes)
            if name in self.derived_blob_names:
                iv = self.derived_blob_ivars[name]
                return None, None, len(iv), tuple([len(element) for element in iv])
        
        nested = any(isinstance(i, list) for i in self.blob_names)
        
        if nested:
        
            found = False
            for i, group in enumerate(self.blob_names):
                for j, element in enumerate(group):
                    if element == name:
                        found = True
                        break            
                if element == name:
                    break
                            
            if not found:
                raise KeyError('Blob {!s} not found.'.format(name))
            
            return i, j, self.blob_nd[i], self.blob_dims[i]
        else:
            i = self.blob_names.index(name)
            return None, None, self.blob_nd[i], self.blob_dims[i]
    
    def _get_item(self, name):
        
        i, j, nd, dims = self.blob_info(name)
    
        fn = "{0!s}.blob_{1}d.{2!s}.pkl".format(self.prefix, nd, name)
                                
        # Might have data split up among processors or checkpoints
        by_proc = False
        by_dd = False
        if not os.path.exists(fn):
            
            # First, look for processor-by-processor outputs
            fn = "{0!s}.000.blob_{1}d.{2!s}.pkl".format(self.prefix, nd, name)
            if os.path.exists(fn):
                by_proc = True        
                by_dd = False
            # Then, those where each checkpoint has its own file    
            else:
                by_proc = False
                by_dd = True
                
                search_for = "{0!s}.dd????.blob_{1}d.{2!s}.pkl".format(\
                    self.prefix, nd, name)
                _ddf = glob.glob(search_for)
                        
                if self.include_checkpoints is None:
                    ddf = _ddf
                else:
                    ddf = []
                    for dd in self.include_checkpoints:
                        ddid = str(dd).zfill(4)
                        tmp = "{0!s}.dd{1!s}.blob_{2}d.{3!s}.pkl".format(\
                            self.prefix, ddid, nd, name)
                        ddf.append(tmp)
                             
                # Need to put in order if we want to match up with
                # chain etc.
                ddf = np.sort(ddf)
                
                # Start with the first
                fn = ddf[0]
                        
        fid = 0
        to_return = []
        while True:
            
            if not os.path.exists(fn):
                break
        
            all_data = []
            data_chunks = read_pickle_file(fn, nloads=None, verbose=False)
            for data_chunk in data_chunks:
                all_data.extend(data_chunk)
            del data_chunks
            
            print("# Loaded {}".format(fn))
                
            # Used to have a squeeze() here for no apparent reason...
            # somehow it resolved itself.
            all_data = np.array(all_data, dtype=np.float64)
            to_return.extend(all_data)
            
            if not (by_proc or by_dd):
                break
                
            fid += 1
            
            if by_proc:
                fn = "{0!s}.{1!s}.blob_{2}d.{3!s}.pkl".format(self.prefix,\
                    str(fid).zfill(3), nd, name)
            else:
                if (fid >= len(ddf)):
                    break
                    
                fn = ddf[fid]
        
        mask = np.logical_not(np.isfinite(to_return))
        masked_data = np.ma.array(to_return, mask=mask)
        
        # CAN BE VERY CONFUSING
        #if by_proc and rank == 0:
        #    fn = "{0!s}.blob_{1}d.{2!s}.pkl".format(self.prefix, nd, name)
        #    write_pickle_file(masked_data, fn, ndumps=1, open_mode='w',\
        #        safe_mode=False, verbose=False)
        
        self.blob_data = {name: masked_data}
        
        return masked_data
    
    
    
