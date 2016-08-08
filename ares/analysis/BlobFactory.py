"""

BlobFactory.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Dec 11 14:24:53 PST 2015

Description: 

"""

import os
import re
import numpy as np
from inspect import ismethod
from types import FunctionType
from scipy.interpolate import RectBivariateSpline

try:
    import dill as pickle
except ImportError:
    import pickle
    
try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1    
    
# Some standard blobs    
    
class default_blobs(object):
    def __init__(self):
        blobs_1d = ['dTb', 'igm_Ts', 'igm_Tk', 'cgm_h_2', 'igm_h_1', 
            'igm_k_heat_h_1', 'cgm_k_ion_h_1']
        blobs_scalar = ['z_B', 'z_C', 'z_D']
        for key in blobs_1d:    
            for tp in list('BCD'):
                blobs_scalar.append('%s_%s' % (key, tp))
                
        self.blob_names = [blobs_scalar, blobs_1d]
        self.blob_ivars = [None, np.arange(5, 41, 1)]
    
    
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
        return eval('obj_base.%s' % s)

    # Nested attribute
    blob_attr = None
    obj_list = [obj_base]
    for i in range(len(attr_split)):
        
        # One particular chunk of the attribute name
        s = attr_split[i]

        new_obj = eval('obj_base.%s' % s)
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

    def _parse_blobs(self):
                        
        try:
            names = self.pf['blob_names']
        except KeyError:
            names = None

        if names is None:
            self._blob_names = self._blob_ivars = None
            self._blob_dims = self._blob_nd = None
            self._blob_funcs = None
            return None
        else:
            # Otherwise, figure out how many different kinds (shapes) of
            # blobs we have
            assert type(names) in [list, tuple], \
                "Must supply blob_names as list or tuple!"

            self._blob_names = names
            if 'blob_ivars' in self.pf:
                if self.pf['blob_ivars'] is None:
                    self._blob_ivars = [None] * len(names)
                else:
                    self._blob_ivars = self.pf['blob_ivars']
            else:
                self._blob_ivars = [None] * len(names)

            self._blob_nd = []
            self._blob_dims = []
            self._blob_funcs = []
            for i, element in enumerate(self._blob_names):
                
                # Scalar blobs handled first
                if self._blob_ivars[i] is None:
                    self._blob_nd.append(0)
                    self._blob_dims.append(0)
                    
                    if self.pf['blob_funcs'] is None:
                        self._blob_funcs.append([None] * len(element))
                    elif self.pf['blob_funcs'][i] is None:
                        self._blob_funcs.append([None] * len(element))
                    else:
                        self._blob_funcs.append(self.pf['blob_funcs'][i])
                        
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
                    continue

                assert len(element) == len(self.pf['blob_funcs'][i]), \
                    "blob_names must have same length as blob_funcs!"
                self._blob_funcs.append(self.pf['blob_funcs'][i])

        self._blob_nd = tuple(self._blob_nd)                    
        self._blob_dims = tuple(self._blob_dims)            
        self._blob_names = tuple(self._blob_names)
        self._blob_ivars = tuple(self._blob_ivars)
        self._blob_funcs = tuple(self._blob_funcs)
        
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
            self._all_blob_names = []
            for i in range(self.blob_groups):
                self._all_blob_names.extend(self.blob_names[i])    
        
        return self._all_blob_names
        
    @property
    def blob_groups(self):
        if not hasattr(self, '_blob_groups'):
            if self.blob_nd is not None:
                self._blob_groups = len(self.blob_nd)
            else:
                self._blob_groups = 0
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
    def blob_funcs(self):
        if not hasattr(self, '_blob_funcs'):
            self._parse_blobs()
        return self._blob_funcs

    @property
    def blobs(self):
        if not hasattr(self, '_blobs'):
            if not self.blob_names:
                self._blobs = []
            else:
                self._generate_blobs()

        return self._blobs

    def get_blob(self, name, ivar=None):
        """
        This is meant to recover a blob from a single simulation, i.e.,
        NOT a whole slew of them from an MCMC.
        """
        for i in range(self.blob_groups):
            for j, blob in enumerate(self.blob_names[i]):
                if blob == name:
                    break
            
            if blob == name:
                break        

        if self.blob_nd[i] > 0 and (ivar is None):
            raise ValueError('Must provide ivar!')
        elif self.blob_nd[i] == 0:
            return float(self.blobs[i][j])
        elif self.blob_nd[i] == 1:
            
            if len(self.blob_ivars[i]) == 1:
                iv = self.blob_ivars[i][0]
            else:
                iv = self.blob_ivars[i]     
            
            assert ivar in iv
            k = list(iv).index(ivar)

            return float(self.blobs[i][j][k])

        elif self.blob_nd[i] == 2:
            assert len(ivar) == 2
            # also assert that both values are in self.blob_ivars!
            # Actually, we don't have to abide by that. As long as a function
            # is provided we can evaluate the blob anywhere (with interp)

            for n in range(2):
                assert ivar[n] in self.blob_ivars[i][n]

            k = list(self.blob_ivars[i][0]).index(ivar[0])
            l = list(self.blob_ivars[i][1]).index(ivar[1])

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
                    x = np.array(self.blob_ivars[i]).squeeze()
                    if (self.blob_funcs[i][j] is None) and (key in self.history):
                        blob = np.interp(x, self.history['z'][-1::-1],
                            self.history[key][-1::-1])
                    elif self.blob_funcs[i][j] is None:
                        raise KeyError('Blob %s not in history!' % key)
                    else:
                        fname = self.blob_funcs[i][j]
                        func = parse_attribute(fname, self)
                                                
                        if ismethod(func):
                            blob = np.array(map(func, x))
                        else:
                            blob = np.interp(x, func[0], func[1])
                else:
                    # Must have blob_funcs for this case
                    fname = self.blob_funcs[i][j]
                    tmp_f = parse_attribute(fname, self)

                    xarr, yarr = map(np.array, self.blob_ivars[i])

                    if type(tmp_f) is FunctionType:
                        func = tmp_f
                    elif type(tmp_f) is tuple:
                        z, E, flux = tmp_f
                        func = RectBivariateSpline(z, E, flux)
                    else:
                        raise TypeError('Sorry: don\'t understand blob %s' % name)
                    
                    blob = []
                    for x in xarr:
                        tmp = []
                        for y in yarr:                            
                            tmp.append(func(x, y))

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
        found = False
        for i, group in enumerate(self.blob_names):
            for j, element in enumerate(group):
                if element == name:
                    found = True
                    break            
            if element == name:
                break
                        
        if not found:
            raise KeyError('Blob %s not found.' % name)        
                
        return i, j, self.blob_nd[i], self.blob_dims[i]
    
    def _get_item(self, name):
        
        i, j, nd, dims = self.blob_info(name)
    
        fn = "%s.blob_%id.%s.pkl" % (self.prefix, nd, name)
        
        # Might have data split up among processors
        by_proc = False
        if not os.path.exists(fn):
            by_proc = True
            fn = "%s.000.blob_%id.%s.pkl" % (self.prefix, nd, name)
        
        fid = 0
        to_return = []
        while True:
            
            if not os.path.exists(fn):
                break
        
            f = open(fn, 'rb')
    
            all_data = []
            while True:
                try:
                    data = pickle.load(f)
                except EOFError:
                    break
                
                all_data.extend(data)
                
            f.close()    
                
            # Used to have a squeeze() here for no apparent reason...
            # somehow it resolved itself.
            all_data = np.array(all_data, dtype=np.float64)
            to_return.extend(all_data)
            
            if not by_proc:
                break
                
            fid += 1
            fn = "%s.%s.blob_%id.%s.pkl" % (self.prefix, str(fid).zfill(3),
                nd, name)
        
        mask = np.logical_not(np.isfinite(to_return))
        masked_data = np.ma.array(to_return, mask=mask)
        
        # CAN BE VERY CONFUSING
        #if by_proc and rank == 0:
        #    f = open("%s.blob_%id.%s.pkl" % (self.prefix, nd, name), 'wb')
        #    pickle.dump(masked_data, f)
        #    f.close()
        
        self.blob_data = {name: masked_data}
        
        return masked_data
    
    
    
