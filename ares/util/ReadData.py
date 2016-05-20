"""

ReadData.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Aug 27 10:55:19 MDT 2014

Description: 

"""

import numpy as np
import imp as _imp
import os, re, sys

#try:
#    import dill as pickle
#except ImportError:
import pickle

try:
    import h5py
    have_h5py = True
except ImportError:
    have_h5py = False

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
except ImportError:
    rank = 0
    
ARES = os.environ.get('ARES')
sys.path.insert(1, '%s/input/litdata' % ARES)

def read_lit(prefix, path=None):
    """
    Read data from the literature.
    
    Parameters
    ----------
    prefix : str
        Everything preceeding the '.py' in the name of the module.
    path : str
        If you want to look somewhere besides $ARES/input/litdata, provide
        that path here.

    """

    if path is not None:
        prefix = '%s/%s' % (path, prefix)
    
    # Load custom defaults    
    if os.path.exists('%s/input/litdata/%s.py' % (ARES, prefix)):
        _f, _filename, _data = _imp.find_module(prefix, 
            ['%s/input/litdata/' % ARES])
        mod = _imp.load_module('%s' % prefix, _f, _filename, _data)
    else:
        mod = None
    
    return mod

def flatten_energies(E):
    """
    Take fluxes sorted by band and flatten to single energy dimension.
    """

    to_return = []
    for i, band in enumerate(E):
        if type(band) is list:
            for j, flux_seg in enumerate(band):
                to_return.extend(flux_seg)
        else:
            to_return.extend(band)
    
    return np.array(to_return)

def flatten_flux(flux):
    return flatten_energies(flux)

def split_flux(energies, fluxes):
    """
    Take flattened fluxes and re-sort into band-grouped fluxes.
    """
    
    i_E = np.cumsum(map(len, energies))
    fluxes_split = np.hsplit(fluxes, i_E)

    return fluxes_split

def _sort_flux_history(all_fluxes):
    pass    

def _sort_history(all_data, prefix='', squeeze=False):
    """
    Take list of dictionaries and re-sort into 2-D arrays.
    
    Parameters
    ----------
    all_data : list
        Each element is a dictionary corresponding to data at a particular 
        snapshot.
    prefix : str
        Will prepend to all dictionary keys in output dictionary.

    Returns
    -------
    Dictionary, sorted by gas properties, with entire history for each one.
    """

    data = {}
    for key in all_data[0]:
        if type(key) is int and not prefix.strip():
            name = int(key)
        else:
            name = '%s%s' % (prefix, key)
        
        data[name] = []
        
    # Loop over time snapshots
    for element in all_data:

        # Loop over fields
        for key in element:
            if type(key) is int and not prefix.strip():
                name = int(key)
            else:
                name = '%s%s' % (prefix, key)
                
            data[name].append(element[key])

    # Cast everything to arrays
    for key in data:
        if squeeze:
            data[key] = np.array(data[key]).squeeze()
        else:
            data[key] = np.array(data[key])

    return data

tanh_gjah_to_ares = \
{
 'J_0/J_21': 'tanh_J0',
 'dz_J': 'tanh_Jdz',
 'z_{0,j}': 'tanh_Jz0',
 'T_0': 'tanh_T0',
 'dz_T': 'tanh_Tdz',
 'z_{0,T}': 'tanh_Tz0',
 'x_0': 'tanh_x0',
 'dz_x': 'tanh_xdz',
 'z_{0,x}': 'tanh_xz0',
 'b_\\nu': 'tanh_bias_freq',
 'b_T': 'tanh_bias_temp',
}    

fcoll_gjah_to_ares = \
{
'\\log_{10}\\xi_\\mathrm{LW}': 'xi_LW',
'\\log_{10}\\xi_\\mathrm{XR}': 'xi_XR',
'\\log_{10}xi_\\mathrm{UV}': 'xi_UV',
'\\log_{10}T_\\mathrm{min}': 'Tmin',
}
    
def _load_hdf5(fn):    
    inits = {}
    f = h5py.File(fn, 'r')
    for key in f.keys():
        inits[key] = np.array(f[key].value)
    f.close()

    return inits

def _load_npz(fn):
    data = np.load(fn)
    new = {'z': data['z'].copy(), 'Tk': data['Tk'].copy(), 'xe': data['xe'].copy()}
    data.close()
    return new

def _load_inits(fn=None):

    if fn is None:
        fn = '%s/input/inits/initial_conditions.npz' % ARES
        inits = _load_npz(fn)

    else:
        if re.search('.hdf5', fn):
            inits = _load_hdf5(fn)
        else:
            inits = _load_npz(fn)

    return inits        

def read_pickled_blobs(fn):
    """
    Reads arbitrary meta-data blobs from emcee that have been pickled.
    
    Parameters
    ----------
    chain : str
        Name of file containing flattened chain.
    logL : str
        Name of file containing likelihoods.
    pars : str
        List of parameters corresponding to second dimension of chain.
        
    Returns
    -------
    All the stuff.
    
    """

    pass
    
def flatten_blobs(data):
    """
    Take a 3-D array, eliminate dimension corresponding to walkers, thus
    reducing it to 2-D
    """

    # Prevents a crash in MCMC.ModelFit
    if np.all(data == {}):
        return None

    if len(data.shape) != 4:
        raise ValueError('chain ain\'t the right shape.')    

    new = []
    for i in range(data.shape[1]):
        new.extend(data[:,i,:,:])

    return new
    
def flatten_chain(data):
    """
    Take a 3-D array, eliminate dimension corresponding to walkers, thus
    reducing it to 2-D
    """

    if len(data.shape) != 3:
        raise ValueError('chain ain\'t the right shape.')    

    new = []
    for i in range(data.shape[1]):
        new.extend(data[:,i,:])

    return new

def flatten_logL(data):
    """
    Take a 2-D array, eliminate dimension corresponding to walkers, thus
    reducing it to 1-D
    """

    if len(data.shape) != 2:
        raise ValueError('chain ain\'t the right shape.')    

    new = []
    for i in range(data.shape[1]):
        new.extend(data[:,i])

    return new

def read_pickled_dict(fn):
    f = open(fn, 'rb')
    
    results = []
    
    while True:
        try:
            # This array should be (nsteps, ndims)
            results.append(pickle.load(f))
        except EOFError:
            break
    
    f.close()
    
    return results
            
def read_pickle_file(fn):
    f = open(fn, 'rb')

    ct = 0
    results = []
    while True:
        try:
            data = pickle.load(f)
            results.extend(data)
            ct +=1
        except EOFError:
            break

    #print "Read %i chunks from %s." % (ct, fn)

    f.close()
    
    return np.array(results)

def read_pickled_blobs(fn):
    return read_pickle_file(fn)    
    
def read_pickled_logL(fn):    
    # Removes chunks dimension
    data = read_pickle_file(fn)
    
    Nd = len(data.shape)
    
    # A flattened logL should have dimension (iterations)
    if Nd == 1:
        return data
        
    # (walkers, iterations)
    elif Nd >= 2:
        new_data = []
        for element in data:
            if Nd == 2:
                new_data.extend(element)
            else:
                new_data.extend(element[0,:])
        return np.array(new_data)
    
   
    else:
        raise ValueError('unrecognized logL shape')
    
def read_pickled_chain(fn):
    
    # Removes chunks dimension
    data = read_pickle_file(fn)
    
    Nd = len(data.shape)

    # Flattened chain
    if Nd == 2:
        return np.array(data)
    
    # Unflattnened chain
    elif Nd == 3:
        new_data = []
        for element in data:
            new_data.extend(element)
            
        return np.array(new_data)
        
    else:
        print Nd
        raise ValueError('unrecognized chain shape')
            
    
def delete_nan_rows(array):
    """
    Finds a copy of the given array with all rows (slices of constant 0th
    index) which contain nan's or inf's removed.

    array a numpy.ndarray with more than one dimension

    returns a copy of array with rows with nan's or inf's removed
    """
    indices = []
    # same shape as array with boolean type; True only for nan's and infs
    is_inf_or_nan = (np.isinf(array)|np.isnan(array))
    # for loop finds which rows to delete
    for i in range(len(array)):
        if not np.all(~is_inf_or_nan[i,...]):
            indices.append(i)
    del is_inf_or_nan
    # this does not change array; just returns modified copy
    print "delete_nan_rows was run. It deleted %i rows." % (len(indices),)
    return np.delete(array, indices, axis=0)
        
    
    
