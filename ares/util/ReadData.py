"""

ReadData.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Aug 27 10:55:19 MDT 2014

Description: 

"""

import numpy as np
import os, pickle, cPickle, re
from pickle import UnpicklingError

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

def load_inits(fn=None):

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
        raise ValueError('unrecognized chain shape')
            
    
    
        
    
    