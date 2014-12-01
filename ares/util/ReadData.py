"""

ReadData.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Aug 27 10:55:19 MDT 2014

Description: 

"""

import numpy as np
import os, pickle, re

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
    
def _load_hdf5(fn):    
    inits = {}
    f = h5py.File(fn, 'r')
    for key in f.keys():
        inits[key] = np.array(f[key].value)
    f.close()

    return inits

def _load_npz(fn):
    return dict(np.load(fn))

def load_inits(fn=None):

    if fn is None:
        if have_h5py:
            fn = '%s/input/inits/initial_conditions.hdf5' % ARES
            inits = _load_hdf5(fn)
        else:
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
    
def read_pickled_chain(fn, logL=None, arrayify=True):
    """
    Reads MCMC chain from emcee that has been pickled.
    
    Parameters
    ----------
    fn : str
        Name of file containing flattened chain.
    logL : str
        Name of file containing likelihoods.

    Returns
    -------
    All the stuff.
    
    """
    
    f = open(fn, 'rb')
    
    results = []
    
    while True:
        try:
            # This array should be (nsteps, ndims)
            results.extend(pickle.load(f))
        except EOFError:
            break
            
    f.close()
    
    fc = np.array(results)
    
    if rank == 0:
        print "Loaded %s." % fn
    
    if logL is not None:
        
        f = open(logL)
        
        lnL = []
        while True:
            try:
                lnL.extend(pickle.load(f))
            except EOFError:
                break

        f.close()
        
        if rank == 0:
            print "Loaded %s." % logL
        
        return fc, np.array(lnL)
    else:
        return fc

    
    
    
    