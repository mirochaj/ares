"""

ReadData.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Aug 27 10:55:19 MDT 2014

Description: 

"""

import pickle
import numpy as np

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1

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
    
def read_pickled_chain(fn, logL=None):
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

    
    
    
    