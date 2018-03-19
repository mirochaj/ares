"""

ReadData.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Aug 27 10:55:19 MDT 2014

Description: 

"""

import numpy as np
import imp as _imp
import os, re, sys, glob
from .Pickling import read_pickle_file

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
    
HOME = os.environ.get('HOME')
ARES = os.environ.get('ARES')
sys.path.insert(1, '{!s}/input/litdata'.format(ARES))

_lit_options = glob.glob('{!s}/input/litdata/*.py'.format(ARES))
lit_options = []
for element in _lit_options:
    lit_options.append(element.split('/')[-1].replace('.py', ''))

def read_lit(prefix, path=None, verbose=True):
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
        prefix = '{0!s}/{1!s}'.format(path, prefix)
    
    has_local = os.path.exists('./{!s}.py'.format(prefix))
    has_home = os.path.exists('{0!s}/.ares/{1!s}.py'.format(HOME, prefix))
    has_litd = os.path.exists('{0!s}/input/litdata/{1!s}.py'.format(ARES, prefix))
    
    # Load custom defaults
    if has_local:
        loc = '.'    
    elif has_home:
        loc = '{!s}/.ares/'.format(HOME)
    elif has_litd:
        loc = '{!s}/input/litdata/'.format(ARES)
    else:
        return None

    if has_local + has_home + has_litd > 1:
        print("WARNING: multiple copies of {!s} found.".format(prefix))
        print("       : precedence: CWD -> $HOME -> $ARES/input/litdata")

    _f, _filename, _data = _imp.find_module(prefix, [loc])
    mod = _imp.load_module(prefix, _f, _filename, _data)
    
    # Save this for sanity checks later
    mod.path = loc
    
    return mod
    
def flatten_energies(E):
    """
    Take fluxes sorted by band and flatten to single energy dimension.
    """
    
    to_return = []
    for i, band in enumerate(E):
        if type(band) is list:
            to_return.extend(np.concatenate(band))
        elif type(band) is np.ndarray:
            to_return.extend(band)
        else:
            to_return.append(float(band))

    return to_return

def flatten_energies_OLD(E):
    """
    Take fluxes sorted by band and flatten to single energy dimension.
    """

    to_return = []
    for i, band in enumerate(E):
        if type(band) is list:
            for j, flux_seg in enumerate(band):
                to_return.extend(flux_seg)
        else:
            try:
                to_return.extend(band)
            except TypeError:
                to_return.append(band)

    return np.array(to_return)

def flatten_flux(arr):
    return flatten_energies(arr)

def flatten_emissivities(arr, z, Eflat):
    """
    Return an array as function of redshift and (flattened) energy.
    
    The input 'arr' will be a nested thing that is pretty nasty to deal with.

    The first dimension corresponds to band 'chunks'. Elements within each
    chunk may be a single array (2D: function of redshift and energy), or, in
    the case of sawtooth regions of the spectrum, another list where each
    element is some sub-chunk of a sawtooth background.

    """

    to_return = np.zeros((z.size, Eflat.size))

    k1 = 0
    k2 = 0
    for i, band in enumerate(arr):
        if type(band) is list:
            for j, flux_seg in enumerate(band):
                # flux_seg will be (z, E)
                N = len(flux_seg[0].squeeze())
                if k2 is None:
                    k2 = N
                k2 += N    
                    
                print('{} {} {} {} {}'.format(i, j, N, k1, k2))
                to_return[:,k1:k2] = flux_seg.squeeze()
                k1 += N
                
        else:
            # First dimension is redshift.
            print('{!s}'.format(band.shape))
            to_save = band.squeeze()
            
            # Rare occurence...
            if to_save.ndim == 1:
                if np.all(to_save == 0):
                    continue
            
            N = len(band[0].squeeze())
            if k2 is None:
                k2 = N
            print('{} {} {} {} {} {}'.format('hey', i, j, N, k1, k2))
            k2 += N
            to_return[:,k1:k2] = band.copy()
            k1 += N


    return to_return

def split_flux(energies, fluxes):
    """
    Take flattened fluxes and re-sort into band-grouped fluxes.
    """
    
    i_E = np.cumsum(list(map(len, energies)))
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
            name = '{0!s}{1!s}'.format(prefix, key)

        data[name] = []

    # Loop over time snapshots
    for element in all_data:

        # Loop over fields
        for key in element:
            if type(key) is int and not prefix.strip():
                name = int(key)
            else:
                name = '{0!s}{1!s}'.format(prefix, key)
                
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
        assert ARES is not None, "$ARES environment variable has not been set!"
        fn = '{!s}/input/inits/initial_conditions.npz'.format(ARES)
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
        raise ValueError("Chain shape {} incorrect. Should be 3-D".format(data.shape))    

    new = []
    for i in range(data.shape[1]):
        new.extend(data[:,i,:])

    # Is there a reason not to cast this to an array?
    return new

def flatten_logL(data):
    """
    Take a 2-D array, eliminate dimension corresponding to walkers, thus
    reducing it to 1-D
    """

    if len(data.shape) != 2:
        raise ValueError("loglikelihood shape {} incorrect. Should be 2-D".format(data.shape))

    new = []
    for i in range(data.shape[1]):
        new.extend(data[:,i])

    return new
            
def concatenate(lists):
    return np.concatenate(lists, axis=0)

def read_pickled_blobs(fn):
    return concatenate(read_pickle_file(fn, nloads=None, verbose=False))
    
def read_pickled_logL(fn):    
    # Removes chunks dimension
    data = concatenate(read_pickle_file(fn, nloads=None, verbose=False))
    
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
    data = concatenate(read_pickle_file(fn, nloads=None, verbose=False))
    
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
        raise ValueError('Unrecognized chain shape')
        
    
    
