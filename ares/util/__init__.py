import re
import numpy as np
from .Math import *
from .Aesthetics import labels
from collections import Iterable
from .WriteData import CheckPoints
from .ProgressBar import ProgressBar
from .ProblemTypes import ProblemType
from .RestrictTimestep import RestrictTimestep
from .SetDefaultParameterValues import SetAllDefaults
from .CheckForParameterConflicts import CheckForParameterConflicts

try:
    from mathutils.differentiate import central_difference
except ImportError:
    pass

try:
    import h5py
except ImportError:
    pass

defaults = SetAllDefaults()

def parse_kwargs(**kwargs):
    """
    Parse kwargs dictionary - populate with defaults.
    """    
    
    pf = defaults.copy()
    
    if not kwargs:
        pf.update(ProblemType(1))
    elif 'problem_type' in kwargs:
        pf.update(ProblemType(kwargs['problem_type']))
    
    for kwarg in kwargs:
        if kwarg not in defaults.keys():
            print 'WARNING: Unrecognized parameter: %s' % kwarg
    
    pf.update(kwargs)
                
    conflicts = CheckForParameterConflicts(pf)

    if conflicts:
        raise Exception('Conflict(s) in input parameters.')

    return pf
    
class evolve:
    """ Make things that may or may not evolve with time callable. """
    def __init__(self, val):
        self.val = val
        self.callable = val == types.FunctionType
    def __call__(self, z = None):
        if self.callable:
            return self.val(z)
        else:
            return self.val
            
def sort(pf, prefix='spectrum', make_list=True, make_array=False):
    """
    Turn any item that starts with prefix_ into a list, if it isn't already.
    Hack off the prefix when we're done.
    """            

    result = {}
    for par in pf.keys():
        if par[0:len(prefix)] != prefix:
            continue

        new_name = par.partition('_')[-1]
        if (isinstance(pf[par], Iterable) and type(pf[par]) is not str) \
            or (not make_list):
            result[new_name] = pf[par]
        elif make_list:
            result[new_name] = [pf[par]]

    # Make sure all elements are the same length?      
    if make_list or make_array:  
        lmax = 1
        for par in result:
            lmax = max(lmax, len(result[par]))

        for par in result:
            if len(result[par]) == lmax:
                continue

            result[par] = lmax * [result[par][0]]

            if make_array:
                result[par] = np.array(result[par])

    return result

def _load_hdf5(fn):    
    inits = {}
    f = h5py.File(fn)
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

def num_freq_bins(Nx, zi=40, zf=10, Emin=2e2, Emax=3e4):
    """
    Compute number of frequency bins required for given log-x grid.

    Defining the variable x = 1 + z, and setting up a grid in log-x containing
    Nx elements, compute the number of frequency bins required for a 1:1 
    mapping between redshift and frequency.

    """
    x = np.logspace(np.log10(1.+zf), np.log10(1.+zi), Nx)
    R = x[1] / x[0]

    # Create mapping to frequency space
    Etmp = 1. * Emin
    n = 1
    while Etmp < Emax:
        Etmp = Emin * R**(n - 1)
        n += 1

    # Subtract 2: 1 because we overshoot Emax in while loop, another because
    # n is index-1-based (?)

    return n-2

logbx = lambda b, x: np.log10(x) / np.log10(b)