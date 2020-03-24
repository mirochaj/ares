"""

Misc.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sun Oct 19 19:50:31 MDT 2014

Description: 

"""

import sys
import re, os
import numpy as np
import matplotlib.pyplot as pl
from scipy.integrate import cumtrapz
from ..physics.Constants import sigma_T
from .SetDefaultParameterValues import SetAllDefaults
from .CheckForParameterConflicts import CheckForParameterConflicts

if sys.version_info[0] >= 3:
    if sys.version_info[1] > 3:
        from collections.abc import Iterable
    else:
        from collections import Iterable
else:
    from collections import Iterable

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

logbx = lambda b, x: np.log10(x) / np.log10(b)

def get_cmd_line_kwargs(argv):
    
    cmd_line_kwargs = {}

    for arg in argv[1:]:
        try:
            pre, post = arg.split('=')
        except ValueError:
            # To deal with parameter values that have an '=' in them.
            pre = arg[0:arg.find('=')]
            post = arg[arg.find('=')+1:]

        # Need to do some type-casting
        if post.isdigit():
            cmd_line_kwargs[pre] = int(post)
        elif post.isalpha():
            if post == 'None':
                cmd_line_kwargs[pre] = None
            elif post in ['True', 'False']:
                cmd_line_kwargs[pre] = bool(post)    
            else:    
                cmd_line_kwargs[pre] = str(post)
        elif post[0] == '[':
            vals = post[1:-1].split(',')
            cmd_line_kwargs[pre] = np.array(map(float, vals))
        else:
            try:
                cmd_line_kwargs[pre] = float(post)
            except ValueError:
                # strings with underscores will return False from isalpha
                cmd_line_kwargs[pre] = str(post)
    
    return cmd_line_kwargs

def get_hg_rev():
    import subprocess
    try:
        ARES = os.getenv('ARES')
        pipe = subprocess.Popen(["hg", "id", "-i", ARES], stdout=subprocess.PIPE)
    except:
        return 'unknown'
        
    return pipe.stdout.read().strip()
    
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
        if (isinstance(pf[par], Iterable) and not isinstance(pf[par], basestring)) \
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

def tau_post_EoR(sim):
    """
    Compute Thomson optical depth from z=0 to the lowest redshift contained
    in an ares.simulations.MultiPhaseMedium calculation.
    
    .. note:: Assumes ___ about helium EoR?
    
    Parameters
    ----------
    sim : instance of ares.simulations.MultiPhaseMedium or Global21cm
    
    Returns
    -------
    Array of redshifts and (cumulative) optical depth out to those redshifts.
    
    """
    
    zmin = sim.history['z'].min()
    ztmp = np.arange(0, zmin+0.001, 0.001)

    QHII = 1.0
    nH = sim.grid.cosm.nH(ztmp)
    dldz = sim.grid.cosm.dldz(ztmp)

    integrand = QHII * nH

    if 'igm_he_1' in sim.history:
        QHeII = 1.0
        xHeIII = 1.0
        nHe = sim.grid.cosm.nHe(ztmp)
        integrand += (QHeII + 2. * xHeIII) * nHe 

    integrand *= sigma_T * dldz

    tau = cumtrapz(integrand, ztmp, initial=0)
    
    return ztmp, tau

def tau_CMB(sim):
    """
    Compute CMB optical depth history.
    
    Parameters
    ----------
    sim : instance of ares.simulations.MultiPhaseMedium or Global21cm
    
    
    
    """
    
    # Sort in ascending order
    zall = sim.history['z'][-1::-1]
    
    zmin = zall.min()
    
    # Create array of redshifts from z=0 up to final snapshot
    ztmp = np.arange(0, zmin+0.001, 0.001)
    
    QHII = sim.history['cgm_h_2'][-1::-1]
    
    if 'igm_h_2' in sim.history:
        xHII = sim.history['igm_h_2'][-1::-1]
    else:
        xHII = 0.0
        
    nH = sim.grid.cosm.nH(zall)
    dldz = sim.grid.cosm.dldz(zall)

    integrand = (QHII + (1. - QHII) * xHII) * nH

    # Add helium optical depth
    if 'igm_he_1' in sim.history:
        QHeII = sim.history['cgm_h_2'][-1::-1]
        xHeII = sim.history['igm_he_2'][-1::-1]
        xHeIII = sim.history['igm_he_3'][-1::-1]        
    else:
        QHeII = sim.history['cgm_h_2'][-1::-1]
        xHeII = 0.0
        xHeIII = 0.0
        
    nHe = sim.grid.cosm.nHe(zall)    
        
    integrand += (QHeII + (1. - QHeII) * xHeII + 2. * xHeIII) \
        * nHe

    integrand *= sigma_T * dldz

    tau = cumtrapz(integrand, zall, initial=0)
        
    tau[zall > 100] = 0.0    
    
    # Make no arrays that go to z=0
    zlo, tlo = tau_post_EoR(sim)
            
    tau_tot = tlo[-1] + tau

    zarr = np.concatenate((zlo, zall))
    tau_all_z = np.concatenate((tlo, tau_tot))
    
    return zarr, tau_all_z


def get_attribute(s, ob):
    """
    Break apart a string `s` and recursively fetch attributes from object `ob`.
    """
    spart = s.partition('.')

    f = ob
    for part in spart:
        if part == '.':
            continue
        
        f = f.__getattribute__(part)
        
    return f

def split_by_sign(x, y):
    """
    Split apart an array into its positive and negative chunks.
    """

    splitter = np.diff(np.sign(y))

    if np.all(splitter == 0):
        ych = [y]
        xch = [x]
    else:
        splits = np.atleast_1d(np.argwhere(splitter != 0).squeeze()) + 1
        ych = np.split(y, splits)
        xch = np.split(x, splits)

    return xch, ych
    
def plot_by_chunk(x, y, ax, **kwargs):
    xx, yy = split_by_sign(x, y)
    
    if 'label' in kwargs:
        label = kwargs['label']
        del kwargs['label']
    else:
        label = None
    
    apply_label = False
    applied_label = False
    for i, chunk in enumerate(yy):
        pos = np.all(chunk > 0)
        
        apply_label = pos
        
        if apply_label and applied_label:
            apply_label = False
        
        if i == (len(yy) - 1) and (not applied_label):
            apply_label = True
            
                
        ax.loglog(xx[i], np.abs(yy[i]), lw=3 if pos else 1,
            label=label if apply_label else None, **kwargs)
    
        if apply_label:
            applied_label = True
            
    
    return ax
