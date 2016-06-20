"""
Ueda, Y., Akiyama, M., Ohta, K., & Miyaji, T. 2003, ApJ, 598, 886

Notes
-----
There are three different models here:
(1) Pure Luminosity Evolution (`ple`)
(2) Pure Density Evolution (`pde`)
(3) Luminosity Dependent Density Evolution (`ldde`)

The cosmology assumed was (H0, Om, Ol) = (70, 0.3, 0.7)

"""

import numpy as np

default_evolution = 'ldde'

qsolf_info = \
{
 'logLmin': 41.5,
 'logLmax': 46.5,
 'band': (2., 10.),
 'band_units': 'keV',
}    

qsolf_ple_pars = \
{
 'A': 14.1e-6,
 'logLstar': 43.66,
 'gamma1': 0.82,
 'gamma2': 2.37,
 'p1': 2.7,     
 'p2': 0.0,
 'zc': 1.15,
 'evolution': 'ple',
}

qsolf_ple_err = \
{
 'A': 1.0e-6,
 'logLstar': 0.17,
 'gamma1': 0.13,
 'gamma2': 0.16,
 'p1': 0.21,             # should be +0.17/-0.25
 'p2': 1e-10,            # held fixed
 'zc': 0.145,            # should be +0.2/-0.07
}

qsolf_pde_pars = \
{
 'A': 2.64e-6,
 'logLstar': 44.11,
 'gamma1': 0.93,
 'gamma2': 2.23,
 'p1': 4.2,     
 'p2': 0.0,
 'zc': 1.14,    
 'evolution': 'pde',
}

qsolf_pde_err = \
{
 'A': 0.18e-6,
 'logLstar': 0.23,
 'gamma1': 0.13,
 'gamma2': 0.15,
 'p1': 0.32,             
 'p2': 1e-10,            # held fixed
 'zc': 0.145,            # should be +0.13/-0.16
}

qsolf_ldde_pars = \
{
 'A': 5.04e-6,
 'logLstar': 43.94,
 'gamma1': 0.86,
 'gamma2': 2.23,
 'p1': 4.23,     
 'p2': -1.5,
 'zc': 1.9,    
 'logLa': 44.6,
 'alpha': 0.335, 
 'evolution': 'ldde',
}

qsolf_ldde_err = \
{
 'A': 0.33e-6,
 'logLstar': 0.23,       # Should be +0.21/-0.26
 'gamma1': 0.15,
 'gamma2': 0.13,
 'p1': 0.39,             
 'p2': 1e-10,            # held fixed
 'zc': 1e-10,            # held fixed
 'logLa': 1e-10,         # held fixed
 'alpha': 0.07,
}

kwargs_by_evolution = \
{
 'ple': qsolf_ple_pars, 
 'pde': qsolf_pde_pars,
 'ldde': qsolf_ldde_pars,
} 

errs_by_evolution = \
{
 'ple': qsolf_ple_err, 
 'pde': qsolf_pde_err,
 'ldde': qsolf_ldde_err,
} 

def _parse_kwargs(**kwargs):
    if not kwargs:
        kwargs = kwargs_by_evolution[default_evolution]
    elif 'evolution' in kwargs:
        kw = kwargs_by_evolution[kwargs['evolution']]
        kw.update(kwargs)
        kwargs = kw  
    elif 'evolution' not in kwargs:
        kwargs['evolution'] = default_evolution
        
    return kwargs

_eofz_f1 = lambda z, p1: (1. + z)**p1
_eofz_f2 = lambda z, p1, p2, zc: _eofz_f1(zc, p1) * ((1. + z) / (1. + zc))**p2

def _zc_of_L(L, **kwargs):
    """
    Compute cutoff redshift for luminosity-dependent density evolution.
    """
        
    La = 10**kwargs['logLa']

    if L < La:
        zc_ast = kwargs['zc'] * (L / La)**kwargs['alpha']
    elif L >= La:
        zc_ast = kwargs['zc']
        
    return zc_ast
    
def _evolution_factor_pde(z, **kwargs):
    """
    Pure density evolution model.
    """
        
    if z < kwargs['zc']:
        return _eofz_f1(z, kwargs['p1'])
    else:
        return _eofz_f2(z, kwargs['p1'], kwargs['p2'], kwargs['zc'])

def _evolution_factor_ldde(z, L, **kwargs):

    try:
        kwargs['zc'] = _zc_of_L(L, **kwargs)
        eofz = _evolution_factor_pde(z, **kwargs)
    except ValueError:
        eofz = np.zeros_like(L)        
        zcarr = np.array(map(lambda LL: _zc_of_L(LL, **kwargs), L))
        for i, zcval in enumerate(zcarr):
            kwargs['zc'] = zcval
            eofz[i] = _evolution_factor_pde(z, **kwargs)
            
    return eofz
    
def _DoublePowerLaw(L, **kwargs):
    # Defaults from PDE model
    Lstar = 10**kwargs['logLstar']
    return kwargs['A'] / ((L / Lstar)**kwargs['gamma1'] \
        + (L / Lstar)**kwargs['gamma2'])

def LuminosityFunction(L, z, **kwargs):
    """
    Compute the 2-10 keV quasar luminosity function.
    
    Parameters
    ----------
    L : int, float, np.ndarray
        Luminosity of interest [erg / s]
    z : int, float
        Redshift of interest
        
    kwags['evolution'] : str
        "ple": Pure Luminosity Evolution (Eq. 11)
        "pde": Pure Density Evolution (Eq. 12)
        "ldde": Luminosity-Dependent Density Evolution (Eqs. 16-17)
        
    """
    
    kwargs = _parse_kwargs(**kwargs)
                
    if kwargs['evolution'] == 'ple':
        Lprime = L / _evolution_factor_pde(z, **kwargs)
        NofL = _DoublePowerLaw(Lprime, **kwargs)
    elif kwargs['evolution'] == 'pde':
        NofL = _DoublePowerLaw(L, **kwargs)
        NofL *= _evolution_factor_pde(z, **kwargs)
    elif kwargs['evolution'] == 'ldde':
        NofL = _DoublePowerLaw(L, **kwargs)
        NofL *= _evolution_factor_ldde(z, L, **kwargs)         
    else:
        raise ValueError('Unrecognized evolution model: %s' \
            % kwargs['evolution'])
    
    return NofL
    
def Spectrum():
    pass
    

