"""
Ueda, Y., Akiyama, M., Ohta, K., & Miyaji, T. 2003, ApJ, 598, 886

Notes
-----
There are three different models here:
(1) Pure Luminosity Evolution (PLE)
(2) Pure Density Evolution (PDE)
(3) Luminosity Dependent Density Evolution (LDDE)

The cosmology assumed was (H0, Om, Ol) = (70, 0.3, 0.7)

"""

import numpy as np

qsolf_info = \
{
 'logLmin': 41.5,
 'logLmax': 46.5,
 'band': (2., 10.),
 'band_units': 'keV',
}    

qsolf_ple_pars = \
{
 'A': 14.1,
 'logLstar': 43.66,
 'gamma1': 0.82,
 'gamma2': 2.37,
 'p1': 2.7,     
 'p2': 0.0,
 'zc': 1.15,
}

qsolf_ple_err = \
{
 'A': 1.0,
 'logLstar': 0.17,
 'gamma1': 0.13,
 'gamma2': 0.16,
 'p1': 0.21,             # should be +0.17/-0.25
 'p2': 1e-10,            # held fixed
 'zc': 0.145,            # should be +0.2/-0.07
}

qsolf_pde_pars = \
{
 'A': 2.64,
 'logLstar': 44.11,
 'gamma1': 0.93,
 'gamma2': 2.23,
 'p1': 4.2,     
 'p2': 0.0,
 'zc': 1.14,    
}

qsolf_pde_err = \
{
 'A': 0.18,
 'logLstar': 0.23,
 'gamma1': 0.13,
 'gamma2': 0.15,
 'p1': 0.32,             
 'p2': 1e-10,            # held fixed
 'zc': 0.145,            # should be +0.13/-0.16
}

kwargs_by_model = {'ple': qsolf_ple_pars, 'pde': qsolf_pde_pars} 

_eofz_f1 = lambda z, p1: (1. + z)**p1
_eofz_f2 = lambda z, p1, p2, zc: _eofz_f1(zc, p1) * ((1. + z) / (1. + zc))**p2

def _evolution_factor_pde(z, p1=4.2, p2=0.0, zc=1.14, **kwargs):
    """
    Everything must be a scalar at the moment.
    """

    if z < zc:
        return _eofz_f1(z, p1)
    else:
        return _eofz_f2(z, p1, p2, zc)

#def _zc(L, zcst=1.9, La=_La):
#    pass

def DoublePowerLaw(L, A=2.64, logLstar=44.11, gamma1=0.93, gamma2=2.23, **kwargs):
    # Defaults from PDE model
    Lstar = 10**logLstar
    return A / ((L / Lstar)**gamma1 + (L / Lstar)**gamma2)

def LuminosityFunction(L, z=0., evolution='pde', **kwargs):
    """
    Compute the 2-10 keV quasar luminosity function.
    
    Parameters
    ----------
    L : int, float
    
    evolution : str
        
    """
    
    if not kwargs:
        kwargs = kwargs_by_model[evolution]
        
    if evolution == 'ple':
        Lprime = L / _evolution_factor_pde(z, **kwargs)
        NofL = DoublePowerLaw(Lprime, **kwargs)
    elif evolution == 'pde':
        NofL = DoublePowerLaw(L, **kwargs)
        NofL *= _evolution_factor_pde(z, **kwargs)
    elif evolution == 'ldde':
        raise NotImplemented('ldde help!')
    else:
        raise ValueError('Unrecognized evolution model: %s' % evolution)
    
    return NofL
    
def Spectrum():
    pass
    

