"""
Module for reading-in BPASS version 2 results.

Reference: Eldridge, Stanway et al, 2017, PASA 34, 58

"""

#from .eldridge2017 import *
#from .eldridge2017 import _load # Must load explicitly

import re, os
import numpy as np
from ares.data import ARES
from scipy.interpolate import interp1d
from ares.physics.Constants import h_p, c, erg_per_ev, g_per_msun, s_per_yr, \
    s_per_myr, m_H, Lsun

_input = ARES + '/bpass_v2/v2.2.1/'

metallicities = \
{
 '040': 0.040,
 '020': 0.020,
 '008': 0.008,
 '004': 0.004,
 '002': 0.002,
 '001': 0.001,
}

sf_laws = \
{
 'continuous': 1.0,       # solar masses per year
 'instantaneous': 1e6,    # solar masses
}

imf_options = None

info = \
{
 'flux_units': r'$L_{\odot} \ \AA^{-1}$',
}

_log10_times = np.arange(6, 11.1, 0.1)
times = 10**_log10_times / 1e6            # Convert from yr to Myr

def _kwargs_to_fn(**kwargs):
    """
    Determine filename of appropriate BPASS lookup table based on kwargs.
    """

    # All files share this prefix
    fn = 'spectra'

    if kwargs['source_binaries']:
        fn += '-bin'
    else:
        fn += '-sin'

    # Only support Chabrier IMF for now
    fn += '-imf_chab100'

    assert kwargs['source_imf'].lower().startswith('chab')

    # Metallicity
    fn += '.z{!s}.dat'.format(str(int(kwargs['source_Z'] * 1e3)).zfill(3))

    if kwargs['source_sed_degrade'] is not None:
        fn += '.deg{}'.format(kwargs['source_sed_degrade'])

    return _input + '/' + fn

def _load(fn=None, **kwargs):
    """
    Return wavelengths, fluxes, for given set of parameters (at all times).
    """

    Zvals_l = list(metallicities.values())
    Zvals = np.sort(Zvals_l)

    # Interpolate
    if kwargs['source_Z'] not in Zvals_l:
        tmp = kwargs.copy()

        _fn = []
        spectra = []
        del tmp['source_Z']
        for Z in Zvals:
            _w1, _d1, fn = _load(source_Z=Z, **tmp)
            spectra.append(_d1.copy())
            _fn.append(fn)

        wavelengths = wave = _w1
        data = spectra

    # No interpolation necessary
    else:
        if fn is None:
            fn = _fn = _kwargs_to_fn(**kwargs)
        else:
            _fn = fn

        _raw_data = np.loadtxt(fn)

        data = np.array(_raw_data[:,1:])
        wavelengths = _raw_data[:,0]

        data *= Lsun

    return wavelengths, times, data, _fn
