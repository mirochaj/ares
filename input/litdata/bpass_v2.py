"""
Module for reading-in BPASS version 1.0 results.

Reference: Eldridge, JJ., and Stanway, E.R., 2009, MNRAS, 400, 1019

"""

import re, os
import numpy as np
from scipy.interpolate import interp1d
from eldridge2009 import _load as _load_bpass_v1
from ares.physics.Constants import h_p, c, erg_per_ev, g_per_msun, \
    s_per_yr, s_per_myr, m_H, Lsun

_input = os.getenv('ARES') + '/input/bpass_v2/SEDS'

metallicities = \
{
 '040': 0.040,
 '030': 0.030,
 '020': 0.020,
 '014': 0.014,
 '010': 0.010,
 '008': 0.008,
 '006': 0.006,
 '004': 0.004,
 '003': 0.003,
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

    assert kwargs['source_ssp'], "BPASS v2 only supports source_ssp=True."

    # All files share this prefix
    fn = 'spectra-'
    if kwargs['source_binaries']:
        fn += 'bin'
    else:
        fn += 'sin'

    # Assume Salpeter IMF
    if kwargs['source_imf'] in [1.35, 2.35, 'salpeter']:
        fn += '-imf135'
    elif kwargs['source_imf'] in ['Chabrier', 'chabrier', 'chab']:
        fn += '-imf_chab'
    else:
        # Assume it's a number
        fn += '-imf{}'.format(int(kwargs['source_imf']))

    fn += '_{}'.format(int(kwargs['source_imf_Mmax']))

    # Metallicity
    fn += '.z{!s}'.format(str(int(kwargs['source_Z'] * 1e3)).zfill(3))

    if kwargs['source_sed_degrade'] is not None:
        fn += '.deg{}'.format(kwargs['source_sed_degrade'])

    fn += '.dat'

    return _input + '/' + fn

def _load(**kwargs):
    """
    Return wavelengths, fluxes, for given set of parameters (at all times).
    """

    fn = _kwargs_to_fn(**kwargs)
    wavelengths, data, _fn = _load_bpass_v1(fn=fn, **kwargs)

    # Any special treatment needed?

    return wavelengths, data, _fn
