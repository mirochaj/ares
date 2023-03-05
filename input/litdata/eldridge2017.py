"""
BPASS version 2

Eldridge, Stanway et al, 2017, PASA 34, 58

Paper describes version 2.1. Latest updates in 2.2 described in
Stanway & Eldridge (2018).
"""

import re, os
import numpy as np
from ares.data import ARES
from scipy.interpolate import interp1d
from ares.physics.Constants import h_p, c, erg_per_ev, g_per_msun, s_per_yr, \
    s_per_myr, m_H, Lsun

_input = ARES + '/input/bpass_v2/'

metallicities = \
{
 '040': 0.040, '030': 0.040, '020': 0.020, '010': 0.010,
 '008': 0.008, '006': 0.006, '004': 0.004, '003': 0.003,
 '001': 0.002, '001': 0.001,
}

info = \
{
 'flux_units': r'$L_{\odot} \ \AA^{-1}$',
}

#_log10_times = np.arange(6, 10.1, 0.1)
#times = 10**_log10_times / 1e6            # Convert from yr to Myr
n = np.arange(1, 42)
times = 10**(6+0.1*(n-2)) / 1e6

def _kwargs_to_fn(**kwargs):
    """
    Determine filename of appropriate BPASS lookup table based on kwargs.
    """



    path = 'BPASSv2_imf{}'.format(str((kwargs['source_imf'] - 1)).replace('.', ''))
    path += '_{}'.format(str(int(kwargs['source_imf_Mmax'])))

    if kwargs['source_ssp']:
        path += '/OUTPUT_POP/'
    else:
        path += '/OUTPUT_CONT/'

    # All files share this prefix
    fn = 'spectra'

    if kwargs['source_binaries']:
        fn += '-bin'
    else:
        pass

    if kwargs['source_nebular'] == 1:
        fn += '+nebula'

    # Metallicity
    fn += '.z{!s}'.format(str(int(kwargs['source_Z'] * 1e3)).zfill(3))

    if kwargs['source_sed_degrade'] is not None:
        fn += '.deg{}'.format(kwargs['source_sed_degrade'])

    fn += '.dat'

    return _input + path + fn

def _load(**kwargs):
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
        fn = _fn = _kwargs_to_fn(**kwargs)

        _raw_data = np.loadtxt(fn)

        data = np.array(_raw_data[:,1:])
        wavelengths = _raw_data[:,0]

        data *= Lsun

    return wavelengths, data, _fn
