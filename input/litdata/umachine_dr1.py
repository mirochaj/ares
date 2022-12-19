"""
Behroozi et al. (2019) compilation of observed data.
"""

import os
import numpy as np
from ares.data import ARES

info = \
{
 'reference': 'Behroozi, Wechsler, Hearin, & Conroy, 2019, MNRAS, 488, 3143',
}

_input = ARES + '/input/umachine-data/umachine-dr1/observational_constraints'

def get_data(field, sources=None):
    """
    field options are 'smf', 'uvlf', 'ssfr', 'csfr', 'qf'
    """

    data = {}
    for fn in os.listdir(_input):
        if not fn.endswith(field):
            continue

        if field == 'csfr':
            src = fn[0:fn.find('.')]
        else:
            src = fn[0:fn.find('_')]

        if sources is not None:
            if src not in sources:
                continue

        data[src] = {}

        # First, read-in header only to figure out what we're dealing with.
        f = open(f"{_input}/{fn}", 'r')

        hdr = {}
        line = f.readline()
        while line.startswith('#'):
            name, colon, val = line[1:].partition(":")
            hdr[name] = val.strip()
            line = f.readline()
        f.close()

        # Special case: cosmic SFRD
        if hdr['type'].startswith('cosmic sfr'):
            zlo, zhi, logSFRD, errlo, errhi = \
                np.loadtxt(f"{_input}/{fn}", unpack=True)

            if type(zlo) in [int, float, np.float64]:
                zarr = np.array([[zlo], [zhi]]).T
                err = np.array([[errlo], [errhi]]).T
            else:
                zarr = np.array([zlo, zhi]).T
                err = np.atleast_2d(np.array([errlo, errhi])).T

            data[src] = zarr, logSFRD, err

            continue

        if 'zlow' not in hdr:
            print(f"Dunno what to do with {src}")
            continue

        ##
        # Everything else
        zlo, zhi = float(hdr['zlow']), float(hdr['zhigh'])
        zmean = (zlo + zhi) / 2.

        Mlo, Mhi, phi, errlo, errhi = np.loadtxt(f"{_input}/{fn}",
            unpack=True)

        if np.any(phi < 0):
            phi_is_log = True
        else:
            phi_is_log = False

        xerr = (Mhi - Mlo) / 2.
        M = (Mhi + Mlo) / 2.

        if phi_is_log:
            yerr = np.array([errlo, errhi]).T
        else:
            yp = np.log10(phi + errhi) - np.log10(phi)
            ym = np.log10(phi) - np.log10(phi - errlo)
            yerr = np.array([ym, yp])
            if yerr.ndim == 1:
                yerr = np.atleast_2d(yerr).T

        x = np.array([Mlo, Mhi]).T

        data[src][(zlo, zhi)] = \
            np.atleast_2d(x), phi, np.atleast_2d(yerr)

    ##
    # Done
    return data
