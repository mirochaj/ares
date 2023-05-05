"""

ReadData.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Aug 27 10:55:19 MDT 2014

Description:

"""

import os
import re
import sys
import glob
import numpy as np

from ..data import ARES
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
            data[key] = np.array(data[key], dtype=float).squeeze()
        else:
            data[key] = np.array(data[key], dtype=float)

    return data

def concatenate(lists):
    return np.concatenate(lists, axis=0)
