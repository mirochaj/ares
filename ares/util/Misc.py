"""

Misc.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sun Oct 19 19:50:31 MDT 2014

Description:

"""
import os
import copy
import subprocess
import numpy as np
from ..data import ARES
from .Stats import bin_e2c
from ..physics.Constants import c, erg_per_ev, h_p, E_LL, E_LyA

numeric_types = [int, float, np.int64, np.int32, np.float64, np.float32]

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
                cmd_line_kwargs[pre] = True if post == 'True' else False
            else:
                cmd_line_kwargs[pre] = str(post)
        elif post[0] == '[':
            vals = post[1:-1].split(',')
            cmd_line_kwargs[pre] = np.array([float(val) for val in vals])
        else:
            try:
                cmd_line_kwargs[pre] = float(post)
            except ValueError:
                # strings with underscores will return False from isalpha
                cmd_line_kwargs[pre] = str(post)

    return cmd_line_kwargs

def get_hash(repo_path=ARES, repo_env=None):
    """
    Return the unique git hash associated with the HEAD of some repository.

    This intended to be used to save the current version of some code you're
    using with any output files to help with debugging later. For example,
    I have some output files for a calculation I've done that change over time,
    indicating a problem/development, and I want to know precisely when that
    change happened. In practice, I usually save the output of this function
    as metadata in an hdf5 file (e.g., as an attribute of some dataset or
    as a dataset of its own).

    Parameters
    ----------
    repo_path : str, None
        Absolute path to root directory of repo of interest (where .git lives)
    repo_env : str, None
        Name of environment variable that points to repo (again, the root
        directory where .git lives).

    Returns
    -------
    A string containing the unique hash of the current HEAD of git repo.

    Known Flaws
    -----------
    If you run your code with uncommitted changes, then this hash may not help
    you find a bug, as the bug could have been in your uncommitted changes.
    There's not really a good solution to this, other than to always run your
    code with a 'clean' install!

    """

    assert (repo_path is not None) or (repo_env is not None), \
        "Must supply path to git repo or environment variable that points to it."

    try:
        cwd = os.getcwd()

        if repo_env is not None:
            PATH = os.environ.get(repo_env)
        else:
            PATH = repo_path

        os.chdir(PATH)

        # git rev-parse HEAD
        pipe = subprocess.Popen(["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE)

        # Move back to where we were
        os.chdir(cwd)
    except Exception as err:
        print("Failure to obtain hash due to following error: {}".format(err))
        return 'unknown'

    return pipe.stdout.read().strip()

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

def get_rte_segments(Emin, Emax):
    """
    Break radiation field into chunks we know how to deal with.

    For example, ranges over which there is "sawtooth modulation" of the
    background from HI, HeI, and HeII absorption.

    Parameters
    ----------

    Returns
    -------
    List of band segments, each a tuple of the form (Emin/eV, Emax/eV).

    """

    # Pure X-ray
    if (Emin > E_LL) and (Emin > 4 * E_LL):
        return [(Emin, Emax)]

    bands = []

    # Check for optical/IR
    if (Emin < E_LyA) and (Emax <= E_LyA):
        bands.append((Emin, Emax))
        return bands

    # Emission straddling Ly-a -- break off low energy chunk.
    if (Emin < E_LyA) and (Emax > E_LyA):
        bands.append((Emin, E_LyA))

        # Keep track as we go
        _Emin_ = np.max(bands)
    else:
        _Emin_ = Emin

    # Check for sawtooth
    if _Emin_ >= E_LyA and _Emin_ < E_LL:
        bands.append((_Emin_, min(E_LL, Emax)))

    #if (abs(Emin - E_LyA) < 0.1) and (Emax >= E_LL):
    #    bands.append((E_LyA, E_LL))
    #elif abs(Emin - E_LL) < 0.1 and (Emax < E_LL):
    #    bands.append((max(E_LyA, E_LL), Emax))

    if Emax <= E_LL:
        return bands

    # Check for HeII
    if Emax > (4 * E_LL):
        bands.append((E_LL, 4 * E_LyA))
        bands.append((4 * E_LyA, 4 * E_LL))
        bands.append((4 * E_LL, Emax))
    else:
        bands.append((E_LL, Emax))

    return bands

def has_sawtooth(Emin, Emax):
    """
    Identify bands that should be split into sawtooth components.
    Be careful to not punish users unnecessarily if Emin and Emax
    aren't set exactly to Ly-a energy or Lyman limit.
    """

    has_sawtooth  = (abs(Emin - E_LyA) < 0.1) or (abs(Emin - 4 * E_LyA) < 0.1)
    has_sawtooth &= Emax > E_LyA

    return has_sawtooth

def get_rte_grid(zi, zf, nz=100, Emin=1., Emax=10.2, start_at_Emin=True):
    """
    Determine the grid of redshifts and photon energies that we'll evolve
    cosmic radiation backgrounds through.

    .. note :: The provided redshift range will be spanned *exactly*. In order
        to take advantage of this discretization scheme, perfectly spanning
        the redshift window of interest cannot simultaneously perfectly span the
        desired energy range. This is generally OK. Things to consider include,
        e.g., whether there's an emission line of interest at one end of the
        energy range, or whether one prefers better resolution at the lower
        or upper part of the range. See `start_hi` keyword argument below.

    Parameters
    ----------
    zi : int, float
        Initial redshift (high redshift). This is inclusive, i.e., the highest
        redshift included will be `zi` exactly.
    zf : int, float
        Final redshift (low redshift; zf < zi). This is inclusive, i.e., the
        lowest redshift included will be `zf` exactly.
    nz : int
        Number of gridpoints to use to sample the redshift axis.
    Emin : int, float
        Minimum photon energy to consider [eV].
    Emax : int, float
        Maximum photon energy to consider [eV].
    start_hi : bool
        Determines whether the energy grid is pinned to start at Emin
        (start_at_Emin=True) or Emax (start_at_Emin=False).

    Returns
    -------
    Tuple containing (array of redshifts, array of energies).

    """

    N = num_freq_bins(nz, zi=zi, zf=zf, Emin=Emin, Emax=Emax)

    x = np.logspace(np.log10(1 + zf), np.log10(1 + zi), nz)
    z = x - 1.
    R = x[1] / x[0]

    if start_at_Emin:
        E = Emin * R**np.arange(N)
    else:
        E = np.flip(Emax * R**-np.arange(N), 0)

    return z, E

def get_rte_bands(zi, zf, nz=100, Emin=1., Emax=10.2, start_at_Emin=True,
    E_user=None):
    """
    From an array of (potentially) unevenly spaced wavelengths [Angstroms],
    construct a series of bands.

    Returns
    -------
    Tuple containing (band edges [Angstroms], band width [Hz])
    """

    # `E` will always be ascending.
    if E_user is not None:
        E = E_user
    else:
        z, E = get_rte_grid(zi=zi, zf=zf, nz=nz, Emin=Emin, Emax=Emax,
            start_at_Emin=start_at_Emin)

    freqs = E * erg_per_ev / h_p
    waves = c * 1e8 / freqs

    if len(waves) == 1:
        return [None], np.ones(1)

    is_asc = np.all(np.diff(waves) > 0)
    assert not is_asc, "`waves` should be in descending order."

    waves_asc = waves[::-1]

    # Set upper edge of all bands by halving distance between centers
    bands_up = [waves_asc[i] + 0.5 * (waves_asc[i+1] - waves_asc[i]) \
        for i in range(len(waves) - 1)]

    b_up = waves_asc[-1] + 0.5 * (waves_asc[-1] - bands_up[-1])

    bands_lo = copy.deepcopy(bands_up)
    # Insert lowest band
    b_lo = waves_asc[0] - 0.5 * (bands_up[0] - waves_asc[0])

    bands_lo.insert(0, b_lo)
    bands_up.append(b_up)

    bands = np.array([bands_lo, bands_up]).T[::-1,::-1]
    dfreq = np.abs(np.diff(c * 1e8 / bands, axis=1))

    return bands, dfreq

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

def get_field_from_catalog(field, pos, Lbox, dims=512, mesh=None,
    weight_by_field=True, by_volume=True):
    """
    Convert a catalog, i.e., a list of (lum, x, y, z), to luminosity
    (or whatever) on a mesh.

    .. note :: If you're applying some threshold like Mmin, do so
        BEFORE running this routine. In this case, ``catalog`` should
        be a numpy masked array.

    .. note :: If weight_by_field == False, the units of the output
        will just number of halos per voxel, i.e., independent
        of the field.

    Parameters
    ----------
    catalog : np.ndarray
        Should have shape (Ngalaxies, 4)
    dims : int
        Linear dimensions of box to create. Can alternatively provide
        desired grid resolution (in Mpc / h) via ``mesh`` keyword
        argument (see below).
    mesh : int, float
        If supplied, should be the linear dimension of voxels used
        in histogram [Mpc / h].
    weight_by_field : bool
        If True, will weight by field (density or luminosity usually)
    by_volume : bool
        If True, will divide by voxel volume so field has units of
        x / cMpc^3, where x = whatever the field is (e.g., mass, luminosity).
        Otherwise, units will be the same as the input array. We generally
        set this to True for things like halo mass density, and False
        for things like the total ionizing photon output, which we want
        as an absolute photon production rate, not production rate density.

    """

    if mesh is None:
        mesh = Lbox / float(dims)

    xe = np.arange(0, Lbox+mesh, mesh)
    ye = np.arange(0, Lbox+mesh, mesh)
    ze = np.arange(0, Lbox+mesh, mesh)

    _x, _y, _z = pos.T

    data = np.array([_x, _y, _z]).T
    hist, edges = np.histogramdd(data, bins=[xe, ye, ze],
        weights=field if weight_by_field else None, density=False)

    if by_volume:
        hist /= mesh**3

    return bin_e2c(xe), hist
