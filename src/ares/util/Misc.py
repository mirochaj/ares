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

letters = list('abcdefg')
numeric_types = [int, float, np.int64, np.int32, np.float64, np.float32]

_hmod_terms = 'shot', '1h', '2h'
_hmod_labels = r'$I_1 x I_2 (\nu_1 = \nu_2)$', \
        r'$I_1 x I_2 (\nu_1 \neq \nu_2)$', \
        r'$g \times I$', '$gg$'

def get_hmod_elements(sim, fluctuation_type=0, redundancy_convention='lower'):
    """
    Figure out which terms in "inter-population cross-correlation matrix"
    should be non-zero.

    There are four types of fluctuations:
    1. Intensity autos
    2. Galaxy catalog / intensity crosses
    3. Galaxy autos
    4. Intensity internal crosses

    Note that there's not really an analog of internal crosses for 
    galaxies. In principle their could be (e.g., ELG x LRG), but I 
    don't think we'll ever do that.

    Parameters
    ----------
    sim : object
        An ares.simulations.Simulation instance.
    fluctuation_type : int
        Corresponding to items 1-4 listed above.
    redundancy_convention : str
        Can be 'lower' or 'upper'. Controls whether we keep only the lower
        or upper diagonal of the matrix in cases where it is symmetric,
        which at this stage is really just for 2-halo intensity autos.

    Returns
    -------
    A 3-D array containing the interpop cross-corr matrix (final two axes)
    for shot, 1-h, and 2-h terms (first axis of length 3).
    
    """

    results = np.zeros([3] + [len(sim.pops)]*2)
    
    for j, term in enumerate(_hmod_terms):

        has_power = np.zeros([len(sim.pops)]*2)
        for k1, pop1 in enumerate(sim.pops):
            
            # No shot noise for diffuse emission sources
            if (term == 'shot') and pop1.is_diffuse:
                continue
            
            for k2, pop2 in enumerate(sim.pops):

                if (term == 'shot') and pop2.is_diffuse:
                    continue
                
                # For intensity autos, upper and lower halves
                # of matrix are redundant. Keep upper only.
                if fluctuation_type == 0:
                    if redundancy_convention == 'lower' and (k2 > k1):
                        continue
                    elif redundancy_convention == 'upper' and (k2 < k1):
                        continue
                    
                # For internal cross spectrum, 
                
                # For galaxy-intensity cross or galaxy autos, 
                # diffuse sources don't contribute.
                if (fluctuation_type == 1) and (pop1.is_diffuse):
                    continue
                if (fluctuation_type == 2) and (pop1.is_diffuse or pop2.is_diffuse):
                    continue
                                
                # OK
                if term == 'shot':
                    has_power[k1,k2] = \
                        pop1.id_num_actual == pop2.id_num_actual
                elif term == '1h':
                    has_power[k1,k2] = \
                        (pop1.is_central_pop + pop2.is_central_pop) in [0,3]
                elif term == '2h':
                    has_power[k1,k2] = 1
                else:
                    raise NotImplementedError(f'Unknown term={term}')
        
    
        # Save
        results[j,:,:] = has_power

    return results

def get_wave_or_equivalent(x_in, units, units_out):
    """
    Convert between photon wavelength, energy, and frequency.

    Parameters
    ----------
    x_in : int, float, np.ndarray
        Array of values that we'd like convert to different units.
    units : str
        Units of `x_in`, e.g., 'cm', 'ang', 'mic', 'hz', 'ghz', 'ev', 'keV'.
    units_out : str
        Units we'd like to convert `x_in` to.

    Returns
    -------
    Input array `x_in` converted to output units `units_out`.

    """
    if units.lower() == units_out.lower():
        return x_in
    
    if type(x_in) in [tuple, list]:
        x_in = np.array(x_in)
    
    ##
    # Start by convert input unit to cm
    if units.lower() == 'cm':
        x_cm = x_in
    elif units.lower().startswith('ang'):
        x_cm = x_in * 1e-8
    elif (units.lower() == 'um') or units.lower().startswith('mic'):
        x_cm = x_in * 1e-4
    elif units.lower() == 'hz':
        x_cm = c / x_in
    elif units.lower() == 'mhz':
        x_cm = c / (x_in * 1e6)
    elif units.lower() == 'ghz':
        x_cm = c / (x_in * 1e9)
    elif units.lower() == 'ev':
        x_cm = h_p * c / (x_in * erg_per_ev)
    elif units.lower() == 'kev':
        x_cm = h_p * c / (x_in * 1e3 * erg_per_ev)
    else:
        raise NotImplemented(f'Unrecognized input unit={units}')
    
    if units_out.lower() == 'cm':
        return x_cm
    elif units_out.lower().startswith('ang'):
        return x_cm * 1e8
    elif (units.lower() == 'um') or units_out.lower().startswith('mic'):
        return x_cm * 1e4
    elif units_out.lower() == 'hz':
        return c / x_cm
    elif units_out.lower() == 'mhz':
        return c / x_cm / 1e6
    elif units_out.lower() == 'ghz':
        return c / x_cm / 1e9
    elif units_out.lower() == 'ev':
        return h_p * c / x_cm / erg_per_ev
    elif units_out.lower() == 'kev':
        return h_p * c / x_cm / erg_per_ev / 1e3
    else:
        raise NotImplemented(f'Unrecognized input unit={units}')
    
def get_pop_info(popid):
    """
    Parse `popid`, as we (as of March 2025) allow non-integer IDs.

    Parameters
    ----------
    popid : int, str, tuple
        For old-school ARES calculations (burn), this would just be an integer
        used to index some ares.simulations.Simulation.pops list. Now, we can
        pass things like '2a', which generally means 'satellite galaxies that
        belong to population 0' (the 'a' maps back to pop 0, 'b' to pop 1, etc).
        This is a little confusing mixing numbers and letters, but I think it's
        less confusing than indicating '2a' as '20', or requiring users to
        provide a tuple, e.g., (2, 0) or (2, 'a').

    Returns
    -------
    Tuple containing (ARES popid, parent popid [if applicable], pop name as str).

    """

    # In this case, 'classic' behavior: just an integer, i.e.,
    # central galaxies.
    if (type(popid) == int) or popid.isnumeric():
        return int(popid), int(popid), str(popid)

    if type(popid) == tuple:
        assert popid[1] < popid[0]
        if type(popid[1]) == str:
            s = letters.index(popid[1])
        else:
            s = letters[popid[1]]

        return popid[0], popid[1], f'{int(popid[0])}{s}'

    if type(popid) == str:
        return int(popid[0]), int(letters.index(popid[1])), popid

    raise NotImplemented('help')

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

def get_band_edges(waves):
    assert np.all(np.diff(waves) > 0), \
        "Must supply wavelengths in ascending order."

    # Set upper edge of all bands by halving distance between centers
    bands_up = [waves[i] + 0.5 * (waves[i+1] - waves[i]) \
        for i in range(len(waves) - 1)]

    b_up = waves[-1] + (waves[-1] - bands_up[-1])

    bands_lo = copy.deepcopy(bands_up)
    # Insert lowest band
    b_lo = waves[0] - (bands_up[0] - waves[0])

    bands_lo.insert(0, b_lo)
    bands_up.append(b_up)

    bands = np.array([bands_lo, bands_up]).T

    return bands

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

    bands = get_band_edges(waves_asc)[::-1,:]

    ## Set upper edge of all bands by halving distance between centers
    #bands_up = [waves_asc[i] + 0.5 * (waves_asc[i+1] - waves_asc[i]) \
    #    for i in range(len(waves) - 1)]

    #b_up = waves_asc[-1] + 0.5 * (waves_asc[-1] - bands_up[-1])

    #bands_lo = copy.deepcopy(bands_up)
    ## Insert lowest band
    #b_lo = waves_asc[0] - 0.5 * (bands_up[0] - waves_asc[0])

    #bands_lo.insert(0, b_lo)
    #bands_up.append(b_up)

    #bands = np.array([bands_lo, bands_up]).T[::-1,::-1]
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
