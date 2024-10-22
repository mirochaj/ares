"""

NbodySim.py

Author: Jordan Mirocha
Affiliation: Jet Propulsion Laboratory
Created on: Sat Dec  3 14:28:58 PST 2022

Description:

"""

import gc
import numpy as np
from ..util import ProgressBar
from .LightCone import LightCone
from scipy.integrate import cumtrapz
from ..simulations import Simulation
from scipy.interpolate import interp1d
from ..util.Stats import bin_c2e, bin_e2c
from ..physics.Constants import cm_per_mpc

try:
    import powerbox as pbox
except ImportError:
    pass

class NbodySim(LightCone): # pragma: no cover
    def __init__(self, model_name, catalog, verbose=True, base_dir='nbody_mock',
        fxy=None, fov=None, Lbox=0, dims=0, mem_concious=False,
        seed_halo_occ=None, seed_nsers=None, seed_pa=None,
        zmin=0.07, zmax=1.4, zchunks=None, **kwargs):
        """
        Initialize a galaxy population from a simulated halo lightcone.

        Parameters
        ----------
        catalog : tuple

            First element: Filename prefix.
            Second element: indices in each output file corresponding to
                (RA, DEC, z, log10(Mhalo/Msun)).
            Third element: Array of redshift chunks at which we have
                saved the catalog.
        fov : tuple
            Can restrict sky area to patch from fov[0] <= RA < fov[1] and
            fov[2] <= DEC < fov[3]. If None, will return whole dataset.
        """

        self.verbose = verbose
        self.kwargs = kwargs
        self.base_dir = base_dir
        self.model_name = model_name
        self.fxy = fxy
        self.fov = fov
        self.Lbox = Lbox
        self.dims = dims
        self.mem_concious = mem_concious
        self.zmin = zmin
        self.zmax = zmax
        self.zlim = zmin, zmax
        self.zchunks = zchunks

        x, y = self.fxy
        self.fbox = x - 0.5 * fov, x + 0.5 * fov, \
                    y - 0.5 * fov, y + 0.5 * fov
        self.seed_halo_occ = seed_halo_occ
        self.seed_nsers = seed_nsers
        self.seed_pa = seed_pa

        # No need for these -- N-body sim does it for us
        self.seed_rho = -np.inf
        self.seed_halo_mass = -np.inf
        self.seed_halo_pos = -np.inf

        self.prefix, self.indices, self.zchunks = catalog

    def get_catalog(self, zlim=None, logmlim=None, popid=0, verbose=True):
        """
        Get a galaxy catalog in (RA, DEC, redshift) coordinates.

        Parameters
        ----------
        zlim : tuple
            Restrict redshift range to be between:

                zlim[0] <= z < zlim[1].

        logmlim : tuple
            Restrict halo mass range to be between:

                10**logmlim[0] <= Mh/Msun 10**logmlim[1]

        Returns
        -------
        A tuple containing (ra, dec, redshift, <mass or magnitudes or SFR or w/e>)

        """

        ##
        # First, figure out bounding redshift chunks.
        if zlim is not None:
            zlo, zhi = zlim
            ilo = np.digitize(zlo, self.zchunks[:,0]) - 1
            ihi = np.digitize(zhi, self.zchunks[:,0]) - 1
        else:
            zlo, zhi = self.zchunks[0,0], self.zchunks[-1,-1]
            ilo = 0
            ihi = self.zchunks.shape[0] - 1

        ##
        # Read at least one chunk. Implies that supplied `zlim` is smaller than
        # our chunks, so ilo==ihi.
        ihi = max(ihi, ilo+1)

        # Loop over chunks, read-in data
        N = 0
        for i in range(ilo, ihi):
            z1, z2 = self.zchunks[i]
            fn = f"{self.prefix}_{z1:.2f}_{z2:.2f}.txt"

            ##
            # Hack out galaxies outsize `zlim`.
            # `data` will be (number of halos, number of fields saved)
            _data = np.loadtxt(fn, usecols=self.indices)

            if verbose:
                print(f"! Loaded {fn}.")

            ##
            # Isolate halos in requested mass range.
            if logmlim is not None:
                okM = np.logical_and(_data[:,-1] >= logmlim[0],
                                     _data[:,-1] <  logmlim[1])
            else:
                okM = 1
            ##
            # Isolate halos in right z range.
            # (should be all except potentially at edges of domain).
            if zlim is not None:
                okz = np.logical_and(_data[:,-2] >= zlim[0],
                                     _data[:,-2] <  zlim[1])
            else:
                okz = 1

            ##
            # [optional] isolate halos in desired sky region.
            if self.fov is not None:
                okp = np.logical_and(_data[:,0] >= self.fbox[0],
                                     _data[:,0] <  self.fbox[1])
                okp*= np.logical_and(_data[:,1] >= self.fbox[2],
                                     _data[:,1] <  self.fbox[3])
            else:
                okp = 1

            ##
            # Append to any previous chunk's data.
            if N == 0:
                data = _data[okM*okz*okp==1,:].copy()
            else:
                data = np.vstack((_data[okM*okz*okp==1,:], data))

            N += 1

            print('hi', N, sum(okM), sum(okz), sum(okp))


        ##
        # Return transpose, so users can run, e.g.,
        # >>> ra, dec, z, logm = <instance_name>.get_catalog()
        # First, need to 10** the halo masses.
        _x_, _y_, _z_, _m_ = data.T

        data = np.array([_x_, _y_, _z_, 10**_m_])

        return data
