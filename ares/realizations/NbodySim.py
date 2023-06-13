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
    def __init__(self, Lbox=256, dims=128, zlim=(0.2, 2), verbose=True,
        prefix='ares_mock', **kwargs):
        """
        Initialize a galaxy population from log-normal density fields generated
        from the matter power spectrum.

        Parameters
        ----------
        Lbox : int, float
            Linear dimension of volume in Mpc/h.
        dims : int
            Number of grid points in each dimension, so total number of
            grid elements per co-eval cube is dims**3.
        zlim : tuple
            Defines domain size along line of sight, zlim[0] <= z < zlim[1].
        kwargs : dictionary
            Set of parameters that defines an ares.simulations.Simulation.

        """
        self.Lbox = Lbox
        self.dims = dims
        self.zlim = zlim
        self.verbose = verbose
        self.kwargs = kwargs
        self.prefix = prefix

    def get_catalog(self, zlim=None, logmlim=(11,12), idnum=0, verbose=True):
        """
        Get a galaxy catalog in (RA, DEC, redshift) coordinates.

        .. note :: This is essentially a wrapper around `_get_catalog_from_coeval`,
            i.e., we're just figuring out how many chunks are needed along the
            line of sight and re-generating the relevant cubes.

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

        if zlim is None:
            zlim = self.zlim

        zmin, zmax = zlim
        mmin, mmax = 10**np.array(logmlim)

        # First, get full domain info
        ze, zmid, Re = self.get_domain_info(zlim=self.zlim, Lbox=self.Lbox)
        Rc = bin_e2c(Re)
        dz = np.diff(ze)

        # Deterministically adjust the random seeds for the given mass range
        # and redshift range.
        fmh = int(logmlim[0] + (logmlim[1] - logmlim[0]) / 0.1)

        seeds = None#self.seed_rho * np.arange(1, len(zmid)+1)

        # Figure out if we're getting the catalog of a single chunk
        chunk_id = None
        for i, Rlo in enumerate(Re[0:-1]):
            zlo, zhi = ze[i:i+2]

            if (zlo == zlim[0]) and (zhi == zlim[1]):
                chunk_id = i
                break

        ##
        # Print-out information about FOV
        # arcmin / Mpc -> deg / Mpc
        theta_max = self.sim.cosm.ComovingLengthToAngle(zmin, 1) \
            * (self.Lbox / self.sim.cosm.h70) / 60.
        theta_min = self.sim.cosm.ComovingLengthToAngle(zmax, 1) \
            * (self.Lbox / self.sim.cosm.h70) / 60.

        pbar = ProgressBar(Rc.size, name=f"lc(z>={zmin},z<{zmax})",
            use=chunk_id is None)
        pbar.start()

        ct = 0
        zlo = zmin * 1.
        for i, Rlo in enumerate(Re[0:-1]):
            pbar.update(i)

            zlo, zhi = ze[i:i+2]

            if chunk_id is not None:
                if i != chunk_id:
                    continue

            if (zhi <= zlim[0]) or (zlo >= zlim[1]):
                continue

            #if (zlo, zhi, mmin) in self._cache_cats:
            #    _ra, _de, _red, _m = self._cache_cats[(zlo, zhi, mmin)]
            #else:
            halos = self.get_halo_population(z=zmid[i],
                mmin=mmin, mmax=mmax, verbose=verbose, idnum=idnum)

            if halos[0] is None:
                ra = dec = red = mass = None
                continue

            _ra, _de, _red = self._get_catalog_from_coeval(halos, z0=zlo)
            _m = halos[-1]

            okr = np.logical_and(_ra <  0.5 * theta_min,
                                 _ra > -0.5 * theta_min)
            okd = np.logical_and(_de <  0.5 * theta_min,
                                 _de > -0.5 * theta_min)
            ok = np.logical_and(okr, okd)

                # Cache intermediate outputs too!
                #self._cache_cats[(zlo, zhi, mmin)] = \
                #    _ra[ok==1], _de[ok==1], _red[ok==1], _m[ok==1]

                #_ra, _de, _red, _m = self._cache_cats[(zlo, zhi, mmin)]

            if ct == 0:
                ra = _ra.copy()
                dec = _de.copy()
                red = _red.copy()
                mass = _m.copy()
            else:
                ra = np.hstack((ra, _ra))
                dec = np.hstack((dec, _de))
                red = np.hstack((red, _red))
                mass = np.hstack((mass, _m))

            ct += 1

            del _ra, _de, _red, _m, halos
            gc.collect()

        pbar.finish()

        #self._cache_cats[(zmin, zmax, mmin)] = ra, dec, red, mass

        return ra, dec, red, mass

        m, ra, dec, red = self.get_halo_population(z)

        return ra, dec, red

    def get_halo_population(self, z, mmin=0, mmax=np.inf, verbose=False,
        idnum=0):
        """
        This returns "raw" halo data for a given redshift, i.e., we're just
        pulling halo masses and their positions in co-eval boxes.

        Returns
        -------
        Tuple containing (x, y, z, mass), where x, y, and z are halo positions
        in cMpc / h (between 0 and self.Lbox), and mass is in Msun.
        
        """
        m, xx, yy, zz = self.sim.pops[0].pf['pop_halos'](z).T

        ok = np.logical_and(m >= mmin, m < mmax)

        return xx[ok==1], yy[ok==1], zz[ok==1], m[ok==1]

    def get_density_field(self, z):
        return self.sim.pops[0].pf['pop_density'](z)
