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
        prefix='ares_mock', seed_rot=None, seed_trans=None,
        apply_rotations=False, apply_translations=False, **kwargs):
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
        self.apply_rotations = apply_rotations
        self.apply_translations = apply_translations

        self.seed_rot = seed_rot
        self.seed_tra = seed_trans

    def get_halo_population(self, z, mmin=0, mmax=np.inf, verbose=False,
        idnum=0, **seed_kwargs):
        """
        This returns "raw" halo data for a given redshift, i.e., we're just
        pulling halo masses and their positions in co-eval boxes.

        Returns
        -------
        Tuple containing (x, y, z, mass), where x, y, and z are halo positions
        in cMpc / h (between 0 and self.Lbox), and mass is in Msun.

        """

        # Should setup to keep box in memory until we change to a different z

        m, xx, yy, zz = self.sim.pops[0].pf['pop_halos'](z).T

        ok = np.logical_and(m >= mmin, m < mmax)

        return xx[ok==1], yy[ok==1], zz[ok==1], m[ok==1]

    def get_density_field(self, z):
        return self.sim.pops[0].pf['pop_density'](z)

    def get_seed_kwargs(self, chunk=None, logmlim=None):
        return {}
