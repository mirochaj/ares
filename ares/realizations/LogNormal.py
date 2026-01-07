"""

LogNormal.py

Author: Jordan Mirocha
Affiliation: Jet Propulsion Laboratory
Created on: Sat Dec  3 14:28:42 PST 2022

Description:

"""

import gc
import numpy as np
from ..util import ProgressBar
from .LightCone import LightCone
from ..util.Misc import get_pop_info
from functools import cached_property
from scipy.interpolate import interp1d
from ..util.Stats import bin_c2e, bin_e2c
from ..physics.Constants import cm_per_mpc
from scipy.integrate import cumulative_trapezoid

try:
    import powerbox as pbox
except ImportError:
    pass

#try:
#    from numba import njit, prange
#
#    @njit
#    def _interp_linear(xx, x, y):
#        return np.interp(xx, x, y)
#
#    @njit
#    def _trapz(x, y):
#        return np.trapezoid(y, x=x)
#except ImportError:
#    pass


class LogNormal(LightCone): # pragma: no cover
    def __init__(self, model_name, Lbox=256, dims=128, zmin=0.05, zmax=2, verbose=True,
        seed_rho=None, seed_halo_mass=None, seed_halo_pos=None, seed_halo_occ=None,
        seed_rot=None, seed_trans=None, seed_profile=None, seed_sats=None,
        seed_lum=None,
        apply_rotations=False, apply_translations=False,
        bias_model=0, bias_params=None, bias_replacement=1, bias_within_bin=False,
        randomise_in_cell=True, base_dir='ares_mock', mem_concious=0,
        distribute_sats_spatially=True, profile_info=None,
        dz_max=0.01, lightcone_max_evol=np.inf, lightcone_corr=True, **kwargs):
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
        zmin, zmax : int, float
            Defines domain size along line of sight, zmin <= z < zmax.
        dz_max : float
            Will sub-sample along the line of sight direction in `dz_max` sized
            redshift increments, e.g., when computing fluxes from sources.
        kwargs : dictionary
            Set of parameters that defines an ares.simulations.Simulation.

        """
        self.Lbox = Lbox
        self.dims = dims
        self.zmin = zmin # Remember: just used for file-naming! More precision in zlim
        self.zmax = zmax # Remember: just used for file-naming! More precision in zlim
        self.zlim = (zmin, zmax)
        self.dz_max = dz_max
        self.lightcone_max_evol = lightcone_max_evol
        self.lightcone_corr = lightcone_corr
        self.seed_rho = seed_rho
        self.seed_halo_mass = seed_halo_mass
        self.seed_halo_pos = seed_halo_pos
        self.seed_halo_occ = seed_halo_occ
        self.seed_rot = seed_rot
        self.seed_tra = seed_trans
        self.seed_profile = seed_profile
        self.profile_info = profile_info
        self.seed_sats = seed_sats
        self.seed_lum = seed_lum
        self.apply_rotations = apply_rotations
        self.apply_translations = apply_translations
        self.distribute_sats_spatially = distribute_sats_spatially

        # Only used for NbodySimLC models
        self.zlayers = None

        self.fxy = (0., 0.)
        self.bias_model = bias_model
        self.bias_params = bias_params
        self.bias_replacement = bias_replacement
        self.bias_within_bin = bias_within_bin
        self.randomise_in_cell = randomise_in_cell
        self.verbose = verbose
        self.kwargs = kwargs
        self.base_dir = base_dir
        self.model_name = model_name

        self.mem_concious = mem_concious

        if self.bias_model > 0:
            assert self.bias_params is not None, \
                "Must provide `bias_params=[a,b]` for `bias_model>0`!"

        ##
        # Adjust upper bound in zlim based on box size!
        ze, zmid, Re = self.get_domain_info(zlim=(zmin, zmax), Lbox=self.Lbox)

        self.zlim = np.min(ze), np.max(ze)
        if verbose:
            print(f"# Overriding user-supplied zlim slightly to accommodate box size.")
            print(f"# Old zlim=({zmin:.3f},{zmax:.3f})")
            print(f"# New zlim=({self.zlim[0]:.3f},{self.zlim[1]:.3f})")
            print(f"# Number of co-eval layers: {zmid.size}")

        ##
        # Initialize caches here to avoid repeated hasattr calls
        self._cache_subhalo_cdf_ = {}
        self._cache_mgtm_ = {}

    def get_fov_from_L(self, z, Lbox):
        """
        Return FOV in degrees (single dimension) given redshift and Lbox in
        cMpc / h.
        """
        return (self.sim.cosm.get_angle_from_length_comoving(z, 1) / 60.) \
            * (Lbox / self.sim.cosm.h70)

    def get_L_from_fov(self, z, fov):
        """
        Get length scale corresponding to given field of view.

        .. note :: This is in co-moving Mpc, NOT cMpc / h!

        """
        ang_per_L = self.sim.cosm.get_angle_from_length_comoving(z, 1) / 60.

        return fov / ang_per_L

    def get_memory_estimate(self, zlim=None, logmlim=None, Lbox=None, dims=None):
        """
        Return rough estimate of memory needed vs. redshift in GB.

        .. note :: Assumes you need (x, y, z, mass) for each halo. Also, this
            is an estimate for the entire halo population -- the memory needed
            for a single population will be less if (for example) f_occ < 1.

        Returns
        -------
        Tuple containing (redshift bin centers, memory consumption at each z,
            cumulative memory consumption at z'<= z).

        """

        if Lbox is None:
            Lbox = self.Lbox

        if dims is None:
            dims = self.dims

        ze, zmid, Re = self.get_domain_info(zlim=zlim, Lbox=Lbox)
        mmin, mmax = 10**np.array(logmlim)

        #theta = [self.get_fov_from_L(_z_, Lbox=Lbox) for _z_ in zmid]

        mc = 0
        mem_z = [] # Memory for each redshift separately
        mem_c = [] # Cumulative
        for i, z in enumerate(zmid):
            iz = np.argmin(np.abs(self.halos.tab_z - z))
            ok = np.logical_and(self.halos.tab_M >= mmin,
                                self.halos.tab_M < mmax)

            m = self.halos.tab_M[ok==1]
            dndm = self.halos.tab_dndm[iz,ok==1]

            nall = cumulative_trapezoid(dndm * m, x=np.log(m), initial=0.0)
            nbar = np.trapezoid(dndm * m, x=np.log(m)) \
                 - np.exp(np.interp(np.log(mmin), np.log(m), np.log(nall)))

            # Memory to hold (x, y, z, m) for N halos
            N = nbar * (Lbox / self.sim.cosm.h70)**3
            mz = N * 8 * 4 # 4 is for (x, y, z, m)
            # Memory to hold density for dims**3 voxels
            mz += dims**3 * 8

            # Running tally over redshift
            mc += mz

            mem_z.append(mz)
            mem_c.append(mc)

        return zmid, np.array(mem_z) / 1e9, np.array(mem_c) / 1e9

    def get_nbar(self, z, ze, mmin, mmax=np.inf, fov=None, dz=None):
        """
        Return expected number density of halos at given z for given minimum
        mass.

        .. note :: This is the actual number density in cMpc^-3, not
            (cMpc / h)^-3!

        Parameters
        ----------
        z : int, float
            Redshift of interest.
        mmin : int, float
            Minimum mass threshold in solar masses.
        fov : int, float
            If not None, defines the field of view (single dimension) in deg.

        Returns
        -------
        If fov is None, returns the space density of objects in cMpc^-3. If
        fov is supplied, the returned value is the total number of objects
        in the volume defined by the field of view and dz interval.

        """

        #iz = np.argmin(np.abs(self.halos.tab_z - z))
        #ok = np.logical_and(self.halos.tab_M >= mmin,
        #                    self.halos.tab_M < mmax)
#
        #m = self.halos.tab_M
        #dndlnm = self.halos.tab_dndlnm[iz,:]
        #nbar = np.trapezoid(dndlnm[ok==1], x=np.log(m[ok==1]))

        nbar, mbar = self.get_mean_halo_density(z, ze, mmin, mmax)

        # Correct for FOV
        if (fov is not None) and (dz is not None):
            vol = self.get_survey_vol(z, fov, dz)
            nbar *= vol
        elif (fov is not None) or (dz is not None):
            raise ValueError("Must provide `fov` AND `dz` or neither!")

        return nbar
    
    def get_mean_halo_density(self, z, ze, mmin, mmax):
        """
        Generates two numbers (or fields): the expected halo number density and
        mass density.

        Parameters
        ----------

        Returns
        -------    
    
        """

        ##
        # In this case, we need to iterate through redshifts and re-compute HMF in
        # each slice.
        if self.lightcone_corr:
            #zlayers = self.get_redshift_layers(zlim=self.zlim)
            #zmids = zlayers.mean(axis=1)
            #iz = np.argmin(np.abs(zmid - zmids))

            Lpix = self.Lbox / float(self.dims)
            zpix_e, zpix_c, zpix_Re = \
                self.sim.cosm.get_lightcone_boundaries(ze, Lpix)

            # Each co-eval chunk is just self.dims long
            # Need corresponding redshift in each 
            m = self.halos.tab_M
            nb1d = np.zeros(self.dims)
            mb1d = np.zeros(self.dims)
            for i, _z_ in enumerate(zpix_c):
                iz = np.argmin(np.abs(self.halos.tab_z - _z_))
                ok = np.logical_and(self.halos.tab_M >= mmin,
                                    self.halos.tab_M < mmax)
    
                dndlnm = self.halos.tab_dndlnm[iz,:]
                nb1d[i] = np.trapezoid(dndlnm[ok==1], x=np.log(m[ok==1]))
                mb1d[i] = np.trapezoid(dndlnm[ok==1] * m[ok==1], x=np.log(m[ok==1]))
                
            ##
            # Make nbar and mbar 3-D just because they are likely to be multiplied
            # by a 3-D array (like density field) outside this routine.
            gridlike = np.ones([self.dims]*3)
            nbar = nb1d[None,None,:] * gridlike
            mbar = mb1d[None,None,:] * gridlike

        # Otherwise, just use midpoint of co-eval cube
        else:
            iz = np.argmin(np.abs(self.halos.tab_z - z))
            ok = np.logical_and(self.halos.tab_M >= mmin,
                                self.halos.tab_M < mmax)
    
            m = self.halos.tab_M
            dndlnm = self.halos.tab_dndlnm[iz,:]
            nbar = np.trapezoid(dndlnm[ok==1], x=np.log(m[ok==1]))
            mbar = np.trapezoid(dndlnm[ok==1] * m[ok==1], x=np.log(m[ok==1]))

        # Done
        return nbar, mbar

    def get_survey_vol(self, z, fov, dz):
        print("This could be more precise")
        Lperp = self.get_L_from_fov(z, fov)
        Lpara = self._mf.cosmo.comoving_distance(z+0.5*dz).to_value() \
              - self._mf.cosmo.comoving_distance(z-0.5*dz).to_value()
        return Lperp**2 * Lpara

    def get_ps_mm(self, z, k):
        """
        Compute the matter power spectrum. Just read from HMF.
        """

        if not hasattr(self, '_cache_ps'):
            self._cache_ps = {}

        if z in self._cache_ps:
            return self._cache_ps[z](k)

        iz = np.argmin(np.abs(self.halos.tab_z - z))

        power = interp1d(self.halos.tab_k_lin,
            self.halos.tab_ps_lin[iz,:], kind='cubic')

        self._cache_ps[z] = power

        return power(k)

    def get_density_field(self, z, seed=None):
        """
        This is a wrapper around `get_box` that will optionally perform a
        lightcone correction, i.e., account for the fact that for sufficiently
        large boxes there will be evolution in P(k) along the line of sight.
        """

        if not self.lightcone_corr:
            return self.get_box(z=z, seed=seed).delta_x()

        ##
        # If operating within a larger calculation (probably the case),
        # we need to be more careful. First, check how much P(k) evolves
        # over a single co-eval cube, and then generate two realizations if
        # necessary to form an interpolant along the line of sight.
        # First, get full domain info
        ze, zmid, Re = self.get_domain_info(zlim=self.zlim, Lbox=self.Lbox)
        zlayers = self.get_redshift_layers(zlim=self.zlim)

        iz = np.argmin(np.abs(z - zmid))
        if z < zlayers[iz,0]:
            iz -= 1

        zlo, zhi = zlayers[iz,:]

        # Just use a large-scale mode
        kbig = 1e-3

        Plo = self.get_ps_mm(zlo, kbig)
        Phi = self.get_ps_mm(zhi, kbig)

        if np.abs(Plo - Phi) / Plo < self.lightcone_max_evol:
            return self.get_box(z=z, seed=seed).delta_x()

        ##
        box_lo = self.get_box(z=zlo, seed=seed).delta_x()
        box_hi = self.get_box(z=zhi, seed=seed).delta_x()

        # Need redshifts of each voxel along LoS.
        Lpix = self.Lbox / float(self.dims)
        zpix_e, zpix_c, zpix_Re = \
            self.sim.cosm.get_lightcone_boundaries((zlo, zhi), Lpix)

        # Need to replace z-axis
        new_box = np.zeros_like(box_lo)
        for i, zz in enumerate(zpix_c):
            func = interp1d([zlo, zhi],
                np.array([box_lo[:,:,i], box_hi[:,:,i]]), axis=0)
            new_box[:,:,i] = func(zz)

        return new_box

    def get_box(self, z, seed=None):
        """
        Get a 3-D realization of a log-normal field at input redshift.

        Returns
        -------
        powerbox.powerbox.LogNormalPowerBox object, attribute `delta_x()` can
        be used to retrieve the box itself (in little delta).
        """

        if not hasattr(self, '_cache_box'):
            self._cache_box = {}

        if (z, seed) in self._cache_box:
            return self._cache_box[(z, seed)]

        power = lambda k: self.get_ps_mm(z, k)

        pb = pbox.LogNormalPowerBox(N=self.dims, dim=3, pk=power,
            boxlength=self.Lbox / self.sim.cosm.h70, seed=seed)

        # Only keep one box in memory at a time.
        if len(self._cache_box.keys()) > 0:
            del self._cache_box
            gc.collect()

            self._cache_box = {}

        self._cache_box[(z, seed)] = pb
        #print('NOT CACHING BOX')

        return pb

    def get_halo_positions(self, z, N, delta_x, m=None, seed=None,
        bias_model=None):
        """
        Generate a set of halo positions.

        Parameters
        ----------
        z : int, float
            Redshift -- only used for bias_model > 0.
        N : int, float
            If bias_model == 0, this is the expected number of halos in the
            volume.
            If bias_model == 1, this is the actual number, i.e., assumes we
            have already done a Poisson draw given <N>.

            If we've using lightcone corrections, this will be a 3-D array 
            containing the expected number of halos in each voxel.

        delta_x : np.ndarray
            Halo (over-)density on a 3-D grid.
        m : np.ndarray
            Array of halo masses [Msun]

        Returns
        -------
        Array containing 3-D positions of halos, shape (number of halos, 3).
        In Lbox / h [cMpc] units.
        """

        # Get all voxel positions
        args = [self.tab_xc] * 3
        X = np.meshgrid(*args, indexing='ij')

        # Make it look like a catalog, (N vox, 3).
        # Will modify this in subsequent steps.
        pvox = np.array([x.ravel() for x in X]).T

        # This is sneaky don't worry about it
        if bias_model is not None:
            _bias_model_ = bias_model
        else:
            _bias_model_ = self.bias_model

        # This is the same thing that powerbox is doing in
        # `create_discrete_sample`, just trying to have a unified call
        # sequence for other options here.
        if _bias_model_ == 0:

            n = N / (self.Lbox / self.sim.cosm.h70)**3

            # Expected number of halos in each cell, just scaling mean number
            # (over whole box) by 1+delta and voxel volume
            n_exp = n * (1. + delta_x) * (self.dx / self.sim.cosm.h70)**3

            # Actual number after Poisson draw
            np.random.seed(seed)
            n_act = np.random.poisson(n_exp)

            # Repeat position of each voxel N times, one for each halo that
            # lives there.
            pos = pvox.repeat(n_act.ravel(), axis=0)

        # In this case, we're increasing the probability that halos are drawn
        # from overdensities in a potentially halo mass dependent way.
        elif _bias_model_ == 1:

            n_act = m.size

            ivox = np.arange(pvox.shape[0])
            delta_flat = delta_x.ravel()

            # Right now, alpha(m) = p0 * (m / 1e12)**p1
            p0, p1 = self.bias_params

            if self.bias_within_bin:
                pbar = ProgressBar(m.size, name=f"pos(m)", use=True)
                pbar.start()

                alpha = p0 * (m / 1e12)**p1

                pos = np.zeros((m.size, 3))
                for h, _m_ in enumerate(m):
                    P_of_rho = (1+delta_flat)**alpha[h]
                    P_of_rho /= np.sum(P_of_rho)

                    # replace=True means a given voxel can house multiple halos.
                    # Might want to make this mass-dependent...
                    # This is slow mostly because our probability distribution is
                    # re-generated for each mass. Could achieve speed-up by
                    # doing this procedure in a few mass bins? In practice, we're
                    # generating mocks in narrow mass ranges (0.1-0.5 dex), so
                    # the mass range is already likely to be small.
                    i = np.random.choice(ivox, p=P_of_rho,
                        replace=self.bias_replacement)

                    pos[h] = pvox[i]

                    if h % 100 == 0:
                        pbar.update(h)

                pbar.finish()
            else:

                # Compute "biasing probability" for entire mass bin.
                lo, hi = m.min(), m.max()
                mbin = 10**np.mean(np.log10([lo, hi]))

                # This is the HALOGEN approach
                alpha = p0 * (mbin / 1e12)**p1

                P_of_rho = (1. + delta_flat)**alpha
                P_of_rho /= np.sum(P_of_rho)

                # Take a random draw with probability set by density.
                # `ivox` contains the flattened coordinates of each pixel
                # as does `P_of_rho`. Passing in `m.size` sets number of
                # draws.
                i = np.random.choice(ivox, p=P_of_rho,
                    replace=self.bias_replacement, size=m.size)

                pos = pvox[i,:]


        ##
        # This doesn't depend on biasing method, just add a little jitter
        # to positions of halos so they aren't all at voxel centers.
        if self.randomise_in_cell:
            shape = N, self.dims
            # Shift relative to bin centers
            pos += np.random.uniform(size=(np.sum(n_act), 3),
                low=-0.5*self.dx, high=0.5*self.dx)

        ##
        # Done
        return pos

    @property
    def _cache_subhalo_cdf(self):
        return self._cache_subhalo_cdf_

    @property
    def _cache_mgtm(self):
        return self._cache_mgtm_

    @cached_property
    def halos(self):
        pop0 = self.sim.pops[0]
        halos = pop0.halos
        # Returning the hidden attribute here means we'll skip hasattr's
        return pop0._halos

    def get_halo_masses(self, z, N, logmlim=(11, 15), seed=None,
        subhalos=False, Mc=None, iz=None, iM=None):
        """
        Draw halos from a model halo mass function.

        Parameters
        ----------
        z : int, float
            Redshift.
        N : int
            Number of halos to draw.
        mmin : float
            Minimum mass [Msun].
        mmax : float
            Maximum mass [Msun]
        subhalos : bool
            If True, draw from subhalo mass function. In this case, must
            also provide central halo mass via `Mc`.
        Mc : float
            Central halo mass [Msun]. Only applicable if `subhalos`=True.
        iz : int
            Index in redshift array.
        iM : int
            Index in halo mass array.

        """
        # Grab dn/dm and construct CDF to randomly sampled HMF.
        if (iz is None) and (iM is None):
            if subhalos:
                iz = None
                iM = np.argmin(np.abs(Mc - self.halos.tab_M))
            else:
                iz = np.argmin(np.abs(self.halos.tab_z - z))
                iM = None

        if subhalos:
            key_id = (iz, iM, logmlim, subhalos)
        else:
            key_id = (iM, logmlim, subhalos)

        if key_id in self._cache_subhalo_cdf.keys():
            m, cdf = self._cache_subhalo_cdf[key_id]
        else:

            # Don't bother with m << mmin halos
            mmin = 10**logmlim[0]
            mmax = 10**logmlim[1]

            # Compute CDF
            if key_id in self._cache_mgtm:
                m, dndm, ngtm, ntot = self._cache_mgtm[key_id]
            else:
                ok = np.logical_and(self.halos.tab_M >= mmin,
                                    self.halos.tab_M <  mmax)

                m = self.halos.tab_M[ok==1]

                if subhalos:
                    assert Mc is not None, "Must provide `Mc` if subhalos=True!"

                    # We only keep dn/dlnM for some reason, convert to dn/dm
                    dndm = self.halos.tab_dndlnm_sub[iM,ok==1] / m

                    #ngtm = cumulative_trapezoid(dndm[-1::-1] * m[-1::-1], x=-np.log(m[-1::-1]),
                    #    initial=0)[-1::-1]

                    ngtm = self.halos.tab_ngtm_sub[iM,ok==1] #\
                         #- self.sim.pops[0].halos.tab_ngtm_sub[iM,immax]
                    #nltm = ngtm[0]

                else:
                    dndm = self.halos.tab_dndm[iz,ok==1]
                    ngtm = self.halos.tab_ngtm[iz,ok==1]

                ntot = np.trapezoid(dndm * m, x=np.log(m))
                self._cache_mgtm[key_id] = m, dndm, ngtm, ntot

            nltm = ntot - ngtm
            cdf = nltm / ntot

            self._cache_subhalo_cdf[key_id] = m, cdf

        # Assign halo masses according to HMF.
        if seed is not None:
            np.random.seed(seed)

        r = np.random.rand(N)

        mass = np.exp(np.interp(r, cdf, np.log(m)))
        #mass = np.exp(_interp_linear(r, cdf, np.log(m)))

        return mass

    def get_prof_params(self, num, seed):
        """
        Return arrays of Sersic indices, positions angles, and ellipticies.

        Parameters
        ----------
        num : int
            Number of galaxies to draw.
        seed : int
            Random seed. Should be determined in LightCone class using the
            get_seed_kwargs function for a given co-eval redshift layer.

        Returns
        -------
        Tuple with three elements: (sersic index, position angle [deg],
        ellipticity = 1 - b / a).
        """
        # Uniform for now.
        np.random.seed(seed)

        # Sersic indices and position angles
        nsers = np.random.random(size=num) * 5.9 + 0.3
        pa = np.random.random(size=num) * 360

        # Ellipticity = 1 - b/a
        ellip = np.random.random(size=num)

        return nsers, pa, ellip

    def get_catalog_halos(self, zlim=None, logmlim=(11,12), popid=0, verbose=True,
        satellites=False, logmlim_sats=None, max_sources=None):
        """
        Get a halo catalog in (RA, DEC, redshift) coordinates.

        .. note :: This is essentially a wrapper around `_get_catalog_from_coeval`,
            i.e., we're just figuring out how many layers are needed along the
            line of sight and re-generating the relevant cubes.

        Parameters
        ----------
        zlim : tuple
            Restrict redshift range to be between:

                zlim[0] <= z < zlim[1]

        logmlim : tuple
            Restrict halo mass range to be between:

                10**logmlim[0] <= Mh/Msun 10**logmlim[1]

        Returns
        -------
        A tuple containing (ra, dec, redshift, halo mass).

        """

        pid, pid_par, pid_str = get_pop_info(popid)

        if logmlim_sats is None:
            logmlim_sats = logmlim

        if zlim is None:
            zlim = self.zlim

        zmin, zmax = zlim
        mmin, mmax = 10**np.array(logmlim)

        # Version of Lbox in actual cMpc
        L = self.Lbox / self.sim.cosm.h70

        # First, get full domain info
        ze, zmid, Re = self.get_domain_info(zlim=self.zlim, Lbox=self.Lbox)
        Rc = bin_e2c(Re)
        dz = np.diff(ze)

        # Deterministically adjust the random seeds for the given mass range
        # and redshift range.
        #fmh = int(logmlim[0] + (logmlim[1] - logmlim[0]) / 0.1)

        # Figure out if we're getting the catalog of a single layer
        layer_id = None
        for i, Rlo in enumerate(zmid):
            zlo, zhi = ze[i:i+2]

            if (zlo == zlim[0]) and (zhi == zlim[1]):
                layer_id = i
                break

        ##
        # Setup random seeds for random rotations and translations
        #np.random.seed(self.seed_rot)
        #r_rot = np.random.randint(0, high=4, size=(len(Re)-1)*3).reshape(
        #    len(Re)-1, 3
        #)
#
        #np.random.seed(self.seed_tra)
        #r_tra = np.random.rand(len(Re)-1, 3)

        ##
        # Print-out information about FOV
        # arcmin / Mpc -> deg / Mpc
        theta_zmin = self.sim.cosm.get_angle_from_length_comoving(zmin, 1) * L / 60.
        theta_zmax = self.sim.cosm.get_angle_from_length_comoving(zmax, 1) * L / 60.

        pbar = ProgressBar(Rc.size, name=f"lc(z>={zmin},z<{zmax})",
            use=layer_id is None)
        pbar.start()

        # Keep running tally of sources
        ct = 0
        # Track max_sources
        _hit_max_sources = False

        # Track parent halos of satellites
        parents = None

        zlo = zmin * 1.
        for i, Rlo in enumerate(Re[0:-1]):
            pbar.update(i)

            zlo, zhi = ze[i:i+2]

            if layer_id is not None:
                if i != layer_id:
                    continue

            if (zhi <= zlim[0]) or (zlo >= zlim[1]):
                continue

            if _hit_max_sources:
                break

            seed_kwargs = self.get_seed_kwargs(i, logmlim, pid)

            ##
            # Optional: lightcone correction
            need_corr = False
            if self.lightcone_corr:
                # Use lightcone_max_evol parameter to determine how much
                # to sub-sample. Restrict attention to range of halo masses
                # for which we expect 1 /per box.
                tol = self.lightcone_max_evol

                izmi = np.argmin(np.abs(self.sim.pops[0].halos.tab_z - zmid[i]))
                Mh = self.sim.pops[0].halos.tab_M
                ngtm = self.sim.pops[0].halos.tab_ngtm[izmi,:]
                mmax = np.interp(10., ngtm[-1::-1] * L**3, Mh[-1::-1])
                imax = np.argmin(np.abs(Mh - mmax))

                okm = np.logical_and(Mh >= mmin, Mh < mmax)
                izlo = np.argmin(np.abs(self.sim.pops[0].halos.tab_z - zlo))
                izhi = np.argmin(np.abs(self.sim.pops[0].halos.tab_z - zhi))
                hmf_lo = self.sim.pops[0].halos.tab_dndlnm[izlo,okm==1]
                hmf_hi = self.sim.pops[0].halos.tab_dndlnm[izhi,okm==1]
                err = np.abs(hmf_hi - hmf_lo) / hmf_hi

                need_corr = np.any(err > tol)

                # How many chunks do we need?
                N = 2
                dz = zhi - zlo
                while np.any(err > tol):
                    zsub_e = np.linspace(zlo, zhi, N+1)
                    zsub = bin_e2c(zsub_e)

                    err_prev = err.copy()

                    hmfs = []
                    err = np.zeros(okm.sum())
                    for ll, _z_ in enumerate(zsub_e):
                        _i_ = np.argmin(np.abs(self.sim.pops[0].halos.tab_z - _z_))
                        hmfs.append(self.sim.pops[0].halos.tab_dndlnm[_i_,okm==1])

                        if ll == 0:
                            continue

                        _err = np.abs(hmfs[ll] - hmfs[ll-1]) / hmfs[ll]
                        err = np.maximum(err, _err)

                    N += 1

                    if np.allclose(err, err_prev) and self.verbose*verbose:
                        print(f"HMF evolution along LoS reached minimum with N={N}")
                        break

                print(f"! Will sub-cycle in {N} intervals from ({zlo}, {zhi})")
                ##
                # Need to map these redshift intervals to cMpc / h units


            # Contains (x, y, z, mass)
            # Note that x, y, z are in cMpc / h units, not actual cMpc.
            # The values thus run from 0 to Lbox.
            if not need_corr:
                halos = self.get_halo_population(z=zmid[i], ze=(zlo, zhi),
                    mmin=mmin, mmax=mmax, verbose=verbose, popid=popid,
                    **seed_kwargs)
            else:
                # In this case, generate the halo population in segments.
                # The density field will automatically be LC-corrected
                # so we just need to handle sub-cycling over a few redshifts
                ra = []; dec = []; red = []; mass = []
                for ll, _z_ in enumerate(zsub):
                    _halos = self.get_halo_population(z=zmid[i], ze=(zlo, zhi),
                        mmin=mmin, mmax=mmax, verbose=verbose, popid=popid,
                        zsub=_z_, **seed_kwargs)

                    # Convert to lightcone coordinates to slice on redshift
                    _ra, _de, _red = \
                        self._get_catalog_from_coeval(_halos, zlo=zlo)

                    # Select only objects in the right sub-interval
                    oksub = np.logical_and(_red >= zsub_e[ll], _red < zsub_e[ll+1])

                    # Cut out halos outside zsub_e[ll], zsub_e[ll+1]
                    _x_, _y_, _z_, _m_ = _halos

                    # Note that (x, y, z) here are still [0, Lbox],
                    # but we constructed `oksub` from the redshifts properly.
                    ra.extend(_x_[oksub==1])
                    dec.extend(_y_[oksub==1])
                    red.extend(_z_[oksub==1])
                    mass.extend(_m_[oksub==1])

                halos = np.array(ra), np.array(dec), np.array(red), np.array(mass)#np.array([ra, dec, red, mass]).T

            if (type(halos[0]) != np.ndarray) and (halos[0] is None):
                ra = dec = red = mass = None
                continue

            if (halos[0].size == 0):
                ra = dec = red = mass = None
                continue

            ##
            # Convert to (ra, dec, redshift) coordinates.
            # Note: the conversion from cMpc/h to cMpc occurs inside
            # _get_catalog_from_coeval here:
            _ra, _de, _red = self._get_catalog_from_coeval(halos, zlo=zlo)
            _m = halos[-1]

            # Note that halos outside the specific FoV and redshift
            # range are filtered out at a higher level in LightCone.get_catalog

            ##
            # For satellites: one more step before moving to next layer.
            if satellites:

                ra_s, dec_s, red_s, mass_s, par_id = \
                    self.get_catalog_subhalos(_ra, _de, _red, _m,
                        popid=popid, logmlim=logmlim_sats,
                        seed=seed_kwargs['seed_sats'] + pid_par,
                        distribute_in_space=self.distribute_sats_spatially)

                # Replace info about central with satellite info
                _ra, _de, _red, _m = ra_s, dec_s, red_s, mass_s

                # Need to hack off satellites that end up outside the FoV


            # Save results
            if ct == 0:
                ra = _ra.copy()
                dec = _de.copy()
                red = _red.copy()
                mass = _m.copy()

                if satellites:
                    parents = par_id.copy()

            else:
                ra = np.hstack((ra, _ra))
                dec = np.hstack((dec, _de))
                red = np.hstack((red, _red))
                mass = np.hstack((mass, _m))

                if satellites:
                    parents = np.hstack((parents, par_id))

            ct += 1

            del _ra, _de, _red, halos, _m
            if self.apply_rotations or self.apply_translations:
                del _x, _x_, _y, _y_, _z, _z_, _m_

            if satellites:
                del ra_s, dec_s, red_s, mass_s, par_id

            if self.mem_concious:
                gc.collect()

            ##
            # Done with this co-eval layer

        pbar.finish()

        #self._cache_cats[(zmin, zmax, mmin)] = ra, dec, red, mass
        return ra, dec, red, mass, parents

    def get_catalog_subhalos(self, ra_c, dec_c, red_c, mass_c, popid,
        logmlim=(11,15), seed=None, distribute_in_space=True):
        """
        Get a catalog of satellite galaxies for input central catalog.

        Parameters
        ----------
        ra_c : np.ndarray
            Right ascension of all central halos [deg].
        dec_c : np.ndarray
            Declination of all central halos [deg].
        red_c : np.ndarray
            Redshifts of all central halos.
        mass_c : np.ndarray
            Masses of all central halos [Msun].
        pid_c : np.ndarray

        distribute_in_space : bool
            If True, will position subhalos randomly in proportion to the
            projected NFW density profile. If False, subhalos will be placed at
            the location of their parent central. This is really just an option
            implemented for sanity checks.

        """

        pid, pid_par, pid_str = get_pop_info(popid)

        ##
        # All we're going to do is randomly distribute satellites in
        # mass according to the subhalo mass function and in space
        # using an NFW profile.

        # First, grab a few things we need. This is 2-D (Mc, Msat)
        hmf_sub = self.halos.tab_dndlnm_sub

        ok_sub = np.logical_and(self.halos.tab_M >= 10**logmlim[0],
                                self.halos.tab_M <  10**logmlim[1])

        # Expected number of subhalos vs. central halo mass.
        # Just need to do this once per `logmlim`.
        Nexp = np.trapezoid(hmf_sub[:,ok_sub==1],
            x=np.log(self.halos.tab_M[ok_sub==1]), axis=1)

        # Array of radial separations [cMpc]
        d = self.sim.halos.tab_R_nfw

        ##
        # Just loop to start. Could truncate based on where expected
        # number of satellites is effectively zero.
        Nc = len(mass_c)

        ##
        # Reproducibility is important.
        # Make seeds for halo position and mass sampling.
        # Note that this is done in a slightly different way from centrals.
        # Instead of providing seeds for everything by hand, we use one seed
        # to deterministically create seeds for the masses and positions
        # of all subhalos for each central.
        np.random.seed(seed)
        # Recall that max allowed seed value is 2**32 - 1
        # Providing some margin here since we scale below.
        seeds_num = np.random.randint(0, high=2**30, size=Nc)
        seeds_pos = np.random.randint(0, high=2**30, size=Nc)
        seeds_mass = np.random.randint(0, high=2**30, size=Nc)
        seeds_occ = np.random.randint(0, high=2**30, size=Nc)

        # Do we really need a new seed for each central?
        # It is surprisingly expensive to call np.seed on each iteration

        # Determine closest mass and redshift bins for projected density profile
        iM = np.searchsorted(self.halos.tab_M_e, mass_c,
            side='right') - 1
        iz = np.searchsorted(self.halos.tab_z, red_c,
            side='right') - 1

        mpc_per_deg = \
            self.sim.cosm.get_length_comoving_from_angle(red_c, 60.)

        ra = []
        dec = []
        red = []
        mass = []
        par_id = []
        for i in range(Nc):

            # Remaining dimension: halos.tab_R_nfw
            Sigma = self.halos.tab_Sigma_nfw[iz[i],iM[i],:]

            Nsat_exp = int(Nexp[iM[i]])

            # Note that some Nexp==0 objects should statistically end up
            # with one or even a few satellites, but this should be a really
            # small effect and at the moment (at least) not SUs well spent.
            if Nsat_exp == 0:
                continue

            # Poisson random draw to determine actual number of subhalos,
            # given expected number.
            np.random.seed(seeds_num[i])
            Nsat_act_tot = np.random.poisson(Nsat_exp)

            # This looked OK
            #print(i, np.log10(self.halos.tab_M[iM[i]]), Nsat_exp, Nsat_act_tot)
            #input('<enter>')

            if Nsat_act_tot == 0:
                continue

            # Outsources sampling over sub-halo MF
            _m = self.get_halo_masses(red_c[i], Nsat_act_tot,
                logmlim=logmlim, seed=seeds_mass[i],
                subhalos=True, Mc=mass_c[i], iz=iz[i], iM=iM[i])

            ##
            # Apply occupation fraction
            _x, _y, _z, _m = self._filter_by_focc((None, None, None, _m),
                red_c[i], seeds_occ[i], popid)

            if _m is None:
                continue

            Nsat_act = len(_m)

            mass.extend(list(_m))

            ##
            # Now, do positions. Do in 2-D or 3-D?
            if distribute_in_space:

                cdf = self.halos.tab_Sigma_nfw_cdf[iz[i],iM[i],:]

                np.random.seed(seeds_pos[i])
                r = np.random.rand(Nsat_act)

                # Radial displacement of all satellites in cMpc
                r_proj_mpc = np.exp(np.interp(r, cdf, np.log(d)))
                #r_proj_mpc = np.exp(_interp_linear(r, cdf, np.log(d)))

                r_proj_deg = r_proj_mpc / mpc_per_deg[i]

                # Need to turn into RA and DEC
                # Randomly choose an angle
                #np.random.seed(seeds_pos[i] * 2)
                theta = np.random.rand(Nsat_act) * 2 * np.pi

                # Then convert to x and y displacements
                x_deg = np.cos(theta) * r_proj_deg
                y_deg = np.sin(theta) * r_proj_deg

            else:
                x_deg = y_deg = 0

            # Save progress
            ra.extend(list(ra_c[i] + x_deg))
            dec.extend(list(dec_c[i] + y_deg))

            ##
            # Make some dynamical argument to shift redshifts?
            # Someday, sure. For now, just put at same exact z as central.
            red.extend([red_c[i]] * Nsat_act)

            # Save index for the parent halo.
            par_id.extend([i] * Nsat_act)

        #
        #pbar.finish()

        return np.array(ra), np.array(dec), np.array(red), np.array(mass), \
            np.array(par_id, dtype=int)

    def _get_catalog_from_coeval(self, halos, zlo):
        """
        Make a catalog in lightcone coordinates (RA, DEC, redshift).

        .. note :: RA and DEC output in degrees.

        """

        # Right now, in [0, Lbox / h] units.
        xmpc, ympc, zmpc, mass = halos

        # Shift coordinates to +/- 0.5 * Lbox
        xmpc = (xmpc - 0.5 * self.Lbox) / self.sim.cosm.h70
        ympc = (ympc - 0.5 * self.Lbox) / self.sim.cosm.h70

        # Move the front edge of the box to redshift `zlo`
        # Will automatically use interpolation under the hood in `cosm`
        # if interpolate_cosmology_in_z=True.
        d0 = self.sim.cosm.get_dist_los_comoving(0, zlo) / cm_per_mpc

        # Translate LOS distances to redshifts.

        # Distance from z=0 to z
        dofz = self.sim.cosm._tab_dist_los_co / cm_per_mpc
        #
        angl = self.sim.cosm._tab_ang_from_co / 60.
        # Determine redshift by interpolating distance along z
        red = np.interp((zmpc / self.sim.cosm.h70) + d0, dofz,
            self.sim.cosm.tab_z)

        # Conversion from physical to angular coordinates
        deg_per_mpc = np.interp((zmpc / self.sim.cosm.h70) + d0, dofz, angl)

        ra  = xmpc * deg_per_mpc
        dec = ympc * deg_per_mpc

        return ra, dec, red

    def _filter_by_focc(self, cat, z, seed_occ, popid):
        """
        Take a raw catalog of halos and thin according to occupation fraction.

        Parameters
        ----------
        cat : tuple
            Contains (x, y, redshift, mass), where x and y can be co-eval box
            coordinates or RA and DEC.
        z : int, float
            Redshift
        N :
        """

        _x, _y, _z, mass = cat
        N = len(mass)

        # ARES ID, parent ID [if applicable], ID str (user supplied; just -> str)
        pid, pid_par, pid_str = get_pop_info(popid)

        ##
        # Apply occupation fraction here?
        if self.sim.pops[pid].pf['pop_focc'] != 1:

            np.random.seed(seed_occ)

            r = np.random.rand(N)
            focc = self.sim.pops[pid].get_focc(z=z, Mh=mass)

            ok = np.ones(N)
            ok[r > focc] = 0

            # For satellites, positions are determined after this step
            if _x is None:
                pass
            else:
                _x = _x[ok==1]
                _y = _y[ok==1]
                _z = _z[ok==1]

            mass = mass[ok==1]

            # Don't really need to see this anymore.
            #if verbose:
            #    print(f"# Applied occupation fraction cut for pop #{popid} at z={z:.2f} in {np.log10(mmin):.1f}-{np.log10(mmax):.1f} mass range.")
            #    print(f"# [reduced number of halos by {100*(1-ok.sum()/float(ok.size)):.2f}%]")

            if ok.sum() == 0:
                return None, None, None, None
        else:
            focc = r = ok = None

        del focc, ok, r
        if self.mem_concious:
            gc.collect()

        return _x, _y, _z, mass

    def get_halo_population(self, z, ze=None, seed=None, seed_box=None, seed_pos=None,
        seed_occ=None, mmin=1e11, mmax=np.inf, randomise_in_cell=True, popid=0,
        verbose=True, call_gc=False, apply_focc=True, zsub=None, **_kw_):
        """
        Get a realization of a halo population.

        Parameters
        ----------
        z : int, float
            Redshift, will be used to identify co-eval cube.
        ze : tuple
            Contains edges of co-eval cube in redshift (front, back).
        seed : int
            Random seed for halo masses.
        seed_box : int
            Random seed for density field.
        seed_pos : int
            Random seed for halo positions.
        seed_occ : int
            Random seed for halo occupation.
        zsub :

        Returns
        -------
        Tuple containing (x, y, z, mass), where x, y, and z are halo positions
        in cMpc / h (between 0 and self.Lbox), and mass is in Msun.

        """

        if zsub is None:
            zsub = z

        # Unpack popid more [as of March 2025]
        # (id number in ARES, parent ID number [if satellite], name as str)
        pid, pid_par, pid_str = get_pop_info(popid)

        rho = self.get_density_field(z=z, seed=seed_box)

        # Get mean halo abundance in #/cMpc^3 [note: this is *not* (cMpc/h)^-3]
        nbar = self.get_nbar(zsub, ze, mmin=mmin, mmax=mmax)

        # Compute expected number of halos in volume
        h = self.sim.cosm.h70
        Nexp = nbar * (self.Lbox / h)**3

        # If halos are unbiased, perform Poisson draw for number of galaxies
        # in each voxel independently. Then, generate the appropriate number
        # of halo masses.
        if self.bias_model == 0:
            pos = self.get_halo_positions(zsub, Nexp, rho, seed=seed_pos)
            Nact = pos.shape[0]

            # Draw halo masses from HMF

            mass = self.get_halo_masses(zsub, Nact, logmlim=tuple(np.log10([mmin, mmax])),
                seed=seed)

        # In this case, we need to know the masses of halos before we generate
        # their positions. So, take a Poisson draw to obtain the *total*
        # number of halos in the box, *then* generate their masses, *then*
        # generate their positions (which are effectivley mass-dependent).
        elif self.bias_model == 1:
            # First generate positions the easy way just to force this method
            # to have the same number of halos
            pos = self.get_halo_positions(z, Nexp, rho, seed=seed_pos, bias_model=0)
            # Actual number is a Poisson draw
            Nact = pos.shape[0]#np.random.poisson(Nexp)

            # Draw halo masses from HMF

            mass = self.get_halo_masses(zsub, Nact, logmlim=tuple(np.log10([mmin, mmax])),
                seed=seed)

            pos = self.get_halo_positions(zsub, Nact, rho, m=mass,
                seed=seed_pos)
        else:
            raise NotImplemented('help')

        # `pos` is in [0, Lbox / h] domain in each dimension
        _x, _y, _z = pos.T
        N = _x.size

        if N == 0:
            return None, None, None, None

        # Should be within a few percent of <N> unless <N>

        Nerr = abs(Nexp - Nact)
        err = Nerr / Nexp

        # Recall that variance of Poissonian is the same as the mean, so just
        # do a quick check that the number smaller than 2x sqrt(mean). Note
        # that occassionally we might get a bigger difference here, hence the
        # warning instead of raising an exception.
        #if (Nerr > 2 * np.sqrt(Nexp)) and (err > 0.2) and self.verbose:
        #    print(f"# WARNING: Error in halo density is {err*100:.0f}% for m in [{np.log10(mmin):.1f},{np.log10(mmax):.1f}]")
        #    print(f"# (expected {Nexp:.2f} halos, got {Nact:.0f})")
        #    print("# Might be small box issue, but could be OK for massive halos.")

        if np.any(mass < mmin):
            raise ValueError("help")

        ##
        # Apply occupation fraction cut
        if apply_focc:
            _x, _y, _z, mass = self._filter_by_focc((_x, _y, _z, mass),
                z, seed_occ, popid)

        ##
        # Sort by mass? Otherwise will essentially be in order of pixels as
        # determined by np.ravel. That's what we're going with.
        return _x, _y, _z, mass
