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
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from ..util.Stats import bin_c2e, bin_e2c
from ..physics.Constants import cm_per_mpc

try:
    import powerbox as pbox
except ImportError:
    pass

class LogNormal(LightCone): # pragma: no cover
    def __init__(self, model_name, Lbox=256, dims=128, zmin=0.05, zmax=2, verbose=True,
        seed_rho=None, seed_halo_mass=None, seed_halo_pos=None, seed_halo_occ=None,
        seed_rot=None, seed_trans=None, seed_pa=None, seed_nsers=None,
        apply_rotations=False, apply_translations=False,
        bias_model=0, bias_params=None, bias_replacement=1, bias_within_bin=False,
        randomise_in_cell=True, base_dir='ares_mock', mem_concious=1, **kwargs):
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
        self.zmin = zmin
        self.zmax = zmax
        self.zlim = (zmin, zmax)
        self.seed_rho = seed_rho
        self.seed_halo_mass = seed_halo_mass
        self.seed_halo_pos = seed_halo_pos
        self.seed_halo_occ = seed_halo_occ
        self.seed_rot = seed_rot
        self.seed_tra = seed_trans
        self.seed_pa = seed_pa
        self.seed_nsers = seed_nsers
        self.apply_rotations = apply_rotations
        self.apply_translations = apply_translations

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
            print(f"# Number of co-eval chunks: {zmid.size}")

    def get_seed_kwargs(self, chunk, logmlim):
        # Deterministically adjust the random seeds for the given mass range
        # and redshift range.
        fmh = int(logmlim[0] + (logmlim[1] - logmlim[0]) / 0.1)

        ze, zmid, Re = self.get_domain_info(zlim=self.zlim, Lbox=self.Lbox)

        if not hasattr(self, '_seeds'):
            self._seeds = self.seed_rho * np.arange(1, len(zmid)+1)
            self._seeds_hm = self.seed_halo_mass * np.arange(1, len(zmid)+1) * fmh
            self._seeds_hp = self.seed_halo_pos * np.arange(1, len(zmid)+1) * fmh
            self._seeds_ho = self.seed_halo_occ * np.arange(1, len(zmid)+1) * fmh

            if self.seed_nsers is not None:
                self._seeds_nsers = self.seed_nsers * np.arange(1, len(zmid)+1) * fmh
            else:
                self._seeds_nsers = [None] * len(zmid)
            if self.seed_pa is not None:
                self._seeds_pa = self.seed_pa * np.arange(1, len(zmid)+1) * fmh
            else:
                self._seeds_pa = [None] * len(zmid)

        i = chunk
        return {'seed_box': self._seeds[i],
            'seed': self._seeds_hm[i], 'seed_pos': self._seeds_hp[i],
            'seed_occ': self._seeds_ho[i],
            'seed_nsers': self._seeds_nsers[i], 'seed_pa': self._seeds_pa[i]}

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
            iz = np.argmin(np.abs(self.sim.pops[0].halos.tab_z - z))
            ok = np.logical_and(self.sim.pops[0].halos.tab_M >= mmin,
                                self.sim.pops[0].halos.tab_M < mmax)

            m = self.sim.pops[0].halos.tab_M[ok==1]
            dndm = self.sim.pops[0].halos.tab_dndm[iz,ok==1]

            nall = cumtrapz(dndm * m, x=np.log(m), initial=0.0)
            nbar = np.trapz(dndm * m, x=np.log(m)) \
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

    def get_nbar(self, z, mmin, mmax=np.inf, fov=None, dz=None):
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

        iz = np.argmin(np.abs(self.sim.pops[0].halos.tab_z - z))
        ok = np.logical_and(self.sim.pops[0].halos.tab_M >= mmin,
                            self.sim.pops[0].halos.tab_M < mmax)

        m = self.sim.pops[0].halos.tab_M[ok==1]
        dndm = self.sim.pops[0].halos.tab_dndm[iz,ok==1]

        nall = cumtrapz(dndm * m, x=np.log(m), initial=0.0)
        nbar = np.trapz(dndm * m, x=np.log(m)) \
             - np.exp(np.interp(np.log(mmin), np.log(m), np.log(nall)))

        # Correct for FOV
        if (fov is not None) and (dz is not None):
            vol = self.get_survey_vol(z, fov, dz)
            nbar *= vol
        elif (fov is not None) or (dz is not None):
            raise ValueError("Must provide `fov` AND `dz` or neither!")

        return nbar

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

        iz = np.argmin(np.abs(self.sim.pops[0].halos.tab_z - z))

        power = interp1d(self.sim.pops[0].halos.tab_k_lin,
            self.sim.pops[0].halos.tab_ps_lin[iz,:], kind='cubic')

        self._cache_ps[z] = power

        return power(k)

    def get_density_field(self, z, seed=None):
        return self.get_box(z=z, seed=seed)

    def get_box(self, z, seed=None):
        """
        Get a 3-D realization of a log-normal field at input redshift.

        Returns
        -------
        powerbox.powerbox.LogNormalPowerBox object, attribute `delta_x()` can
        be used to retrieve the box itself.
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

    def get_halo_positions(self, z, N, delta_x, m=None, seed=None):
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

        # This is the same thing that powerbox is doing in
        # `create_discrete_sample`, just trying to have a unified call
        # sequence for other options here.
        if self.bias_model == 0:

            n = N / (self.Lbox / self.sim.cosm.h70)**3

            # Expected number of halos in each cell
            n_exp = n * (1. + delta_x) * (self.dx / self.sim.cosm.h70)**3

            # Actual number after Poisson draw
            np.random.seed(seed)
            n_act = np.random.poisson(n_exp)

            # Repeat position of each voxel N times, one for each halo that
            # lives there.
            pos = pvox.repeat(n_act.ravel(), axis=0)

        # In this case, we're increasing the probability that halos are drawn
        # from overdensities in a potentially halo mass dependent way.
        elif self.bias_model == 1:

            n_act = m.size

            ivox = np.arange(pvox.shape[0])
            rho_flat = delta_x.ravel()

            # Right now, alpha(m) = p0 * (m / 1e12)**p1
            p0, p1 = self.bias_params

            if self.bias_within_bin:
                pbar = ProgressBar(m.size, name=f"pos(m)", use=True)
                pbar.start()

                alpha = p0 * (m / 1e12)**p1

                pos = np.zeros((m.size, 3))
                for h, _m_ in enumerate(m):
                    P_of_rho = (1+rho_flat)**alpha[h]
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
                alpha = p0 * (mbin / 1e12)**p1

                P_of_rho = (1. + rho_flat)**alpha
                P_of_rho /= np.sum(P_of_rho)

                # Take a random draw with probability set by density.
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

    def get_halo_masses(self, z, N, mmin=1e11, mmax=np.inf, seed=None):
        # Grab dn/dm and construct CDF to randomly sampled HMF.

        # Don't bother with m << mmin halos
        iz = np.argmin(np.abs(self.sim.pops[0].halos.tab_z - z))
        ok = np.logical_and(self.sim.pops[0].halos.tab_M >= mmin,
                            self.sim.pops[0].halos.tab_M < mmax)

        m = self.sim.pops[0].halos.tab_M[ok==1]
        dndm = self.sim.pops[0].halos.tab_dndm[iz,ok==1]

        # Compute CDF
        ngtm = cumtrapz(dndm[-1::-1] * m[-1::-1], x=-np.log(m[-1::-1]),
            initial=0)[-1::-1]

        ntot = np.trapz(dndm * m, x=np.log(m))
        nltm = ntot - ngtm
        cdf = nltm / ntot

        # Assign halo masses according to HMF.
        np.random.seed(seed)
        r = np.random.rand(N)

        #mass = np.exp(np.interp(np.log(r), np.log(cdf), np.log(m)))
        mass = np.exp(np.interp(r, cdf, np.log(m)))

        #if np.any(np.isnan(mass)):
        #    print('hey wtf', r[np.argwhere(np.isnan(mass))], r.min(), r.max(),
        #        np.interp(r[np.argwhere(np.isnan(mass))], cdf, m))

        return mass

    def get_halo_population(self, z, seed=None, seed_box=None, seed_pos=None,
        seed_occ=None, mmin=1e11, mmax=np.inf, randomise_in_cell=True, popid=0,
        verbose=True, call_gc=False, **_kw_):
        """
        Get a realization of a halo population.

        Returns
        -------
        Tuple containing (x, y, z, mass), where x, y, and z are halo positions
        in cMpc / h (between 0 and self.Lbox), and mass is in Msun.

        """

        pb = self.get_box(z=z, seed=seed_box)

        # Get mean halo abundance in #/cMpc^3 [note: this is *not* (cMpc/h)^-3]
        nbar = self.get_nbar(z, mmin=mmin, mmax=mmax)

        # Compute expected number of halos in volume
        h = self.sim.cosm.h70
        Nexp = nbar * (self.Lbox / h)**3

        # If halos are unbiased, perform Poisson draw for number of galaxies
        # in each voxel independently. Then, generate the appropriate number
        # of halo masses.
        if self.bias_model == 0:
            pos = self.get_halo_positions(z, Nexp, pb.delta_x(), seed=seed_pos)
            Nact = pos.shape[0]

            # Draw halo masses from HMF
            mass = self.get_halo_masses(z, Nact, mmin=mmin, mmax=mmax,
                seed=seed)

        # In this case, we need to know the masses of halos before we generate
        # their positions. So, take a Poisson draw to obtain the *total*
        # number of halos in the box, *then* generate their masses, *then*
        # generate their positions (which are effectivley mass-dependent).
        elif self.bias_model == 1:
            # Actual number is a Poisson draw
            Nact = np.random.poisson(Nexp)

            # Draw halo masses from HMF
            mass = self.get_halo_masses(z, Nact, mmin=mmin, mmax=mmax,
                seed=seed)

            pos = self.get_halo_positions(z, Nact, pb.delta_x(), m=mass,
                seed=seed_pos)
        else:
            raise NotImplemented('help')

        # `pos` is in [0, Lbox / h] domain in each dimension
        _x, _y, _z = pos.T#(pos.T / h) #- 0.5 * (self.Lbox / h)
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
        if (Nerr > 2 * np.sqrt(Nexp)) and (err > 0.2):
            print(f"# WARNING: Error in halo density is {err*100:.0f}% for m in [{np.log10(mmin):.1f},{np.log10(mmax):.1f}]")
            print(f"# (expected {Nexp:.2f} halos, got {Nact:.0f})")
            print("# Might be small box issue, but could be OK for massive halos.")

        if np.any(mass < mmin):
            raise ValueError("help")

        ##
        # Apply occupation fraction here?
        if self.sim.pops[popid].pf['pop_focc'] != 1:

            np.random.seed(seed_occ)

            r = np.random.rand(N)
            focc = self.sim.pops[popid].get_focc(z=z, Mh=mass)

            ok = np.ones(N)
            ok[r > focc] = 0

            _x = _x[ok==1]
            _y = _y[ok==1]
            _z = _z[ok==1]
            mass = mass[ok==1]

            if verbose:
                print(f"# Applied occupation fraction cut for pop #{popid} at z={z:.2f} in {np.log10(mmin):.1f}-{np.log10(mmax):.1f} mass range.")
                print(f"# [reduced number of halos by {100*(1-ok.sum()/float(ok.size)):.2f}%]")

            if ok.sum() == 0:
                return None, None, None, None
        else:
            focc = r = ok = None

        del focc, ok, r, pos
        if self.mem_concious:
            gc.collect()

        return _x, _y, _z, mass
