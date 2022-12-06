"""

LogNormal.py

Author: Jordan Mirocha
Affiliation: Jet Propulsion Laboratory
Created on: Sat Dec  3 14:28:42 PST 2022

Description:

"""

import numpy as np
from ..util import ProgressBar
from .LightCone import LightCone
from scipy.integrate import cumtrapz
from ..simulations import Simulation
from scipy.interpolate import interp1d
from ..util.Stats import bin_c2e, bin_e2c
from ..physics.Constants import cm_per_mpc
from scipy.interpolate import InterpolatedUnivariateSpline as spline

try:
    import powerbox as pbox
except ImportError:
    pass

class LogNormal(LightCone):
    def __init__(self, Lbox=256, dims=128, zlim=(0.2, 2), seed=None,
        seed_halos=None, verbose=True, **kwargs):
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
        self.seed = seed
        self.seed_halos = seed_halos
        self.verbose = verbose
        self.kwargs = kwargs

        fov = self.get_fov_from_L(0.5, Lbox)

        if verbose:
            print("# Maximum FoV given box size:")
            for z in [0.2, 1, 2, 3, 4, 5, 6, 10]:
                fov = self.get_fov_from_L(z, Lbox)
                print(f"# At z={z:.1f}: {fov:.2f} degrees")


    @property
    def sim(self):
        if not hasattr(self, '_sim'):
            self._sim = Simulation(**self.kwargs)
        return self._sim

    @property
    def pops(self):
        if not hasattr(self, '_pops'):
            self._pops = self.sim.pops
        return self._pops

    def get_fov_from_L(self, z, Lbox):
        """
        Return FOV in degrees (single dimension) given redshift and Lbox in
        cMpc / h.
        """
        return (self.sim.cosm.ComovingLengthToAngle(z, 1) / 60.) \
            * (Lbox / self.sim.cosm.h70)

    def get_L_from_fov(self, z, fov):
        """
        Get length scale corresponding to given field of view.

        .. note :: This is in co-moving Mpc, NOT cMpc / h!

        """
        ang_per_L = self.sim.cosm.ComovingLengthToAngle(z, 1) / 60.

        return fov / ang_per_L

    def get_memory_estimate(self, zlim=None, logmlim=None, Lbox=None, dims=None):
        """
        Return rough estimate of memory needed vs. redshift in GB.

        .. note :: Assumes you need (x, y, z, mass) for each halo.

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
            #self._mf.update(z=z)
            #m = self._mf.m / self._mf.cosmo.h # These are Msun / h remember
            #dndm = self._mf.dndm * self._mf.cosmo.h**4

            iz = np.argmin(np.abs(self.sim.pops[0].halos.tab_z - z))
            ok = np.logical_and(self.sim.pops[0].halos.tab_M >= mmin,
                                self.sim.pops[0].halos.tab_M < mmax)

            m = self.sim.pops[0].halos.tab_M[ok==1]
            dndm = self.sim.pops[0].halos.tab_dndm[iz,ok==1]

            nall = cumtrapz(dndm * m, x=np.log(m), initial=0.0)
            nbar = np.trapz(dndm * m, x=np.log(m)) \
                 - np.exp(np.interp(np.log(mmin), np.log(m), np.log(nall)))

            N = nbar * (Lbox / self.sim.cosm.h70)**3
            mz = N * 8 * 4
            mc += mz

            mem_z.append(mz)
            mem_c.append(mc)

        return zmid, np.array(mem_z) / 1e9, np.array(mem_c) / 1e9

    def get_nbar(self, z, mmin, mmax=np.inf, fov=None, dz=None):
        """
        Return expected number density of halos at given z for given minimum
        mass.

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
        #self._mf.update(z=z)
        #m = self._mf.m / self._mf.cosmo.h # These are Msun / h remember
        #dndm = self._mf.dndm * self._mf.cosmo.h**4

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

    def get_box(self, z, seed=None):
        """
        Get a 3-D realization of a log-normal field at input redshift.

        Returns
        -------
        powerbox.powerbox.LogNormalPowerBox object, attribute `delta_x()` can
        be used to retrieve the box itself.
        """

        #if not hasattr(self, '_cache_box'):
        #    self._cache_box = {}

        #if (z, seed) in self._cache_box:
        #    return self._cache_box[(z, seed)]

        power = lambda k: self.get_ps_mm(z, k)

        pb = pbox.LogNormalPowerBox(N=self.dims, dim=3, pk=power,
            boxlength=self.Lbox, seed=seed)

        #self._cache_box[(z, seed)] = pb

        return pb

    def get_halo_population(self, z, seed=None, seed_box=None, mmin=1e11,
        mmax=np.inf, randomise_in_cell=True):
        """
        Get a realization of a halo population.

        Returns
        -------
        Tuple containing (x, y, z, mass).

        """

        pb = self.get_box(z=z, seed=seed_box)

        # Get mean halo abundance in #/cMpc^3
        # Need to put h^3 back because `create_discrete_sample` is going
        # to use grid resolution in cMpc/h, since that's what Lbox is in,
        # to determine the number of tracers.
        h = self.sim.cosm.h70**3
        nbar = self.get_nbar(z, mmin=mmin, mmax=mmax) / h**3

        # Setting `store_pos` to True means the `pb` object (which is cached)
        # will hang-on to these positions, hence the lack of explicit caching
        # for this method.
        pos = pb.create_discrete_sample(nbar, randomise_in_cell=randomise_in_cell,
            store_pos=True, min_at_zero=False)

        _x, _y, _z = (pos.T / h) + 0.5 * self.Lbox / h
        N = _x.size

        if N == 0:
            return None, None, None, None

        # Should be within a few percent of <N>
        err = abs(1 - N / (nbar * self.Lbox**3))
        if err > 0.1:
            print(f"# WARNING: Error wrt <N> = {err*100}% for m in [{np.log10(mmin):.1f},{np.log10(mmax):.1f}]")
            print("# Might be small box issue, but could be OK for massive halos.")

        # Grab dn/dm and construct CDF to randomly sampled HMF.
        #self._mf.update(z=z)

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
        mass = np.exp(np.interp(np.log(r), np.log(cdf), np.log(m)))

        if np.any(mass < mmin):
            raise ValueError("help")

        return _x, _y, _z, mass

    def _get_catalog_from_coeval(self, halos, z0=0.2):
        """
        Make a catalog in lightcone coordinates (RA, DEC, redshift).

        .. note :: RA and DEC output in degrees.

        """

        xmpc, ympc, zmpc, mass = halos

        # Shift coordinates to +/- 0.5 * Lbox
        xmpc = xmpc - 0.5 * self.Lbox / self.sim.cosm.h70
        ympc = ympc - 0.5 * self.Lbox / self.sim.cosm.h70

        # Don't shift zmpc at all, z0 is the front face of the box

        # First, get redshifts
        zarr = np.linspace(0, 10, 1000)
        #dofz = self._mf.cosmo.comoving_distance(zarr).to_value()
        #angl = self._mf.cosmo.arcsec_per_kpc_comoving(zarr).to_value()

        dofz = np.array([self.sim.cosm.ComovingRadialDistance(0, z) \
            for z in zarr]) / cm_per_mpc

        # arcmin / Mpc -> deg / Mpc
        angl = np.array([self.sim.cosm.ComovingLengthToAngle(z, 1) \
            for z in zarr]) / 60.

        # Move the front edge of the box to redshift `z0`
        d0 = np.interp(z0, zarr, dofz)

        # Translate LOS distances to redshifts.
        red = np.interp(zmpc + d0, dofz, zarr)

        # Conversion from physical to angular coordinates
        deg_per_mpc = np.interp(zmpc + d0, dofz, angl)

        ra  = xmpc * deg_per_mpc
        dec = ympc * deg_per_mpc

        return ra, dec, red

    def get_domain_info(self, zlim=None, Lbox=None):
        """
        Figure out how domain will be divided up along line of sight.

        Returns
        -------
        A tuple containing (chunk edges in redshift, chunk edges in comoving Mpc).
        """

        if Lbox is None:
            Lbox = self.Lbox

        # First, get redshifts
        zarr = np.arange(0, 30, 0.05)
        dofz = np.array([self.sim.cosm.ComovingRadialDistance(0, z) \
            for z in zarr]) / cm_per_mpc

        if zlim is None:
            zmin, zmax = self.zlim
        else:
            zmin, zmax = zlim

        Rmin = np.interp(zmin, zarr, dofz)
        Rmax = np.interp(zmax, zarr, dofz)
        Nbox = 1 + int((Rmax - Rmin) // (Lbox / self.sim.cosm.h70))

        Re = np.arange(Rmin, Rmax+(Lbox / self.sim.cosm.h70), Lbox / self.sim.cosm.h70)

        Rc = bin_e2c(Re)
        ze = np.interp(Re, dofz, zarr)
        dz = np.diff(ze)

        # Redshift midpoint
        zmid = np.zeros_like(Rc)
        for i, zlo in enumerate(ze[0:-1]):
            zmid[i] = np.interp(Re[i]+0.5*(Lbox / self.sim.cosm.h70), dofz, zarr)

        return ze, zmid, Re

    def get_catalog(self, zlim=None, logmlim=(11,12), randomise_in_cell=True):
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

        .. note :: This is essentially a wrapper around `_get_catalog_from_coeval`,
            i.e., we're just figuring out how many chunks are needed along the
            line of sight and re-generating the relevant cubes.

        """

        #if not hasattr(self, '_cache_cats'):
        #    self._cache_cats = {}

        #if (zmin, zmax, mmin) in self._cache_cats:
            #print(f"# Loaded from cache (zmin={zmin}, zmax={zmax}, mmin={mmin})")
        #    return self._cache_cats[(zmin, zmax, mmin)]

        if zlim is None:
            zlim = self.zlim

        zmin, zmax = zlim
        mmin, mmax = 10**np.array(logmlim)

        # First, get full domain info
        ze, zmid, Re = self.get_domain_info(zlim=self.zlim, Lbox=self.Lbox)
        Rc = bin_e2c(Re)
        dz = np.diff(ze)

        seeds = self.seed * np.arange(1, len(zmid)+1)
        seeds_h = self.seeds_halos * np.arange(1, len(zmid)+1)

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

        #if self.verbose:
        #    print("# FOV at front edge (z={:.1f}) of lightcone: {:.1f} degrees.".format(
        #        zmin, theta_max
        #    ))
        #    print("# FOV at back edge (z={:.1f}) of lightcone: {:.1f} degrees.".format(
        #        zmax, theta_min
        #    ))

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
            halos = self.get_halo_population(z=zmid[i], seed_box=seeds[i],
                seed=seeds_h[i], mmin=mmin, mmax=mmax,
                randomise_in_cell=randomise_in_cell)

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

        pbar.finish()

        #self._cache_cats[(zmin, zmax, mmin)] = ra, dec, red, mass

        return ra, dec, red, mass


    def get_redshift_chunks(self, zlim):
        """
        Return the edges of each co-eval cube as positioned along the LoS.
        """

        ze, zmid, Re = self.get_domain_info(zlim)

        chunks = [(zlo, ze[i+1]) for i, zlo in enumerate(ze[0:-1])]
        return chunks

    def get_zindex(self, z):
        """
        For a given redshift, return the index of the chunk that contains it
        in the LoS direction.
        """
        zall = self.get_redshift_chunks()
        zlo, zhi = np.array(zall).T
        iz = np.argmin(np.abs(z - zlo))
        if zlo[iz] > z:
            iz -= 1

        return iz

    def get_seds(self, z, idnum=0, wave_range=(800., 5e4), mmin=1e11,
        dlam=20, tol_logM=0.1):
        """

        """

        waves = np.arange(wave_range[0], wave_range[1]+dlam, dlam)

        Mh = self.get_field(z, 'mass', mmin=mmin)
        red = self.get_field(z, 'redshift', mmin=mmin)

            # Could supply "red" instead of "z" here to get some evolution.
        lum = self.sim.pops[idnum].get_sed(z, Mh, waves,
            stellar_mass=False)

        return waves, lum
