"""

LightCone.py

Author: Jordan Mirocha
Affiliation: Jet Propulsion Laboratory
Created on: Sun Dec  4 13:00:50 PST 2022

Description:

"""

import os
import gc
import time
import h5py
import numpy as np
from ..simulations import Simulation
from ..util.Stats import bin_e2c, bin_c2e
from scipy.integrate import cumtrapz, quad
from ..util.ProgressBar import ProgressBar
from ..util.Misc import numeric_types, get_hash
from scipy.spatial.transform import Rotation
from astropy.modeling.models import Sersic2D
from ..physics.Constants import sqdeg_per_std, cm_per_mpc, cm_per_m, \
    erg_per_s_per_nW, c, s_per_myr

try:
    from astropy.io import fits
except ImportError:
    pass

angles_90 = 90 * np.arange(4)

class LightCone(object): # pragma: no cover
    """
    This should be inherited by the other classes in this submodule.
    """

    def get_max_fov(self, zlim):
        """
        Determine the biggest field-of-view we can produce (without repeated
        structures) given the input box size (self.Lbox).

        Returns
        -------
        Maximal field of view [degrees] in linear dimension.
        """

        zlo, zhi = zlim

        # arcmin / Mpc -> deg / Mpc
        L = self.Lbox / self.sim.cosm.h70
        angl_per_Llo = self.sim.cosm.ComovingLengthToAngle(zlo, L) / 60.
        angl_per_Lhi = self.sim.cosm.ComovingLengthToAngle(zhi, L) / 60.

        return angl_per_Llo

    def get_max_timestep(self):
        """
        Based on the size of our box, return the time interval corresponding to
        along the z-axis for each chunk of our lightcone.
        """

        ze, zc, Re = self.get_domain_info()

        te = self.sim.cosm.t_of_z(ze) / s_per_myr

        return np.diff(te)

    def get_pixels(self, fov, pix=1, hdr=None):
        """
        For a given field of view [deg, linear dimension] and pixel scale `pix`
        [arcseconds], return arrays containing bin edges and centers in both
        dimensions, i.e., (RA_edges, RA_centers, DEC_edges, DEC_centers), all
        in degrees.
        """

        if type(fov) in numeric_types:
            fov = np.array([fov]*2)

        npixx = int(fov[0] * 3600 / pix)
        npixy = int(fov[1] * 3600 / pix)

        # Figure out the edges of the domain in RA and DEC (arcsec)
        ra0, ra1 = fov * 3600 * 0.5 * np.array([-1, 1])
        dec0, dec1 = fov * 3600 * 0.5 * np.array([-1, 1])

        # Pixel coordinates
        ra_e = np.arange(ra0, ra1 + pix, pix)
        ra_c = ra_e[0:-1] + 0.5 * pix
        dec_e = np.arange(dec0, dec1 + pix, pix)
        dec_c = dec_e[0:-1] + 0.5 * pix

        assert ra_c.size == npixx
        assert dec_c.size == npixy

        return ra_e / 3600., ra_c / 3600., dec_e / 3600., dec_c / 3600.

    @property
    def sim(self):
        if not hasattr(self, '_sim'):
            self._sim = Simulation(verbose=self.verbose, **self.kwargs)
        return self._sim

    @property
    def pops(self):
        if not hasattr(self, '_pops'):
            self._pops = self.sim.pops
        return self._pops

    @property
    def dx(self):
        if not hasattr(self, '_dx'):
            self._dx = self.Lbox / float(self.dims)
        return self._dx

    @property
    def tab_xe(self):
        """
        Edges of grid zones in Lbox / h [cMpc] units.
        """
        if not hasattr(self, '_xe'):
            self._xe = np.arange(0, self.Lbox+self.dx, self.dx)
        return self._xe

    @property
    def tab_xc(self):
        """
        Centers of grid zones in Lbox / h [cMpc] units.
        """
        return bin_e2c(self.tab_xe)

    @property
    def tab_z(self):
        if not hasattr(self, '_tab_z'):
            self._tab_z = np.arange(0, 30, 0.1)
        return self._tab_z

    @property
    def tab_dL(self):
        """
        Luminosity distance (for each self.tab_z) in cMpc.
        """
        if not hasattr(self, '_tab_dL'):
            self._tab_dL = np.array([self.sim.cosm.get_luminosity_distance(z) \
                for z in self.tab_z]) / cm_per_mpc
        return self._tab_dL

    def get_domain_info(self, zlim=None, Lbox=None):
        """
        Figure out how domain will be divided up along line of sight.

        Returns
        -------
        A tuple containing (chunk edges in redshift, chunk midpoints in redshift,
            chunk edges in comoving Mpc).

        """

        if Lbox is None:
            Lbox = self.Lbox

        # First, get redshifts. This is just for interpolation.
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

    def get_redshift_chunks(self, zlim):
        """
        Return the edges of each co-eval cube as positioned along the LoS.
        """

        ze, zmid, Re = self.get_domain_info(zlim)

        chunks = [(zlo, ze[i+1]) for i, zlo in enumerate(ze[0:-1])]
        return chunks

    def get_mass_chunks(self, logmlim, dlogm):
        mbins = np.arange(logmlim[0], logmlim[1], dlogm)
        return np.array([(mbin, mbin+dlogm) for mbin in mbins])

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

        # Version of Lbox in actual cMpc
        L = self.Lbox / self.sim.cosm.h70

        # First, get full domain info
        ze, zmid, Re = self.get_domain_info(zlim=self.zlim, Lbox=self.Lbox)
        Rc = bin_e2c(Re)
        dz = np.diff(ze)

        # Deterministically adjust the random seeds for the given mass range
        # and redshift range.
        #fmh = int(logmlim[0] + (logmlim[1] - logmlim[0]) / 0.1)

        # Figure out if we're getting the catalog of a single chunk
        chunk_id = None
        for i, Rlo in enumerate(Re[0:-1]):
            zlo, zhi = ze[i:i+2]

            if (zlo == zlim[0]) and (zhi == zlim[1]):
                chunk_id = i
                break

        ##
        # Setup random seeds for random rotations and translations
        np.random.seed(self.seed_rot)
        r_rot = np.random.randint(0, high=4, size=(len(Re)-1)*3).reshape(
            len(Re)-1, 3
        )

        np.random.seed(self.seed_tra)
        r_tra = np.random.rand(len(Re)-1, 3)

        ##
        # Print-out information about FOV
        # arcmin / Mpc -> deg / Mpc
        theta_zmin = self.sim.cosm.ComovingLengthToAngle(zmin, 1) * L / 60.
        theta_zmax = self.sim.cosm.ComovingLengthToAngle(zmax, 1) * L / 60.

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

            seed_kwargs = self.get_seed_kwargs(i, logmlim)

            # Contains (x, y, z, mass)
            # Note that x, y, z are in cMpc / h units, not actual cMpc.
            halos = self.get_halo_population(z=zmid[i],
                mmin=mmin, mmax=mmax, verbose=verbose, idnum=idnum,
                **seed_kwargs)

            if halos[0].size == 0:
                ra = dec = red = mass = None
                continue

            # Might change later if we do domain decomposition
            x0 = y0 = z0 = 0.0
            dx = dy = dz = self.Lbox

            ##
            # Perform random flips and translations here
            if self.apply_rotations:

                _x_, _y_, _z_, _m_ = halos

                # Put positions in space centered on (0,0,0), i.e.,
                # [(-0.5 * dx, 0.5 * dx), (-0.5 * dy, 0.5 * dy), etc.]
                # not [(x0,x0+dx), (y0,y0+dy), (z0,z0+dz)]
                _x = _x_ - (x0 + 0.5 * dx)
                _y = _y_ - (y0 + 0.5 * dy)
                _z = _z_ - (z0 + 0.5 * dz)

                # This is just the format required by Rotation below.
                _view = np.array([_x, _y, _z]).T

                # Loop over axes
                for k in range(3):

                    # Force new viewing angles to be orthogonal to box faces
                    r = r_rot[i,k]
                    _theta = angles_90[r] * np.pi / 180.

                    axis = np.zeros(3)
                    axis[k] = 1

                    rot = Rotation.from_rotvec(_theta * axis)
                    _view = rot.apply(_view)

                # Read in our new 'view' of the catalog, undo the shift
                # so we're back in [(x0,x0+dx), (y0,y0+dy), (z0,z0+dz)] region.
                _x, _y, _z = _view.T
                _x += (0.5 * dx)
                _y += (0.5 * dy)
                _z += (0.5 * dz)

                halos = [_x, _y, _z, _m_]

            else:
                pass

            ##
            # Random translations
            if self.apply_translations:
                _x_, _y_, _z_, _m_ = halos

                # Put positions in space centered on (0,0,0), i.e.,
                # [(-0.5 * dx, 0.5 * dx), (-0.5 * dy, 0.5 * dy), etc.]
                # not [(x0,x0+dx), (y0,y0+dy), (z0,z0+dz)]
                _x = _x_.copy()
                _y = _y_.copy()
                _z = _z_.copy()

                _x += r_tra[i,0] * dx
                overx = _x > dx
                _x[overx] = _x[overx] - dx

                _y += r_tra[i,1] * dy
                overy = _y > dy
                _y[overy] = _y[overy] - dy

                _z += r_tra[i,2] * dz
                overz = _z > dz
                _z[overz] = _z[overz] - dz

                halos = [_x, _y, _z, _m_]

            else:
                pass

            ##
            # Convert to (ra, dec, redshift) coordinates.
            # Note: the conversion from cMpc/h to cMpc occurs inside
            # _get_catalog_from_coeval here:
            _ra, _de, _red = self._get_catalog_from_coeval(halos, z0=zlo)
            _m = halos[-1]

            okr = np.logical_and(_ra <  0.5 * theta_zmin,
                                 _ra > -0.5 * theta_zmin)
            okd = np.logical_and(_de <  0.5 * theta_zmin,
                                 _de > -0.5 * theta_zmin)
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

    def _get_catalog_from_coeval(self, halos, z0=0.2):
        """
        Make a catalog in lightcone coordinates (RA, DEC, redshift).

        .. note :: RA and DEC output in degrees.

        """

        xmpc, ympc, zmpc, mass = halos

        # Shift coordinates to +/- 0.5 * Lbox
        xmpc = (xmpc - 0.5 * self.Lbox) / self.sim.cosm.h70
        ympc = (ympc - 0.5 * self.Lbox) / self.sim.cosm.h70

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


    def thin_sample(self, max_sources=None):

        if (max_sources is not None):
            if (ct == 0) and (max_sources >= Mh.size):
                # In this case, we can accommodate all the galaxies in
                # the catalog, so don't do anything yet.
                pass
            else:
                # Flag entries until we hit target.
                # This is not efficient but oh well.
                for h in range(Mh.size):
                    ok[h] = 0

                    if ok.sum() == max_sources:
                        break

                # This will be the final iteration.
                if ct + ok.sum() == max_sources:
                    self._hit_max_sources = True

    def create_directory_structure(self, path='.', suffix=None):
        """
        Our model is:

        -><prefix>_fov_<FOV in deg>_pix_<pixel scale in arcsec>_L<box/cMpc/h>_N<dims>/
        ->  README
        ->  <parent directory name>.tar.gz
                *(contains) final mock for each channel summed over all z, M, pops.
        ->  ch_<spectral channel edges>
        ->    z_<redshift lower edge, upper edge>_M_<halo mass lower edge, upper>_pid_<pop ID number>.fits
        ->

        """

        base_dir = '{}/{}_fov_{:.1f}_pix_{:.1f}'.format(path, self.prefix,
            self.fov, self.pix, self.Lbox)

    def get_base_dir(self, fov, pix, path='.', suffix=None):
        """
        Generate the name for the root directory where all mocks for a given
        model will go.

        Our model is:

        -><prefix>_fov_<FOV/deg>_pix_<pixel scale / arcsec>_L<box/cMpc/h>_N<dims>/
        ->  README

        Inside this directory, there will be many subdirectories: one for each
        spectral channel of interest.

        There will also be a series of .fits (or .hdf5) files, which represent
        "final" maps, i.e., those that are summed over redshift and mass chunks,
        and also summed over all source populations.

        """


        s = '{}/{}_fov_{:.1f}_pix_{:.1f}_L{:.0f}_N{:.0f}'.format(path,
            self.prefix, fov, pix, self.Lbox, self.dims)

        if suffix is None:
            print("# WARNING: might be worth providing `suffix` as additional identifier.")
        else:
            s += f'_{suffix}'

        return s

    #@profile
    def get_map(self, fov, pix, channel, logmlim, zlim, idnum=0,
        include_galaxy_sizes=False, size_cut=0.5, dlam=20.,
        use_pbar=True, verbose=False, max_sources=None, buffer=None, **kwargs):
        """
        Get a map for a single channel, redshift chunk, mass chunk, and
        source population.

        .. note :: To get a 'full' map, containing contributions from multiple
            redshift and mass chunks, and potentially populations, see the
            wrapper routine `generate_maps`.

        Parameters
        ----------
        fov : int, float
            Field of view (single dimension) in degrees.
        pix : int, float
            Pixel scale in arcseconds.
        channel : tuple, list, np.ndarray
            Edges of the spectral channel of interest [microns].
        zlim : tuple, list, np.ndarray
            Optional redshift range. If None, will include all objects in the
            catalog.

        Returns
        -------
        If `buffer` is None, will return a map in our internal erg/s/sr. If
        `buffer` is supplied, will increment that array, same units.
        Any conversion of units (using `map_units`) takes place *only* in the
        `generate_maps` routine.
        """

        pix_deg = pix / 3600.
        sr_per_pix = pix_deg**2 / sqdeg_per_std

        assert fov * 3600 / pix % 1 == 0, \
            "FOV must be integer number of pixels wide!"

        # In degrees
        if type(fov) in numeric_types:
            fov = np.array([fov]*2)

        assert np.diff(fov) == 0, "Only square FOVs allowed right now."

        zall = self.get_redshift_chunks(zlim=self.zlim)
        assert zlim in zall

        # Figure out the edges of the domain in RA and DEC (degrees)
        # Pixel coordinates
        ra_e, ra_c, dec_e, dec_c = self.get_pixels(fov, pix=pix)

        Npix = [ra_c.size, dec_c.size]

        # Initialize empty map
        img = buffer
        #if buffer is not None:
        #    img = buffer
        #elif save_intermediate:
        #    img = np.zeros([len(zall)] + Npix, dtype=np.float64)
        #else:
        #    img = np.zeros([1] + Npix, dtype=np.float64)

        ##
        # Might take awhile.
        #pb = ProgressBar(len(zall),
        #    name="img(z; Mh>={:.1f}, Mh<{:.1f})".format(logmlim[0], logmlim[1]),
        #    use=use_pbar)
        #pb.start()

        # Track max_sources
        _hit_max_sources = False

        ct = 0

        zlo, zhi = zlim

        ##
        # Loop over redshift chunks and assemble image.
        #for _iz_, (zlo, zhi) in enumerate(zall):

        #    if _hit_max_sources:
        #        break

        #    if (zhi <= zlim[0]) or (zlo >= zlim[1]):
        #        continue

        _z_ = np.mean([zlo, zhi])

            #   if save_intermediate:
            #       iz = _iz_
            #   else:
            #       iz = 0

        ra, dec, red, Mh = self.get_catalog(zlim=(zlo, zhi),
            logmlim=logmlim, idnum=idnum, verbose=verbose)

        # Could be empty chunks for very massive halos and/or early times.
        if ra is None:
            return #None, None, None

        ##
        # Figure out which bin each galaxy is in.
        ra_bin = np.digitize(ra, bins=ra_e)
        dec_bin = np.digitize(dec, bins=dec_e)
        mask_ra = np.logical_or(ra_bin == 0, ra_bin == Npix[0]+1)
        mask_de = np.logical_or(dec_bin == 0, dec_bin == Npix[1]+1)
        ra_ind = ra_bin - 1
        de_ind = dec_bin - 1

        # Mask out galaxies that aren't in our desired image plane.
        okp = np.logical_not(np.logical_or(mask_ra, mask_de))

        # Filter out galaxies outside specified redshift range.
        # [usually don't do this within chunk, but hey, functionality there]
        if zlim is not None:
            okz = np.logical_and(red >= zlo, red < zhi)
            ok = np.logical_and(okp, okz)
        else:
            ok = okp

        # For debugging and tests, we can dramatically limit the
        # number of sources. Thin out the herd here.
        if (max_sources is not None):
            if (ct == 0) and (max_sources >= Mh.size):
                # In this case, we can accommodate all the galaxies in
                # the catalog, so don't do anything yet.
                pass
            else:
                # Flag entries until we hit target.
                # This is not efficient but oh well.
                for h in range(Mh.size):
                    ok[h] = 0

                    if ok.sum() == max_sources:
                        break

                # This will be the final iteration.
                if ct + ok.sum() == max_sources:
                    _hit_max_sources = True

        #if self.verbose:
        #    print("Masked fraction: {:.5f}".format((ok.size - ok.sum()) / float(ok.size)))

        # May have empty chunks, e.g., very massive halos and/or very
        # high redshifts.
        if not np.any(ok):
            return #None, None, None

        # Increment counter
        ct += ok.sum()

        ##
        # Isolate OK entries.
        ra = ra[ok==1]
        dec = dec[ok==1]
        red = red[ok==1]
        Mh = Mh[ok==1]
        ra_ind = ra_ind[ok==1]
        de_ind = de_ind[ok==1]

        # Get geometrical dilution factor
        corr = 1. / 4. / np.pi \
            / (np.interp(red, self.tab_z, self.tab_dL) * cm_per_mpc)**2

        # Get flux from each object. Units = erg/s/cm^2/Ang.
        # Already accounting for geometrical dilution but provided at
        # rest wavelengths, so must divide by (1+z) to get flux in observer
        # frame.

        # Find bounding wavelength range to limit memory consumption, i.e.,
        # don't grab rest-frame SED outside of range needed by observer.
        # This really only helps if the user has instituted a cut in
        # redshift that eliminates a significant fraction of any chunk.
        _zlo = zlim[0] if zlim is not None else red.min()
        _zhi = zlim[1] if zlim is not None else red.max()
        _wlo = channel[0] * 1e4 / (1. + min(red.max(), _zhi))
        _whi = channel[1] * 1e4 / (1. + max(red.min(), _zlo))

        # [waves] = Angstroms rest-frame, [seds] = erg/s/A.
        # Shape of seds is (N galaxies, N wavelengths)
        # Shape of (ra, dec, red) is just (Ngalaxies)
        waves = np.arange(_wlo, _whi+dlam, dlam)

        # Need to supply band or window?
        seds = self.sim.pops[idnum].get_spec(_z_, waves, Mh=Mh,
            units_out='erg/s/Ang', window=dlam+1 if dlam % 2 == 0 else dlam)

        owaves = waves[None,:] * (1. + red[:,None])

        # Frequency "squashing", i.e., our 'per Angstrom' interval is
        # different in the observer frame by a factor of 1+z.
        flux = corr[:,None] * seds[:,:] / (1. + red[:,None])

        ##
        # Need some extra info to do more sophisticated modeling...
        if include_galaxy_sizes:

            Ms = self.sim.pops[idnum].get_smhm(z=red, Mh=Mh) * Mh
            Rkpc = self.pops[0].get_size(z=red, Ms=Ms)

            R_sec = np.zeros_like(Rkpc)
            for kk in range(red.size):
                R_sec[kk] = self.sim.cosm.ProperLengthToAngle(red[kk], Rkpc[kk] * 1e-3)
            R_sec *= 60.

            # Uniform for now.
            nsers = np.random.random(size=Rkpc.size) * 5.9 + 0.3
            pa = np.random.random(size=Rkpc.size) * 360

            # Ellipticity = 1 - b/a
            ellip = np.random.random(size=Rkpc.size)

            # Will paint anything half-light radius greater than a pixel
            if size_cut == 0.5:
                R_X = R_sec
            # General option: paint anything with size, defined as the
            # radius containing `size_cut` fraction of the light, that
            # exceeds a pixel.
            else:
                rarr = np.logspace(-1, 1.5, 500)
                #cog_sfg = [self.sim.pops[idnum].get_sersic_cog(r,
                #    n=nsers[h]) \
                #    for r in rarr]

                rmax = [self.sim.pops[idnum].get_sersic_rmax(size_cut,
                    nsers[h]) for h in range(Rkpc.size)]

                R_X = np.array(rmax) * R_sec

            #R_sec = Rkpc * self.cosmo.arcsec_per_kpc_proper(red).to_value()

            # Size in degrees
            R_deg = R_sec / 3600.
            R_pix = R_deg / pix_deg

            R_X /= (3600 * pix_deg)

            # All in degrees
            x0, y0 = ra, dec
            a, b = R_deg, R_deg

            rr, dd = np.meshgrid(ra_c / pix_deg, dec_c / pix_deg)

        ##
        # Extended emission from IHL, satellites
        if self.sim.pops[idnum].is_diffuse:

            _iz = np.argmin(np.abs(_z_ - self.sim.pops[idnum].halos.tab_z))

            # Remaining dimensions (Mh, R)
            Sall = self.sim.pops[idnum].halos.tab_Sigma_nfw[_iz,:,:]

            mpc_per_arcmin = self.sim.cosm.AngleToComovingLength(_z_,
                pix / 60.)

            rr, dd = np.meshgrid(ra_c * 60 * mpc_per_arcmin,
                                dec_c * 60 * mpc_per_arcmin)


            # Tabulate surface density as a function of displacement
            # and halo mass
            Rmi, Rma = -3, 1
            dlogR = 0.25
            Rall = 10**np.arange(Rmi, Rma+dlogR, dlogR)
            Mall = self.sim.pops[idnum].halos.tab_M


        ##
        # Actually sum fluxes from all objects in image plane.
        for h in range(ra.size):

            #if not ok[h]:
            #    continue

            # Where this galaxy lives in pixel coordinates
            i, j = ra_ind[h], de_ind[h]

            # Compute flux more precisely than summing by differencing
            # the cumulative integral over the band at the channel edges.
            cflux = cumtrapz(flux[h] * owaves[h], x=np.log(owaves[h]),
                initial=0.0)

            # Remember: `owaves` is in Angstroms, `channel` elements are
            # in microns.
            _flux_ = np.interp(channel[1] * 1e4, owaves[h], cflux) \
                   - np.interp(channel[0] * 1e4, owaves[h], cflux)

            # HERE: account for fact that galaxies aren't point sources.
            # [optional]
            if include_galaxy_sizes and R_X[h] >= 1:

                model_SB = Sersic2D(amplitude=1., r_eff=R_pix[h],
                    x_0=ra[h] / pix_deg, y_0=dec[h] / pix_deg,
                    n=nsers[h], theta=pa[h] * np.pi / 180.,
                    ellip=ellip[h])

                # Fractional contribution to total flux
                I = model_SB(rr, dd)
                tot = I.sum()

                if tot == 0:
                    img[i,j] += _flux_
                else:
                    img[:,:] += _flux_ * I / tot

            elif self.sim.pops[idnum].is_diffuse:

                # Image of distances from halo center
                r0 = ra_c[i] * 60 * mpc_per_arcmin
                d0 = dec_c[j] * 60 * mpc_per_arcmin
                Rarr = np.sqrt((rr - r0)**2 + (dd - d0)**2)

                # In Msun/cMpc^3

                #Rall = np.linspace(Rarr.min(), Rarr.max() * 1.1, 1000)
                #Sall = np.zeros_like(Rall)
                #for i, _R_ in enumerate(Rall):
                #    Sall[i] = Sigma(_R_)

                # Interpolate between tabulated solutions.
                iM = np.argmin(np.abs(Mh[h] - Mall))

                I = np.interp(np.log10(Rarr), np.log10(Rall), Sall[iM,:])

                tot = I.sum()

                if tot == 0:
                    img[i,j] += _flux_
                else:
                    img[:,:] += _flux_ * I / tot

            ##
            # Otherwise just add flux to single pixel
            else:
                img[i,j] += _flux_

        #pb.update(_iz_)

        ##
        # Clear out some memory sheesh
        del seds, flux, _flux_, ra, dec, red, Mh, ok, ra_ind, de_ind, \
            mask_ra, mask_de
        gc.collect()

        #pb.finish()

        ##
        # Hmmm
        #if np.any(np.isnan(img)):
        #    print("* WARNING: {:.4f}% of pixels are NaN! Removing...".format(
        #        100 * np.isnan(img).sum() / float(img.size)
        #    ))
        #    img[np.isnan(img)] = 0.0

        #self._cache_maps[(channel, zlim, pix, include_galaxy_sizes, kwtup)] = \
        #    ra_e, dec_e, img * cm_per_m**2 / erg_per_s_per_nW

        #gc.collect()


            #return ra_e, dec_e, img
        #else:
        #    return None, None, None

    def get_fn(self, channel, logmlim, zlim=None, fmt='hdf5', final=False,
        channel_name=None):
        """

        """

        if zlim is None:
            zlim = self.zlim

        fn = 'z_{:.2f}_{:.2f}'.format(*zlim)
        fn += '_M_{:.1f}_{:.1f}'.format(*logmlim)

        if final:
            fn += '/'
        else:
            fn += '_'

        if channel is not None:
            if channel_name is not None:
                fn += channel_name
            else:
                try:
                    fn += '{:.2f}_{:.2f}'.format(*channel)
                except:
                    fn += f'{channel}'

        return fn + '.' + fmt

    def get_README(self, fov, pix, channels, logmlim, zlim, path='.',
        suffix=None, fmt='fits', save=False, is_map=True, channel_names=None):
        """

        """

        s = "#" * 78
        s += '\n# README\n'
        s += "#" * 78
        s +=  "\n# This is an automatically-generated file! \n"
        s += "# In it, we list maps created with a given set of parameters.\n"
        s += "# Note: this is a listing of files that will exist when the "
        s += "map-making is \n# COMPLETE, i.e., they may not all exist yet.\n"
        s += "#" * 78
        s += "\n"
        s += "# channel name [optional]; central wavelength (microns); "
        s += "channel lower edge (microns) ; channel upper edge (microns) ; "
        s += "filename \n"

        # Loop over channels
        if is_map:
            final_dir = 'final_maps'
            rstr = 'maps'
        else:
            final_dir = 'final_cats'
            rstr = 'cats'

        if channels in [None, 'Mh', ['Mh']]:
            channels = ['Mh']

        if channel_names is None:
            if is_map:
                channel_names = [None] * len(channels)
            else:
                channel_names = channels

        # Loop over channels and record filenames. If not supplied, must be
        # galaxy catalog containing only positions (no photometry).
        for h, channel in enumerate(channels):
            fn = self.get_fn(channel, logmlim, zlim=zlim,
                fmt=fmt, final=True, channel_name=channel_names[h])

            if is_map:
                s += "{} ; {:.6f} ; {:.6f} ; {:.6f} ; {}/{} \n".format(
                    channel_names[h],
                    np.mean(channel), channel[0], channel[1], final_dir, fn)
            else:
                s += "{} ; {}/{} \n".format(channel, final_dir, fn)
        #else:
        #    assert not is_map
        #    fn = self.get_fn(None, logmlim, zlim=zlim,
        #        fmt=fmt, final=True)
        #    s += "None ; {}/{}".format(final_dir, fn)


        ##
        # Write
        if save:
            base_dir = self.get_base_dir(fov=fov, pix=pix, path=path,
                suffix=suffix)
            with open(f'{base_dir}/README_{rstr}', 'w') as f:
                f.write(s)
            print(f"# Wrote {base_dir}/README")

        return s

    def generate_lightcone(self, fov, pix, channels):
        """
        Generate a lightcone.
        """
        pass

    def generate_cats(self, fov, pix, channels, logmlim, dlogm=0.5, zlim=None,
        include_galaxy_sizes=False, dlam=20, path='.', channel_names=None,
        suffix=None, fmt='hdf5', hdr={}, max_sources=None,
        include_pops=None, clobber=False, verbose=False, **kwargs):
        """
        Generate galaxy catalogs.

        Parameters
        ----------
        fov : int, float
            Field of view (single dimension) in degrees, so total area is
            FOV^2/deg^2.
        pix : int, float
            Pixel scale in arcseconds.

        """

        # Create root directory if it doesn't already exist.
        base_dir = self.get_base_dir(fov=fov, pix=pix, path=path, suffix=suffix)
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        if not os.path.exists(base_dir + '/final_cats'):
            os.mkdir(base_dir + '/intermediate_cats')
            os.mkdir(base_dir + '/final_cats')

        # At least save halo mass since we get it for free.
        if (channels is None) or (channels == ['Mh']):
            run_phot = False
            channels = ['Mh']
        else:
            run_phot = True

        if channel_names is None:
            channel_names = channels

        ##
        # Write a README file that says what all the final products are
        README = self.get_README(fov=fov, pix=pix, channels=channels,
            logmlim=logmlim, zlim=zlim, path=path, fmt=fmt,
            channel_names=channel_names,
            suffix=suffix, save=True, is_map=False)

        if zlim is None:
            zlim = self.zlim

        if include_pops is None:
            include_pops = range(0, len(self.sim.pops))

        assert fov * 3600 / pix % 1 == 0, \
            "FOV must be integer number of pixels wide!"

        npix = int(fov * 3600 / pix)
        zchunks = self.get_redshift_chunks(self.zlim)
        zcent, ze, Re = self.get_domain_info(self.zlim)
        mchunks = self.get_mass_chunks(logmlim, dlogm)

        final_dir = base_dir + '/final_cats'
        _final_sub = self.get_fn(None, logmlim, zlim=zlim, fmt=fmt,
            final=True, channel_name=None)
        final_sub = _final_sub[0:_final_sub.rfind('/')]
        if not os.path.exists(f'{final_dir}/{final_sub}'):
            os.mkdir(f'{final_dir}/{final_sub}')

        ##
        # Loop over populations, make catalogs.
        for h, channel in enumerate(channels):
            ra_allp = []
            dec_allp = []
            red_allp = []
            dat_allp = []
            for popid, pop in enumerate(self.sim.pops):
                if popid not in include_pops:
                    continue

                intmd_dir = base_dir + f'/intermediate_cats/pop_{popid}'
                if not os.path.exists(intmd_dir):
                    os.mkdir(intmd_dir)

                # Go from low-z to high-z
                ra_allz = []
                dec_allz = []
                red_allz = []
                dat_allz = []
                for i, (zlo, zhi) in enumerate(zchunks):

                    # Look for pre-existing file.

                    if (zhi <= zlim[0]) or (zlo >= zlim[1]):
                        continue

                    ra_z = []
                    dec_z = []
                    red_z = []
                    dat_z = []

                    ct = 0

                    # Go from high to low in mass
                    for j, (mlo, mhi) in enumerate(mchunks[-1::-1]):

                        if max_sources is not None:
                            if ct >= max_sources:
                                break

                        fn = self.get_fn(channel, (mlo, mhi), (zlo, zhi),
                            fmt=fmt, final=False, channel_name=channel_names[h])

                        fn = intmd_dir + '/' + fn

                        # Try to read from disk.
                        if os.path.exists(fn) and (not clobber):
                            print(f"Found {fn}. Set clobber=True to overwrite.")
                            #_Inu = self._load_cat(fn)
                            continue

                        ra, dec, red, Mh = self.get_catalog(zlim=(zlo, zhi),
                            logmlim=(mlo, mhi), idnum=popid)

                        # Could be empty chunks for very massive halos and/or early times.
                        if ra is None:
                            continue

                        # Hack out galaxies outside our requested lightcone.
                        ok = np.logical_and(np.abs(ra)  < fov / 2.,
                                            np.abs(dec) < fov / 2.)

                        if ok.sum() == 0:
                            continue

                        # Limit number of sources, just for testing.
                        if (max_sources is not None):
                            if (ct == 0) and (max_sources >= Mh.size):
                                # In this case, we can accommodate all the galaxies in
                                # the catalog, so don't do anything yet.
                                pass
                            else:
                                # Flag entries until we hit target.
                                # This is not efficient but oh well.
                                for _h in range(Mh.size):
                                    ok[_h] = 0

                                    if ok.sum() == max_sources:
                                        break

                                # This will be the final iteration.
                                if ct + ok.sum() == max_sources:
                                    self._hit_max_sources = True


                        # Isolate OK entries.
                        ra = ra[ok==1]
                        dec = dec[ok==1]
                        red = red[ok==1]
                        Mh = Mh[ok==1]

                        ct += ok.sum()

                        ra_z.extend(list(ra))
                        dec_z.extend(list(dec))
                        red_z.extend(list(red))

                        ##
                        # Unpack channel info
                        # Could be name of field, e.g., 'Mh', 'SFR', 'Mstell',
                        # photometric info, e.g., ('roman', 'F087'),
                        # or special quantities like Ly-a EW or luminosity.
                        # Note: if pops[idnum] is a GalaxyEnsemble object
                        if channel in ['Mh', 'Ms', 'SFR']:
                            dat = Mh
                        elif channel.lower().startswith('ew'):
                            raise NotImplemented('help')
                        else:
                            cam, filt = channel.split('_')

                            _filt, mags = self.sim.pops[popid].get_mags(zcent[i],
                                absolute=False, cam=cam, filters=[filt],
                                Mh=Mh)

                            dat = np.atleast_1d(mags.squeeze())

                        ##
                        # Save
                        # Future: save ra, dec, redshift separately? Could
                        # get heavy.
                        self.save_cat(fn, (ra, dec, red, dat),
                            channel, (zlo, zhi), (mlo, mhi),
                            fov, pix=pix, fmt=fmt, hdr=hdr,
                            clobber=clobber)

                        dat_z.extend(list(dat))

                    ##
                    # Done with all mass chunks

                    # Save intermediate chunk: all masses, single redshift chunk
                    fnt = self.get_fn(channel, logmlim=logmlim,
                        zlim=(zlo, zhi), fmt=fmt, final=False,
                        channel_name=channel_names[h])

                    fnt = intmd_dir + '/' + fnt

                    self.save_cat(fnt, (ra_z, dec_z, red_z, dat_z),
                        channel, (zlo, zhi), (mlo, mhi),
                        fov, pix=pix, fmt=fmt, hdr=hdr, clobber=clobber)

                    ra_allz.extend(ra_z)
                    dec_allz.extend(dec_z)
                    red_allz.extend(red_z)
                    dat_allz.extend(dat_z)

                    del ra_z, dec_z, red_z, dat_z

                ##
                # Done with all redshift chunks for this population.
                fnp = self.get_fn(channel, logmlim=logmlim,
                    zlim=zlim, fmt=fmt, final=False,
                    channel_name=channel_names[h])

                fnp = intmd_dir + '/' + fnp

                self.save_cat(fnp, (ra_allz, dec_allz, red_allz, dat_allz),
                    channel, zlim, logmlim,
                    fov, pix=pix, fmt=fmt, hdr=hdr, clobber=clobber)

                ra_allp.extend(ra_allz)
                dec_allp.extend(dec_allz)
                red_allp.extend(red_allz)
                dat_allp.extend(dat_allz)

            del ra_allz, dec_allz, red_allz, dat_allz

            ##
            # Combine catalogs over populations?
            fnf = self.get_fn(channel, logmlim=logmlim,
                zlim=zlim, fmt=fmt, final=True,
                channel_name=channel_names[h])

            fnf = final_dir + '/' + fnf

            print(f"Trying to save final catalog to {fnf}")
            print(f"final_dir={final_dir}, fnf={fnf}, channel={channel}")
            self.save_cat(fnf, (ra_allp, dec_allp, red_allp, dat_allp),
                channel, zlim, logmlim,
                fov, pix=pix, fmt=fmt, hdr=hdr, clobber=clobber)

        ##
        # Done with channels

        ##
        # Wipe slate clean
        f = open(f"{final_dir}/README", 'w')
        f.write("# filter ; populations included \n")
        f.close()

        s = f"{channel} ; "
        for pop in include_pops:
            s += f"{pop},"

        s = s.rstrip(',') + '\n'

        with open(f"{final_dir}/README", 'a') as f:
            f.write(s)

        if verbose:
            print(f"# Wrote {final_dir}/README")

    def generate_maps(self, fov, pix, channels, logmlim, dlogm=0.5, zlim=None,
        include_galaxy_sizes=False, size_cut=0.9, dlam=20, path='.',
        suffix=None, fmt='hdf5', hdr={}, map_units='MJy', channel_names=None,
        include_pops=None, clobber=False, max_sources=None,
        keep_layers=True, use_pbar=True, verbose=False, **kwargs):
        """
        Write maps in one or more spectral channels to disk.

        Naming convention is:

        "<prefix>_<a bunch of other stuff>" where other stuff is:

            + _ch_<channel lower edge in microns>_<upper edge>
            + _pix_<pixel scale in arcseconds>
            + _fov_<field of view in degrees on a side>
            + _L<box size of "co-eval cubes" in cMpc / h>
            + _N<number of grid zones on a side for each co-eval cube>
            + _z_<zlo>_<zhi>
            + _M_<log10(halo mass / Msun) lower limit>_<upper limit>
            + <suffix>

        The user is encouraged to add descriptive `prefix` and `suffix` that
        will be prepended/appended to this string.

        Returns
        -------
        Right now, nothing. Just saves files to disk.

        """

        # Create root directory if it doesn't already exist.
        base_dir = self.get_base_dir(fov=fov, pix=pix, path=path, suffix=suffix)
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        if not os.path.exists(base_dir + '/final_maps'):
            os.mkdir(base_dir + '/intermediate_maps')
            os.mkdir(base_dir + '/final_maps')

        ##
        # Write a README file that says what all the final products are
        README = self.get_README(fov=fov, pix=pix, channels=channels,
            logmlim=logmlim, zlim=zlim, path=path, channel_names=channel_names,
            suffix=suffix, save=True)

        if np.array(channels).ndim == 1:
            channels = np.array([channels])

        if channel_names is None:
            channel_names = [None] * len(channels)

        if zlim is None:
            zlim = self.zlim

        if include_pops is None:
            include_pops = range(0, len(self.sim.pops))

        assert fov * 3600 / pix % 1 == 0, \
            "FOV must be integer number of pixels wide!"

        npix = int(fov * 3600 / pix)
        zchunks = self.get_redshift_chunks(self.zlim)
        mchunks = self.get_mass_chunks(logmlim, dlogm)
        pchunks = include_pops

        # Assemble list of map layers to run.
        all_chunks = []
        for h, popid in enumerate(pchunks):
            for i, channel in enumerate(channels):
                for j, zchunk in enumerate(zchunks):

                    # Option to limit redshift range.
                    zlo, zhi = zchunk
                    if (zhi <= zlim[0]) or (zlo >= zlim[1]):
                        continue

                    for k, mchunk in enumerate(mchunks):
                        all_chunks.append((popid, channel, channel_names[i],
                            zchunk, mchunk))

        # Make directories to save final maps
        final_dir = base_dir + '/final_maps'
        _final_sub = self.get_fn(None, logmlim, zlim=zlim, fmt=fmt,
            final=True)
        final_sub = _final_sub[0:_final_sub.rfind('/')]
        if not os.path.exists(f'{final_dir}/{final_sub}'):
            os.mkdir(f'{final_dir}/{final_sub}')

        ##
        # Remember: using cgs units internally. Compute conversion factor to
        # users favorite units (within reason).
        # 1 Jy = 1e-23 erg/s/cm^2/sr
        if map_units.lower() == 'si':
            # aka (1e2)^2 / 0.01 = 1e6
            f_norm = cm_per_m**2 / erg_per_s_per_nW
        elif map_units.lower() == 'cgs':
            f_norm = 1.
        elif map_units.lower() == 'mjy':
            # 1 MJy = 1e6 Jy = 1e6 * 1e-23 erg/s/cm^2/sr = 1e17 MJy / cgs units
            f_norm = 1e17
        else:
            raise ValueErorr(f"Unrecognized option `map_units={map_units}`")

        if verbose:
            print(f"# Generating {len(all_chunks)} individual map layers...")

        pb = ProgressBar(len(all_chunks),
            name="img(Mh>={:.1f}, Mh<{:.1f}, z>={:.2f}, z<{:.2f})".format(
                logmlim[0], logmlim[1], zlim[0], zlim[1]),
            use=use_pbar)
        pb.start()

        # Make preliminary buffer for channel map
        cimg = np.zeros([npix]*2)

        ##
        # Start doing work.
        for h, chunk in enumerate(all_chunks):

            # Unpack info about this chunk
            popid, channel, chname, zchunk, mchunk = chunk

            ##
            # Should check for final maps first
            fn_fin = self.get_fn(channel, logmlim=logmlim,
                zlim=zlim, fmt=fmt, final=True, channel_name=chname)
            fn_fin = final_dir + '/' + fn_fin

            if os.path.exists(fn_fin) and (not clobber):
                print(f"# Found final map {fn_fin}. Set clobber=True to re-generate")
                continue

            pb.update(h)

            # Will need channel width in Hz to recover specific intensities
            # averaged over band.
            nu = c * 1e4 / np.mean(channel)
            dnu = (c * 1e4 / channel[0]) - (c * 1e4 / channel[1])

            # Create directory for intermediate products if it doesn't
            # already exist.
            intmd_dir = base_dir + f'/intermediate_maps/pop_{popid}'
            if not os.path.exists(intmd_dir):
                os.mkdir(intmd_dir)

            # What buffer should we increment?
            if (not keep_layers):
                buffer = cimg
            else:
                buffer = np.zeros([npix]*2)

            # Generate map
            # Internal flux units are cgs [erg/s/Hz/sr]
            # but get_map returns a channel-integrated flux, so just erg/s/sr
            self.get_map(fov, pix, channel,
                logmlim=mchunk, zlim=zchunk,
                include_galaxy_sizes=include_galaxy_sizes,
                size_cut=size_cut,
                dlam=dlam, use_pbar=False,
                max_sources=max_sources,
                buffer=buffer, verbose=verbose,
                **kwargs)

            # Save every mass chunk within every redshift chunk if the user
            # says so.
            if keep_layers:
                fn = self.get_fn(channel, mchunk, zchunk,
                    fmt=fmt, final=False, channel_name=chname)
                fn = intmd_dir + '/' + fn
                self.save_map(fn, buffer * f_norm / dnu,
                    channel, zchunk, logmlim, fov,
                    pix=pix, fmt=fmt, hdr=hdr, map_units=map_units,
                    verbose=verbose, clobber=clobber)

                # Increment map for this z chunk
                cimg += buffer

            ##
            # Otherwise, figure out what (if anything) needs to be
            # written to disk now.


            done_w_chan = False
            done_w_pop = False
            done_w_z = False

            # Figure out what files need to be written
            if h == len(all_chunks) - 1:
                # Write everything on final iteration.
                done_w_chan = done_w_pop = done_w_z = True
            else:

                pnext, cnext, nnext, znext, mnext = all_chunks[h+1]

                if (channel[0] != cnext[0]):
                    done_w_chan = True
                if (popid != pnext):
                    done_w_pop = True
                if znext[0] < zchunk[0]:
                    done_w_z = True

            # Done with entire z chunk, i.e., all halo mass chunks.
            if done_w_z:
                fn = self.get_fn(channel, logmlim, zchunk,
                    fmt=fmt, final=False, channel_name=chname)
                fn = intmd_dir + '/' + fn
                self.save_map(fn, cimg * f_norm / dnu,
                    channel, zchunk, logmlim, fov,
                    pix=pix, fmt=fmt, hdr=hdr, map_units=map_units,
                    verbose=verbose, clobber=clobber)

            # Means we've done all redshifts and all masses
            if done_w_chan:
                fn = self.get_fn(channel, logmlim, zlim,
                    fmt=fmt, final=False, channel_name=chname)
                fn = intmd_dir + '/' + fn
                self.save_map(fn, cimg * f_norm / dnu,
                    channel, zlim, logmlim, fov,
                    pix=pix, fmt=fmt, hdr=hdr, map_units=map_units,
                    verbose=verbose, clobber=clobber)

                del cimg
                gc.collect()

                cimg = np.zeros([npix]*2)

        ##
        # Wipe slate clean
        f = open(f"{final_dir}/README", 'w')
        f.write("# map [central wavelength/micron] ; populations included \n")
        f.close()

        ##
        # Save final products

        ##
        # To finish: sum over populations for each channel.
        for h, channel in enumerate(channels):
            tot = np.zeros([npix]*2)

            for popid, pop in enumerate(self.sim.pops):
                if popid not in include_pops:
                    continue

                intmd_dir = base_dir + f'/intermediate_maps/pop_{popid}'

                ##
                # Save image summed over populations.
                fnp = self.get_fn(channel, logmlim=logmlim,
                    zlim=zlim, fmt=fmt, final=False,
                    channel_name=channel_names[h])
                fnp = intmd_dir + '/' + fnp

                _Inu = self._load_map(fnp)
                tot += _Inu

                del _Inu
                gc.collect()

            ##
            # Save final products
            fnp = self.get_fn(channel, logmlim=logmlim,
                zlim=zlim, fmt=fmt, final=True, channel_name=channel_names[h])
            fnp = final_dir + '/' + fnp

            # Intermediate maps will already have been converted to
            # map_units.
            self.save_map(fnp, tot, channel, zlim, logmlim, fov,
                pix=pix, fmt=fmt, hdr=hdr, map_units=map_units,
                clobber=clobber)

            # Make a note of which populations are included in the current
            # tally.

            s = "{:.4f} ; ".format(np.mean(channel))
            for pop in include_pops:
                s += f"{pop},"

            s = s.rstrip(',') + '\n'

            with open(f"{final_dir}/README", 'a') as f:
                f.write(s)

            if verbose:
                print(f"# Wrote {final_dir}/README")

    def save_cat(self, fn, cat, channel, zlim, logmlim, fov, pix=1, fmt='hdf5',
        hdr={}, clobber=False, verbose=False):
        """
        Save galaxy catalog.
        """
        ra, dec, red, X = cat

        if os.path.exists(fn) and (not clobber):
            print(f"# {fn} exists! Set clobber=True to overwrite.")
            return

        if fmt == 'hdf5':
            with h5py.File(fn, 'w') as f:
                f.create_dataset('ra', data=ra)
                f.create_dataset('dec', data=dec)
                f.create_dataset('z', data=red)
                f.create_dataset(channel, data=X)

                # Save hdr
                grp = f.create_group('hdr')
                for key in hdr:
                    grp.create_dataset(key, data=hdr[key])

        elif fmt == 'fits':
            col1 = fits.Column(name='ra', format='D', unit='deg', array=ra)
            col2 = fits.Column(name='dec', format='D', unit='deg', array=dec)
            col3 = fits.Column(name='z', format='D', unit='', array=red)

            col4 = fits.Column(name=channel, format='D', unit='', array=X)
            coldefs = fits.ColDefs([col1, col2, col3, col4])

            hdu = fits.BinTableHDU.from_columns(coldefs)

            hdu.writeto(fn, overwrite=clobber)
        else:
            raise NotImplemented(f'Unrecognized `fmt` option "{fmt}"')

        if verbose:
            print(f"# Wrote {fn}.")

    def save_map(self, fn, img, channel, zlim, logmlim, fov, pix=1, fmt='hdf5',
        hdr={}, map_units='MJy', clobber=False, verbose=True):
        """
        Save map to disk.
        """

        if os.path.exists(fn) and (not clobber):
            if verbose:
                print(f"# {fn} exists! Set clobber=True to overwrite.")
            return

        ra_e, ra_c, dec_e, dec_c = self.get_pixels(fov, pix=pix)

        nu = c * 1e4 / np.mean(channel)

        # Save as MJy/sr in this case.


        if fmt == 'hdf5':
            with h5py.File(fn, 'w') as f:
                f.create_dataset('ebl', data=img)
                f.create_dataset('ra_bin_e', data=ra_e)
                f.create_dataset('ra_bin_c', data=bin_e2c(ra_e))
                f.create_dataset('dec_bin_e', data=dec_e)
                f.create_dataset('dec_bin_c', data=bin_e2c(dec_e))
                f.create_dataset('wave_bin_e', data=channel)
                f.create_dataset('z_bin_e', data=zlim)
                f.create_dataset('m_bin_e', data=logmlim)
                f.create_dataset('nu_bin_c', data=nu)
        elif fmt == 'fits':
            from astropy.io import fits

            hdr = fits.Header(hdr)
            #_hdr.update(hdr)
            #hdr = _hdr
            hdr['DATE'] = time.ctime()

            hdr['NAXIS'] = 2
            if map_units.lower() == 'mjy':
                hdr['BUNIT'] = 'MJy/sr'
            elif map_units.lower() == 'cgs':
                hdr['BUNIT'] = 'erg/s/cm^2/sr'
            elif map_units.lower() == 'si':
                hdr['BUNIT'] = 'nW/m^2/sr'
            else:
                raise ValueError('help')

            hdr['CUNIT1'] = 'deg'
            hdr['CUNIT2'] = 'deg'
            hdr['CDELT1'] = pix / 3600.
            hdr['CDELT2'] = pix / 3600.
            hdr['NAXIS1'] = img.shape[0]
            hdr['NAXIS2'] = img.shape[1]

            hdr['PLATESC'] = pix
            hdr['WAVEMIN'] = channel[0]
            hdr['WAVEMAX'] = channel[1]
            hdr['CENTRWV'] = np.mean(channel)

            # Stuff specific to this modeling
            hdr['ZMIN'] = zlim[0]
            hdr['ZMAX'] = zlim[1]
            hdr['MHMIN'] = logmlim[0]
            hdr['MHMAX'] = logmlim[1]
            # This doesn't work anymore
            #hdr['ARES'] = get_hash().decode('utf-8')

            hdr.update(hdr)

            # Right now, image is in erg/s/cm^2/Hz
            # Convert to MJy/sr
            # Recall that 1 Jy = 1e-23 erg/s/cm^2/Hz

            hdu = fits.PrimaryHDU(data=img, header=hdr)
            hdul = fits.HDUList([hdu])
            hdul.writeto(fn, overwrite=clobber)
            hdul.close()

            del hdu, hdul
        else:
            raise NotImplementedError(f'No support for fmt={fmt}')

        print(f"# Wrote {fn}.")

        #del img
        #gc.collect()

    def _load_map(self, fn):

        fmt = fn[fn.rfind('.')+1:]

        ##
        # Read!
        if fmt == 'hdf5':
            with h5py.File(fn, 'r') as f:
                img = np.array(f[('ebl')])
        elif fmt == 'fits':
            from astropy.io import fits
            with fits.open(fn) as hdu:
                # In whatever `map_units` user supplied.
                img = hdu[0].data
        else:
            raise NotImplementedError(f'No support for fmt={fmt}!')

        return img

    def read_maps(self, fov, channels, pix=1, logmlim=None, dlogm=0.5,
        prefix=None, suffix=None, save_dir=None, keep_layers=True, fmt='hdf5'):
        """
        Assemble an array of maps.
        """

        if save_dir is None:
            save_dir = '.'

        npix = int(fov * 3600 / pix)
        zchunks = self.get_redshift_chunks(self.zlim)
        mchunks = self.get_mass_chunks(logmlim, dlogm)

        if keep_layers:
            layers = np.zeros((len(channels), len(zchunks), len(mchunks), npix, npix))
        else:
            layers = np.zeros((len(channels), npix, npix))

        ra_e, ra_c, dec_e, dec_c = self.get_pixels(fov, pix=pix)

        Nloaded = 0
        for i, channel in enumerate(channels):

            for j, (zlo, zhi) in enumerate(zchunks):

                for k, (mlo, mhi) in enumerate(mchunks):

                    fn = self.get_fn(fov, channel, pix=pix,
                        zlim=(zlo, zhi), prefix=prefix, suffix=suffix,
                        logmlim=(mlo, mhi), fmt=fmt)

                    fn = save_dir + '/' + fn

                    # Try to read from disk.
                    if not os.path.exists(fn):
                        continue

                    if keep_layers:
                        layers[i,j,k,:,:] = self._load_map(fn)
                    else:
                        layers[i,:,:] = self._load_map(fn)

                    print(f"# Loaded {fn}.")
                    Nloaded += 1

        if Nloaded == 0:
            raise IOError("Did not find any files! Are prefix, suffix, and save_dir set appropriately?")

        return channels, zchunks, mchunks, ra_c, dec_c, layers
