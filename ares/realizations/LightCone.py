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

try:
    import pymp
    have_pymp = True
except ImportError:
    have_pymp = False

angles_90 = 90 * np.arange(4)

class LightCone(object): # pragma: no cover
    """
    This should be inherited by the other classes in this submodule.
    """

    def build_directory_structure(self, fov, pix, logmlim=None, dryrun=False):
        """
        Setup file system!
        """

        # User-supplied prefix. Could just be `ares_mock`, or perhaps at some
        # point it signifies a major change to modeling code, etc.
        if dryrun:
            print(f"# Creating {self.base_dir}")
        elif not os.path.exists(f"{self.base_dir}"):
            os.mkdir(f"{self.base_dir}")

        # FOV
        if dryrun:
            print(f"# Creating {self.base_dir}/fov_{fov:.1f}")
        elif not os.path.exists(f"{self.base_dir}/fov_{fov:.1f}"):
            os.mkdir(f"{self.base_dir}/fov_{fov:.1f}")

        # pixel scale
        if dryrun:
            print(f"# Creating {self.base_dir}/fov_{fov:.1f}/pix_{pix:.1f}")
        elif not os.path.exists(f"{self.base_dir}/fov_{fov:.1f}/pix_{pix:.1f}"):
            os.mkdir(f"{self.base_dir}/fov_{fov:.1f}/pix_{pix:.1f}")

        sofar = f"{self.base_dir}/fov_{fov:.1f}/pix_{pix:.1f}"

        # Co-eval box size and grid zones
        if dryrun:
            print(f"# Creating {sofar}/box_{self.Lbox:.0f}")
        elif not os.path.exists(f"{sofar}/box_{self.Lbox:.0f}"):
            os.mkdir(f"{sofar}/box_{self.Lbox:.0f}")

        if dryrun:
            print(f"# Creating {sofar}/box_{self.Lbox:.0f}/dim_{self.dims:.0f}")
        elif not os.path.exists(f"{sofar}/box_{self.Lbox:.0f}/dim_{self.dims:.0f}"):
            os.mkdir(f"{sofar}/box_{self.Lbox:.0f}/dim_{self.dims:.0f}")

        sofar = f"{sofar}/box_{self.Lbox:.0f}/dim_{self.dims:.0f}"

        # Model name
        if dryrun:
            print(f"# Creating {sofar}/{self.model_name}")
        elif not os.path.exists(f"{sofar}/{self.model_name}"):
            os.mkdir(f"{sofar}/{self.model_name}")

        # Lower redshift bound
        if dryrun:
            print(f"# Creating {sofar}/{self.model_name}/zmin_{self.zmin:.3f}")
        elif not os.path.exists(f"{sofar}/{self.model_name}/zmin_{self.zmin:.3f}"):
            os.mkdir(f"{sofar}/{self.model_name}/zmin_{self.zmin:.3f}")

        sofar = f"{sofar}/{self.model_name}/zmin_{self.zmin:.3f}"


        # Directory for intermediate products?
        # Lightconing is deterministic, so given zmin and Lbox, we know
        # where the chunks will be.
        if dryrun:
            print(f"# Creating {sofar}/checkpoints")
        elif not os.path.exists(f"{sofar}/checkpoints"):
            os.mkdir(f"{sofar}/checkpoints")

        chck = f"{sofar}/checkpoints"

        # For each redshift chunk, make a new subdirectory in checkpoints
        # Add a README in checkpoints as well that indicates chunk properties.
        chunks = self.get_redshift_chunks(self.zlim)
        fn_R = f"{chck}/README"

        if dryrun:
            print(f"# Creating {fn_R}")
            for i, (zlo, zhi) in enumerate(chunks):
                print(f"# Creating {chck}/z_{zlo:.3f}_{zhi:.3f}/")
        else:
            with open(fn_R, 'w') as f:
                f.write('# co-eval chunk number; z lower edge; z upper edge\n')
                for i, (zlo, zhi) in enumerate(chunks):
                    f.write(f'{str(i).zfill(3)}; {zlo:.5f}; {zhi:.5f}\n')
                    if not os.path.exists(f"{chck}/z_{zlo:.3f}_{zhi:.3f}/"):
                        os.mkdir(f"{chck}/z_{zlo:.3f}_{zhi:.3f}/")

            # Copy README about co-eval cubes to zmax directory? i.e.,
            # lowest non-checkpoints directory?

        # Upper redshift bound
        if dryrun:
            print(f"# Creating {sofar}/zmax_{self.zlim[1]:.3f}")
        elif not os.path.exists(f"{sofar}/zmax_{self.zlim[1]:.3f}"):
            os.mkdir(f"{sofar}/zmax_{self.zlim[1]:.3f}")

        sofar = f"{sofar}/zmax_{self.zlim[1]:.3f}"

        # Mass range
        if logmlim is not None:
            mlo, mhi = logmlim
            if dryrun:
                print(f"# Creating {sofar}/m_{mlo:.2f}_{mhi:.2f}")
            elif not os.path.exists(f"{sofar}/m_{mlo:.2f}_{mhi:.2f}/"):
                os.mkdir(f"{sofar}/m_{mlo:.2f}_{mhi:.2f}")

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
        angl_per_Llo = self.sim.cosm.get_angle_from_length_comoving(zlo, L) / 60.
        angl_per_Lhi = self.sim.cosm.get_angle_from_length_comoving(zhi, L) / 60.

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
            assert self._sim.pf['interpolate_cosmology_in_z']
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
            self._tab_z = np.arange(0.01, 20, 0.01)
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

        if self.zchunks is not None:
            dofz = [self.sim.cosm.get_dist_los_comoving(0, z) \
                for z in self.zchunks[:,0]]
            dofz.append(self.sim.cosm.get_dist_los_comoving(0, self.zchunks[-1,1]))
            Re = np.array(dofz) / cm_per_mpc
            return np.mean(self.zchunks, axis=1), self.zchunks, Re

        if Lbox is None:
            Lbox = self.Lbox

        if zlim is None:
            zlim = self.zlim

        ze, zmid, Re = self.sim.cosm.get_lightcone_boundaries(zlim, Lbox)

        return ze, zmid, Re

    def get_redshift_chunks(self, zlim):
        """
        Return the edges of each co-eval cube as positioned along the LoS.
        """

        if self.zchunks is not None:
            return self.zchunks

        ze, zmid, Re = self.get_domain_info(zlim)

        chunks = [(zlo, ze[i+1]) for i, zlo in enumerate(ze[0:-1])]
        return np.array(chunks)

    def get_mass_chunks(self, logmlim, dlogm):
        """
        Return segments in log10(halo mass / Msun) space to run maps.

        .. note :: This is mostly for computational purposes, i.e., by dividing
            up the work in halo mass bins, we can limit memory consumption.

        Parameters
        ----------
        logmlim: tuple
            Boundaries of halo mass space we want to run, e.g., logmlim=(10,13)
            will simulate the halo mass range 10^10 - 10^13 Msun.
        dlogm : int, float, np.ndarray
            The log10 mass bin used to divide up the work. For example, if
            dlogm=0.5, we will generate maps or catalogs in mass chunks 0.5 dex
            wide. You can also provide the bin edges explicitly if you'd like,
            which can be helpful if including very low mass halos (whose
            abundance grows rapidly). In this case, dlogm should be, e.g.,
            dlogm=np.ndarray([[10, 10.5], [10.5, 11], [11, 12], [12, 13]])

        """
        if type(dlogm) in numeric_types:
            mbins = np.arange(logmlim[0], logmlim[1], dlogm)
            return np.array([(mbin, mbin+dlogm) for mbin in mbins])
        else:
            return dlogm

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

    #def get_base_dir(self, fov, pix):
    #    """
    #    Generate the name for the root directory where all mocks for a given
    #    model will go.

    #    Our model is:

    #    -><base_dir>_fov_<FOV/deg>_pix_<pixel scale / arcsec>_L<box/cMpc/h>_N<dims>/
    #    ->  README

    #    Inside this directory, there will be many subdirectories: one for each
    #    spectral channel of interest.

    #    There will also be a series of .fits (or .hdf5) files, which represent
    #    "final" maps, i.e., those that are summed over redshift and mass chunks,
    #    and also summed over all source populations.

    #    """


    #    s = '{}/{}_fov_{:.1f}_pix_{:.1f}_L{:.0f}_N{:.0f}'.format(path,
    #        self.prefix, fov, pix, self.Lbox, self.dims)

    #    if suffix is None:
    #        print("# WARNING: might be worth providing `suffix` as additional identifier.")
    #    else:
    #        s += f'_{self.model_name}'

    #    return s

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


    def get_map(self, fov, pix, channel, logmlim, zlim, popid=0,
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
        If `buffer` is None, will return a map in our internal erg/s/cm^2/sr. If
        `buffer` is supplied, will increment that array, same units.
        Any conversion of units (using `map_units`) takes place *only* in the
        `generate_maps` routine.
        """

        pix_deg = pix / 3600.
        #sr_per_pix = pix_deg**2 / sqdeg_per_std

        assert fov * 3600 / pix % 1 == 0, \
            "FOV must be integer number of pixels wide!"

        # In degrees
        if type(fov) in numeric_types:
            fov = np.array([fov]*2)

        assert np.diff(fov) == 0, "Only square FOVs allowed right now."

        zall = self.get_redshift_chunks(zlim=self.zlim)

        ##
        # Make sure `zlim` is in provided redshift chunks.
        # This is mostly to prevent users from doing something they shouldn't.
        ichunk = np.argmin(np.abs(zlim[0] - zall[:,0]))

        assert np.allclose(zlim, zall[ichunk])

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

        seed_kw = self.get_seed_kwargs(ichunk, logmlim)

        ra, dec, red, Mh = self.get_catalog(zlim=(zlo, zhi),
            logmlim=logmlim, popid=popid, verbose=verbose)
        
        # Could be empty chunks for very massive halos and/or early times.
        if ra is None:
            return #None, None, None

        # Correct for field position. Always (0,0) for log-normal boxes,
        # may not be for halo catalogs from sims.
        ra -= self.fxy[0]
        dec -= self.fxy[1]

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
            okz = None
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
        #waves = np.arange(_wlo, _whi+dlam, dlam)

        x = np.array([np.mean(channel) * 1e4 / (1. + _z_)])
        band = (channel[0] * 1e4 / (1. + _z_), channel[1] * 1e4 / (1. + _z_))
        #dfreq = (c * 1e8 / min(band)) - (c * 1e4 / max(band))
        #dlam = band[1] - band[0]
        # Need to supply band or window?
        # Note: NOT using get_spec_obs because every object has a
        # slightly different redshift, want more precise fluxes.

        seds = self.sim.pops[popid].get_lum(_z_, x, Mh=Mh, units='Ang',
            units_out='erg/s/Ang', band=tuple(band))

        # `owaves` is still in Angstroms
        #owaves = waves[None,:] * (1. + red[:,None])

        # Frequency "squashing", i.e., our 'per Angstrom' interval is
        # different in the observer frame by a factor of 1+z.
        #flux = corr[:,None] * seds[:,:] / (1. + red[:,None])
        flux = seds * corr / (1. + red)

        ##
        # Need some extra info to do more sophisticated modeling...
        ##
        # Extended emission from IHL, satellites
        if (not self.sim.pops[popid].is_central_pop):

            Rmi, Rma = -3, 1
            dlogR = 0.25
            Rall = 10**np.arange(Rmi, Rma+dlogR, dlogR)

            if max_sources == 1:

                Sall = self.sim.pops[popid].halos.get_halo_surface_dens(
                    _z_, Mh[0], Rall
                )

                Sall = np.array([Sall])

                Mall = Mh
            else:
                _iz = np.argmin(np.abs(_z_ - self.sim.pops[popid].halos.tab_z))

                # Remaining dimensions (Mh, R)
                Sall = self.sim.pops[popid].halos.tab_Sigma_nfw[_iz,:,:]
                Mall = self.sim.pops[popid].halos.tab_M

            mpc_per_arcmin = self.sim.cosm.get_angle_from_length_comoving(_z_,
                pix / 60.)

            rr, dd = np.meshgrid(ra_c * 60 * mpc_per_arcmin,
                                dec_c * 60 * mpc_per_arcmin)


        elif include_galaxy_sizes:

            Ms = self.sim.pops[popid].get_smhm(z=red, Mh=Mh) * Mh
            Rkpc = self.pops[popid].get_size(z=red, Ms=Ms)

            R_sec = np.zeros_like(Rkpc)
            for kk in range(red.size):
                R_sec[kk] = self.sim.cosm.get_angle_from_length_proper(red[kk], Rkpc[kk] * 1e-3)
            R_sec *= 60.

            # Uniform for now.
            np.random.seed(seed_kw['seed_nsers'])
            nsers = np.random.random(size=Rkpc.size) * 5.9 + 0.3
            np.random.seed(seed_kw['seed_pa'])
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
                #cog_sfg = [self.sim.pops[popid].get_sersic_cog(r,
                #    n=nsers[h]) \
                #    for r in rarr]

                rmax = [self.sim.pops[popid].get_sersic_rmax(size_cut,
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
        # Actually sum fluxes from all objects in image plane.
        for h in range(ra.size):

            #if not ok[h]:
            #    continue

            # Where this galaxy lives in pixel coordinates
            i, j = ra_ind[h], de_ind[h]

            # Grab the flux
            _flux_ = flux[h]

            # HERE: account for fact that galaxies aren't point sources.
            # [optional]
            if not self.sim.pops[popid].is_central_pop:

                # Image of distances from halo center
                r0 = ra_c[i] * 60 * mpc_per_arcmin
                d0 = dec_c[j] * 60 * mpc_per_arcmin
                Rarr = np.sqrt((rr - r0)**2 + (dd - d0)**2)

                # In Msun/cMpc^3

                # Interpolate between tabulated solutions.
                iM = np.argmin(np.abs(Mh[h] - Mall))

                I = np.interp(np.log10(Rarr), np.log10(Rall), Sall[iM,:])

                tot = I.sum()

                if tot == 0:
                    img[i,j] += _flux_
                else:
                    img[:,:] += _flux_ * I / tot

            elif include_galaxy_sizes and R_X[h] >= 1:

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

            ##
            # Otherwise just add flux to single pixel
            else:
                img[i,j] += _flux_


        ##
        # Clear out some memory sheesh
        del seds, flux, _flux_, ra, dec, red, Mh, ok, okp, okz, ra_ind, de_ind, \
            mask_ra, mask_de, corr
        if self.mem_concious:
            gc.collect()

    def get_output_dir(self, fov, pix, zlim, logmlim=None):
        fn = f"{self.base_dir}/fov_{fov:.1f}/pix_{pix:.1f}"
        fn += f"/box_{self.Lbox:.0f}/dim_{self.dims:.0f}"
        fn += f"/{self.model_name}"
        fn += f"/zmin_{self.zmin:.3f}"

        # Need directory for zmax, logmlim range
        final = (zlim[0] == self.zlim[0]) and (zlim[1] == self.zlim[1])

        #
        if final:
            fn += f"/zmax_{zlim[1]:.3f}"
            if logmlim is not None:
                fn += f"/m_{logmlim[0]:.2f}_{logmlim[1]:.2f}"
        else:
            fn += f'/checkpoints/z_{zlim[0]:.3f}_{zlim[1]:.3f}'
            if logmlim is not None:
                fn += f'/m_{logmlim[0]:.2f}_{logmlim[1]:.2f}'

        # Everything should exist up to the m_??.??_??.?? subdirectory
        if not os.path.exists(fn):
            os.mkdir(fn)

        return fn

    def get_map_fn(self, fov, pix, channel, popid, logmlim=None, zlim=None,
        fmt='fits'):
        """

        """

        save_dir = self.get_output_dir(fov=fov, pix=pix,
            zlim=zlim, logmlim=logmlim)

        fn = '{}/map_{:.3f}_{:.3f}_pop_{:.0f}'.format(save_dir,
            channel[0], channel[1], popid)

        return fn + '.' + fmt

    def get_cat_fn(self, fov, pix, channel, popid, logmlim=None, zlim=None,
        fmt='fits'):
        """

        """

        save_dir = self.get_output_dir(fov=fov, pix=pix,
            zlim=zlim, logmlim=logmlim)

        fn = f'{save_dir}/cat_{channel}_pop_{popid:.0f}'

        return fn + '.' + fmt

    def get_README(self, fov, pix, zlim=None, logmlim=None,
        is_map=True, verbose=False):
        """

        """

        assert is_map

        base_dir = self.get_output_dir(fov, pix, zlim=zlim, logmlim=logmlim)

        hdr = "#" * 78
        hdr += '\n# README\n'
        hdr += "#" * 78
        hdr +=  "\n# This is an automatically-generated file! \n"
        hdr += "# It contains some basic metadata for maps once they are available.\n"
        hdr += "# Note: all wavelengths here are in microns.\n"
        hdr += "#" * 78
        hdr += "\n"
        hdr += "# channel name [optional]; central wavelength; "
        hdr += "channel lower edge; channel upper edge; "
        hdr += "population ID; filename \n"

        ##
        # Write
        if not os.path.exists(f"{base_dir}/README"):
            with open(f'{base_dir}/README', 'w') as f:
                f.write(hdr)

            if verbose:
                print(f"# Wrote to {base_dir}/README")

        return hdr

    def generate_lightcone(self, fov, pix, channels):
        """
        Generate a lightcone.
        """
        pass

    def generate_cats(self, fov, pix, channels, logmlim, dlogm=0.5, zlim=None,
        include_galaxy_sizes=False, dlam=20, path='.', channel_names=None,
        suffix=None, fmt='fits', hdr={}, max_sources=None, cat_units='uJy',
        include_pops=None, clobber=False, verbose=False, dryrun=False,
        use_pbar=True, **kwargs):
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
        self.build_directory_structure(fov, pix, dryrun=False)

        # Create root directory if it doesn't already exist.
        base_dir = self.get_output_dir(fov, pix,
            zlim=self.zlim, logmlim=logmlim)

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
        #README = self.get_README(fov=fov, pix=pix, channels=channels,
        #    zlim=zlim, logmlim=logmlim, path=path, fmt=fmt,
        #    channel_names=channel_names,
        #    suffix=suffix, save=True, is_map=False, verbose=verbose)

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

        all_chunks = self.get_layers(channels, logmlim, dlogm=dlogm,
            include_pops=include_pops, channel_names=channel_names)

        # Progress bar
        pb = ProgressBar(len(all_chunks),
            name="cat(Mh>={:.1f}, Mh<{:.1f}, z>={:.3f}, z<{:.3f})".format(
                logmlim[0], logmlim[1], zlim[0], zlim[1]),
            use=use_pbar)
        pb.start()

        ##
        # Start doing work.
        ct = 0

        ra = []
        dec = []
        red = []
        dat = []
        for h, chunk in enumerate(all_chunks):

            # Unpack info about this chunk
            popid, channel, chname, zchunk, mchunk = chunk

            # Get number of z chunk
            iz = np.digitize(zchunk.mean(), bins=zchunks[:,0]) - 1

            # See if we already finished this map.
            fn = self.get_cat_fn(fov, pix, channel, popid,
                logmlim=mchunk, zlim=zchunk)

            pb.update(h)

            if dryrun:
                print(f"# Dry run: would run catalog {fn}")
                continue

            # Try to read from disk.
            if os.path.exists(fn) and (not clobber):
                if verbose:
                    print(f"Found {fn}. Set clobber=True to overwrite.")
                _ra, _dec, _red, _X, Xunit = self._load_cat(fn)
                ra.extend(list(_ra))
                dec.extend(list(_dec))
                red.extend(list(_red))
                dat.extend(list(_X))
            else:

                # Get basic halo properties
                _ra, _dec, _red, _Mh = self.get_catalog(zlim=zchunk,
                    logmlim=mchunk, popid=popid, verbose=verbose)

                # Could be empty chunks for very massive halos and/or early times.
                if _ra is None:
                    # You might think: let's `continue` to the next iteration!
                    # BUT, if we do that, and we're really unlucky and this
                    # happens on the last chunk of work for a given channel,
                    # then no checkpoint will be written below :/
                    # Hence the use of `pass` here intead.
                    pass
                else:
                    # Hack out galaxies outside our requested lightcone.
                    ok = np.logical_and(np.abs(_ra)  < fov / 2.,
                                        np.abs(_dec) < fov / 2.)

                    # Limit number of sources, just for testing.
                    if (max_sources is not None):
                        if (ct == 0) and (max_sources >= _Mh.size):
                            # In this case, we can accommodate all the galaxies in
                            # the catalog, so don't do anything yet.
                            pass
                        else:
                            # Flag entries until we hit target.
                            # This is not efficient but oh well.
                            for _h in range(_Mh.size):
                                ok[_h] = 0

                                if ok.sum() == max_sources:
                                    break

                            # This will be the final iteration.
                            if ct + ok.sum() == max_sources:
                                self._hit_max_sources = True


                    # Isolate OK entries.
                    _ra = _ra[ok==1]
                    _dec = _dec[ok==1]
                    _red = _red[ok==1]
                    _Mh = _Mh[ok==1]

                    ct += ok.sum()

                    ra.extend(list(_ra))
                    dec.extend(list(_dec))
                    red.extend(list(_red))

                    ##
                    # Unpack channel info
                    # Could be name of field, e.g., 'Mh', 'SFR', 'Mstell',
                    # photometric info, e.g., ('roman', 'F087'),
                    # or special quantities like Ly-a EW or luminosity.
                    # Note: if pops[popid] is a GalaxyEnsemble object
                    if channel in ['Mh', 'Ms', 'SFR']:
                        _dat = _Mh
                    elif channel.lower().startswith('ew'):
                        raise NotImplemented('help')
                    else:
                        cam, filt = channel.split('_')

                        _filt, mags = self.sim.pops[popid].get_mags(zcent[iz],
                            absolute=False, cam=cam, filters=[filt],
                            Mh=_Mh)

                        if cat_units == 'mags':
                            _dat = np.atleast_1d(mags.squeeze())
                        elif 'jy' in cat_units.lower():
                            flux = 3631. * 10**(mags / -2.5)

                            if cat_units.lower() == 'jy':
                                _dat = np.atleast_1d(flux.squeeze())
                            elif cat_units.lower() in ['microjy', 'ujy']:
                                _dat = np.atleast_1d(1e6 * flux.squeeze())
                            else:
                                raise NotImplemented('help')
                        else:
                            raise NotImplemented('Unrecognized `cat_units`.')

                    ##
                    # Save
                    self.save_cat(fn, (_ra, _dec, _red, _dat),
                        channel, zchunk, mchunk,
                        fov, pix=pix, fmt=fmt, hdr=hdr,
                        cat_units=cat_units,
                        clobber=clobber, verbose=verbose)


                    dat.extend(list(_dat))

                # End of else block that generates new catalog if one isn't found.

            # Back to level of loop over chunks of work.

            ##
            # Figure out if we're done with all the chunks
            if h == len(all_chunks) - 1:
                done_w_chan = True
            else:
                done_w_chan = channel != all_chunks[h+1][1]

            # If we're done with this channel, save file containing
            # full redshift and mass range.
            if done_w_chan:
                _fn = self.get_cat_fn(fov, pix, channel, popid,
                    logmlim=logmlim, zlim=self.zlim, fmt=fmt)

                self.save_cat(_fn, (ra, dec, red, dat),
                    channel, self.zlim, logmlim,
                    fov, pix=pix, fmt=fmt, hdr=hdr, cat_units=cat_units,
                    clobber=clobber, verbose=verbose)

                del ra, dec, red, dat
                dat = []
                ra = []
                dec = []
                red = []

        pb.finish()

        ##
        # Done
        return

    def get_layers(self, channels, logmlim, dlogm=0.5, include_pops=None,
        channel_names=None):
        """
        Take a list of channels, populations, and bounds in halo mass,
        and construct a list of chunks of work to do of the form:

        all_chunks = [
            (popid, channel, chname, zchunk, mchunk),
            (popid, channel, chname, zchunk, mchunk),
            (popid, channel, chname, zchunk, mchunk),
            (popid, channel, chname, zchunk, mchunk),
          ...
        ]

        Basically this allows us to 'flatten' a series of for loops over
        spectral channels, populations, redshift, and mass chunks into
        a single loop. Just unpack as, e.g.,

        >>> all_chunks = self.get_layers(channels, logmlim, dlogm=dlogm,
        >>>    include_pops=include_pops)
        >>> for chunk in all_chunks:
        >>>    popid, channel, chname, zchunk, mchunk = chunk
        >>>    <do cool stuff>

        """

        if include_pops is None:
            include_pops = range(0, len(self.sim.pops))

        zchunks = self.get_redshift_chunks(self.zlim)
        mchunks = self.get_mass_chunks(logmlim, dlogm)
        pchunks = include_pops

        if channel_names is None:
            channel_names = [None] * len(channels)


        all_chunks = []
        for h, popid in enumerate(pchunks):
            for i, channel in enumerate(channels):
                for j, zchunk in enumerate(zchunks):

                    # Option to limit redshift range.
                    zlo, zhi = zchunk

                    for k, mchunk in enumerate(mchunks):
                        all_chunks.append((popid, channel, channel_names[i],
                            zchunk, mchunk))

        return all_chunks

    def _check_for_corrupted_files(self, fov, pix, channels, logmlim, dlogm,
        include_pops, channel_names=None):
        """
        When running on a cluster, occasionally we get really unlucky and an
        output file will be corrupted, (probably) because we hit the wallclock
        time limit on the job while the file is being written. This routine
        does a cursory check that pre-existing files all have the same size, as
        a quick-and-dirty way of rooting out corrupted files.
        """


        # Assemble list of map layers to run.
        all_chunks = self.get_layers(channels, logmlim, dlogm=dlogm,
            include_pops=include_pops, channel_names=channel_names)

        all_zchunks = np.array(self.get_redshift_chunks(self.zlim))
        all_mchunks = np.array(self.get_mass_chunks(logmlim, dlogm))

        # Check status before we start
        all_sizes = np.zeros(len(all_chunks))
        all_fn = []

        for h, chunk in enumerate(all_chunks):

            # Unpack info about this chunk
            popid, channel, chname, zchunk, mchunk = chunk

            # See if we already finished this map.
            fn = self.get_map_fn(fov, pix, channel, popid,
                logmlim=mchunk, zlim=zchunk)

            all_fn.append(fn)

            if not os.path.exists(fn):
                continue

            all_sizes[h] = os.path.getsize(fn)


        # Find
        usizes = np.unique(all_sizes)

        if len(usizes) > 2:
            print(f"! WARNING: evidence for corrupted file(s)!")
            should_be = usizes.max()

            probs = []
            for h, fn in enumerate(all_fn):
                if all_sizes[h] in [0, should_be]:
                    continue

                probs.append(fn)

                print(f"! Problem file for chunk={h}: {fn}.")

            ##
            # Consistent with failed write as job is killed
            if len(probs) == 1:
                #os.remove(probs[0])
                print(f"! Removed corrupted file {fn}.")
            else:
                raise IOError('! {len(probs)} corrupted files detected. Help?')

        else:
            ##
            # Made it here? All good
            print(f"! No corrupted files detected! All {len(all_chunks)} chunks look good.")


    def generate_maps(self, fov, pix, channels, logmlim, dlogm=0.5,
        include_galaxy_sizes=False, size_cut=0.9, dlam=20,
        suffix=None, fmt='fits', hdr={}, map_units='MJy/sr', channel_names=None,
        include_pops=None, clobber=False, max_sources=None,
        keep_layers=True, use_pbar=False, verbose=False, dryrun=False, **kwargs):
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

        Parameters
        ----------
        fov : int, float
            Field of view, linear dimension, in degrees.
        pix : int, float
            Pixel scale, i.e., size of each pixel (linear dimension) [arcsec]
        channels : list
            List of channel edges, e.g., [(1, 1.05), (1.05, 1.1)] [microns].
        logmlim : tuple
            Halo mass range to include in model (log10(Mhalo/Msun)), e.g.,
            (12, 13).
        dlogm : float
            To limit memory consumption, only generate halos in a log10(mass)
            bin this wide at a time.
        zlim : tuple
            Boundaries of lightcone used to create map in redshift.
        dlam : int, float
            Generate galaxy SEDs at intrinsic resolution of `dlam` (Angstroms)
            before ultimately binning into `channels`.
        include_pops : tuple, list
            Integers corresponding to population ID numbers to be included in
            calculation, e.g., [0] would just include the first population,
            typically star-forming galaxies, while [0, 1] would include the
            first two (ID number 1 is usually quiescent centrals).

        Returns
        -------
        Right now, nothing. Just saves files to disk.

        """

        pix_deg = pix / 3600.

        # Create root directory if it doesn't already exist.
        self.build_directory_structure(fov, pix, dryrun=False)

        # Must do this after building the directory tree otherwise
        # we'll get errors.
        self._check_for_corrupted_files(fov, pix, channels,
            logmlim=logmlim, dlogm=dlogm,
            include_pops=include_pops, channel_names=channel_names)

        ##
        # Initialize a README file / see what's in it.
        README = self.get_README(fov=fov, pix=pix, zlim=self.zlim,
            logmlim=logmlim)

        # For final outputs
        final_dir = self.get_output_dir(fov=fov, pix=pix, zlim=self.zlim,
            logmlim=logmlim)

        # Only reason this may not exist yet is because build_directory_structure
        # doesn't know about the mass range of interest.
        if not os.path.exists(final_dir):
            os.mkdir(final_dir)

        if np.array(channels).ndim == 1:
            channels = np.array([channels])
        elif type(channels) in [list, tuple]:
            channels = np.array(channels)

        if channel_names is None:
            channel_names = [None] * len(channels)

        #if zlim is None:
        zlim = self.zlim

        if include_pops is None:
            include_pops = range(0, len(self.sim.pops))

        assert fov * 3600 / pix % 1 == 0, \
            "FOV must be integer number of pixels wide!"

        npix = int(fov * 3600 / pix)

        ##
        # Remember: using cgs units internally. Compute conversion factor to
        # users favorite units (within reason).
        # 1 Jy = 1e-23 erg/s/cm^2/sr
        if (map_units.lower() == 'si') or ('nw/m^2' in map_units.lower()):
            # aka (1e2)^2 / 0.01 = 1e6
            f_norm = cm_per_m**2 / erg_per_s_per_nW
        elif map_units.lower() == 'cgs':
            f_norm = 1.
        elif 'mjy' in map_units.lower():
            # 1 MJy = 1e6 Jy = 1e6 * 1e-23 erg/s/cm^2/sr = 1e17 MJy / cgs units
            f_norm = 1e17
        else:
            raise ValueErorr(f"Unrecognized option `map_units={map_units}`")

        if '/sr' in map_units.lower():
            sr_per_pix = pix_deg**2 / sqdeg_per_std
            f_norm /= sr_per_pix

        # Assemble list of map layers to run.
        all_chunks = self.get_layers(channels, logmlim, dlogm=dlogm,
            include_pops=include_pops, channel_names=channel_names)

        all_zchunks = np.array(self.get_redshift_chunks(self.zlim))
        all_mchunks = np.array(self.get_mass_chunks(logmlim, dlogm))

        # Array telling us which chunks were already done and which
        # we ran from scratch so at the end we know whether to update
        # the channel maps.
        # Recall that if we changed zmax, final maps will go in a new
        # subdirectory.
        status_done_pre = np.zeros((len(include_pops), len(channels),
            len(all_zchunks), len(all_mchunks)))
        status_done_now = status_done_pre.copy()

        # Check status before we start
        for h, chunk in enumerate(all_chunks):

            # Unpack info about this chunk
            popid, channel, chname, zchunk, mchunk = chunk

            # Identify indices of each (channel, z, m) chunk
            ichan = np.argmin(np.abs(channel[0] - channels[:,0]))
            iz = np.argmin(np.abs(zchunk[0] - all_zchunks[:,0]))
            im = np.argmin(np.abs(mchunk[0] - all_mchunks[:,0]))

            # See if we already finished this map.
            fn = self.get_map_fn(fov, pix, channel, popid,
                logmlim=mchunk, zlim=zchunk)

            if os.path.exists(fn) and (not clobber):
                status_done_pre[popid,ichan,iz,im] = 1

        # Progress bar
        pb = ProgressBar(len(all_chunks),
            name="img(Mh>={:.1f}, Mh<{:.1f}, z>={:.3f}, z<{:.3f})".format(
                logmlim[0], logmlim[1], zlim[0], zlim[1]),
            use=use_pbar)
        pb.start()

        # Make preliminary buffer for channel map (hence 'c' + 'img')
        cimg = np.zeros([npix]*2)

        if verbose:
            print(f"# Generating {len(all_chunks)} individual map layers...")

        ##
        # Start doing work.
        # The way this works is we treat each chunk: (z, M, pop, lambda)
        # separately. We'll keep a running tally of the "final" flux in any
        # given channel map as we go, and only create a new buffer when we
        # finish all the work for a single channel and a given population.
        for h, chunk in enumerate(all_chunks):

            # Unpack info about this chunk
            popid, channel, chname, zchunk, mchunk = chunk

            # Identify indices of each (channel, z, m) chunk
            ichan = np.argmin(np.abs(channel[0] - channels[:,0]))
            iz = np.argmin(np.abs(zchunk[0] - all_zchunks[:,0]))
            im = np.argmin(np.abs(mchunk[0] - all_mchunks[:,0]))

            # Can only move on if ALL chunks are already done, otherwise
            # it means the user has added z or m chunks since the last run,
            # and so the final channel map (saved into new subdirectory
            # to reflect new zmax, logmlim range) must be incremented.
            #if np.all(status_done_pre[popid,ichan,:,:]):
            #    continue

            # See if we already finished this map.
            fn = self.get_map_fn(fov, pix, channel, popid,
                logmlim=mchunk, zlim=zchunk)

            pb.update(h)

            if dryrun:
                print(f"# Dry run: would run map {fn}")
                continue

            # Will need channel width in Hz to recover specific intensities
            # averaged over band.
            nu = c * 1e4 / np.mean(channel)
            dnu = c * 1e4 * (channel[1] - channel[0]) / np.mean(channel)**2

            # What buffer should we increment?
            if (not keep_layers):
                buffer = cimg
            else:
                buffer = np.zeros([npix]*2)

            ran_new = True
            if os.path.exists(fn) and (not clobber):
                # Load map
                _buffer, _hdr = self._load_map(fn)

                if _hdr['BUNIT'] == map_units:
                    _buffer *= (f_norm / dnu)**-1.
                else:
                    raise NotImplemented('help')

                # Might need to adjust units before incrementing
                #buffer += _buffer
                # Increment map for this z chunk
                cimg += _buffer

                if verbose:
                    print(f"# Loaded map {fn}.")

                ran_new = False
            else:
                if verbose:
                    print(f"# Generating map {fn}...")

                # Generate map -> buffer
                # Internal flux units are cgs [erg/s/cm^2/Hz/sr]
                # but get_map returns a channel-integrated flux, erg/s/cm^2/sr
                self.get_map(fov, pix, channel,
                    logmlim=mchunk, zlim=zchunk, popid=popid,
                    include_galaxy_sizes=include_galaxy_sizes,
                    size_cut=size_cut,
                    dlam=dlam, use_pbar=False,
                    max_sources=max_sources,
                    buffer=buffer, verbose=verbose,
                    **kwargs)

                status_done_now[popid,ichan,iz,im] = 1

            # Save every mass chunk within every redshift chunk if the user
            # says so.
            if keep_layers and ran_new:
                _fn = self.get_map_fn(fov, pix, channel, popid,
                    logmlim=mchunk, zlim=zchunk,
                    fmt=fmt)
                self.save_map(_fn, buffer * f_norm / dnu,
                    channel, zchunk, logmlim, fov,
                    pix=pix, fmt=fmt, hdr=hdr, map_units=map_units,
                    verbose=verbose, clobber=clobber)

                # Increment map for this z chunk
                cimg += buffer

            ##
            # Otherwise, figure out what (if anything) needs to be
            # written to disk now.
            done_w_chan = np.all(
                status_done_pre[popid,ichan,:,:] +
                status_done_now[popid,ichan,:,:]
                )

            # This probably means our re-run only added channels, not
            # z chunks or mass chunks.
            was_done_already = np.all(status_done_pre[popid,ichan,:,:] == 1) \
                and (not clobber)

            ##
            # Need to know:
            # Did we do any work to fill out this spectral channel, e.g.,
            # augmenting the redshift or mass range? If so, we need to save
            # a new channel map. If not, we don't need to write anything to
            # disk, but we do need to clear 'cimg' since the next iteration
            # will be a new channel.

            # Also: for mass chunks, we might run, e.g., (11,12) in one call,
            # (12,13) next, and then later decide to do (11,13), in which case
            # all the work is done already *except* creating the final
            # channel map. That's why below we'll either write the final map
            # if we can tell the work wasn't done before OR if we can't find
            # an output file.

            # Filename for the final channel map
            # (note use of self.zlim, not zchunk, and logmlim, not mchunk)
            _fn = self.get_map_fn(fov, pix, channel, popid,
                logmlim=logmlim, zlim=self.zlim, fmt=fmt)

            _fn_exists = os.path.exists(_fn)

            # If we're done with the channel and population, time to write
            # a final "channel map". Afterward, we'll zero-out `cimg` to be
            # incremented starting on the next iteration.
            if done_w_chan and ((not was_done_already) or (not _fn_exists)):

                self.save_map(_fn, cimg * f_norm / dnu,
                    channel, self.zlim, logmlim, fov,
                    pix=pix, fmt=fmt, hdr=hdr, map_units=map_units,
                    verbose=verbose, clobber=clobber)

                del cimg, buffer
                if self.mem_concious:
                    gc.collect()

                base_dir = self.get_output_dir(fov, pix,
                    zlim=self.zlim, logmlim=logmlim)

                write_README = True

                # Check to see if we need to update the README.
                if os.path.exists(f'{base_dir}/README'):
                    _fn_ = np.loadtxt(f'{base_dir}/README', unpack=True,
                        dtype=str, usecols=[5])
                    _fn_ = np.atleast_1d(_fn_)
                    if _fn_.size > 0:
                        fnavail = [element.strip() for element in _fn_]

                        if _fn in fnavail:
                            write_README = False

                # channel name [optional]; central wavelength (microns); channel lower edge (microns) ; channel upper edge (microns) ; filename
                s_ch  = f'{chname}; {np.mean(channel):.5f}; '
                s_ch += f'{channel[0]:.5f}; {channel[1]:.5f}; '
                s_ch += f'{popid}; {_fn} \n'

                ##
                # # Append to README to indicate channel map is complete
                if write_README:
                    with open(f'{base_dir}/README', 'a') as f:
                        f.write(s_ch)

            ##
            # Need to zero-out channel map if done with channel, regardless
            # of how much work was already done before.
            if done_w_chan:
                # Setup blank buffer for next iteration
                cimg = np.zeros([npix]*2)

            ##
            # Next task

        # All done.
        pb.finish()

        return

    def save_cat(self, fn, cat, channel, zlim, logmlim, fov, pix=1, fmt='fits',
        hdr={}, clobber=False, verbose=False, cat_units=''):
        """
        Save galaxy catalog.
        """
        ra, dec, red, X = cat

        if os.path.exists(fn) and (not clobber):
            if verbose:
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

            col4 = fits.Column(name=channel, format='D', unit=cat_units, array=X)
            coldefs = fits.ColDefs([col1, col2, col3, col4])

            hdu = fits.BinTableHDU.from_columns(coldefs)

            if os.path.exists(fn) and (not clobber):
                print(f"# {fn} exists and clobber=False. Moving on.")
            else:
                hdu.writeto(fn, overwrite=clobber)
        else:
            raise NotImplemented(f'Unrecognized `fmt` option "{fmt}"')

        if verbose:
            print(f"# Wrote {fn}.")

    def save_map(self, fn, img, channel, zlim, logmlim, fov, pix=1, fmt='fits',
        hdr={}, map_units='MJy/sr', clobber=False, verbose=True):
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

            if verbose:
                print(f"# Wrote {fn}.")

        elif fmt == 'fits':
            hdr = fits.Header(hdr)
            #_hdr.update(hdr)
            #hdr = _hdr
            hdr['DATE'] = time.ctime()

            hdr['NAXIS'] = 2
            if 'mjy' in map_units.lower():
                hdr['BUNIT'] = map_units
            elif map_units.lower() == 'cgs':
                hdr['BUNIT'] = 'erg/s/cm^2/sr'
            elif 'erg/s/cm^2' in map_units.lower():
                hdr['BUNIT'] = map_units
            elif 'nW/m^2' in map_units.lower():
                hdr['BUNIT'] = map_units
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

            if os.path.exists(fn) and (not clobber):
                print(f"# {fn} exists and clobber=False. Moving on.")
            else:
                hdu = fits.PrimaryHDU(data=img, header=hdr)
                hdul = fits.HDUList([hdu])
                hdul.writeto(fn, overwrite=clobber)
                hdul.close()

                if verbose:
                    print(f"# Wrote {fn}.")

                del hdu, hdul
        else:
            raise NotImplementedError(f'No support for fmt={fmt}')

    def _load_map(self, fn):

        fmt = fn[fn.rfind('.')+1:]

        ##
        # Read!
        if fmt == 'hdf5':
            with h5py.File(fn, 'r') as f:
                img = np.array(f[('ebl')])
        elif fmt == 'fits':
            if self.verbose:
                print(f"! Attempting to load {fn}...")

            with fits.open(fn) as hdu:
                # In whatever `map_units` user supplied.
                img = hdu[0].data
                hdr = hdu[0].header

            print(f"! Loaded {fn}.")
        else:
            raise NotImplementedError(f'No support for fmt={fmt}!')

        return img, hdr

    def _load_cat(self, fn):
        if fn.endswith('hdf5'):
            with h5py.File(fn, 'r') as f:
                ra = np.array(f[('ra')])
                dec = np.array(f[('dec')])
                red = np.array(f[('z')])
                X = np.array(f[('Mh')])
                Xunit = None
        elif fn.endswith('fits'):
            with fits.open(fn) as f:
                data = f[1].data
                ra = data['ra']
                dec = data['dec']
                red = data['z']

                # Hack for now.
                name = data.columns[3].name
                X = data[name]
                Xunit = f[1].header['TUNIT4']
        else:
            raise NotImplemented('Unrecognized file format `{}`'.format(
                fn[fn.rfind('.'):]))

        return ra, dec, red, X, Xunit

    def read_maps(self, fov, channels, pix=1, logmlim=None, dlogm=0.5,
        prefix=None, suffix=None, save_dir=None, keep_layers=True, fmt='fits'):
        """
        Assemble an array of maps.
        """

        raise NotImplemented('needs fixing')

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
                        layers[i,j,k,:,:], _hdr = self._load_map(fn)
                    else:
                        layers[i,:,:] = self._load_map(fn)

                    print(f"# Loaded {fn}.")
                    Nloaded += 1

        if Nloaded == 0:
            raise IOError("Did not find any files! Are prefix, suffix, and save_dir set appropriately?")

        return channels, zchunks, mchunks, ra_c, dec_c, layers
