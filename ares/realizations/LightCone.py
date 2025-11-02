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
from pathlib import Path
from scipy.stats import truncnorm
from ..simulations import Simulation
from scipy.special import gammaincinv
from ..util.Stats import bin_e2c, bin_c2e
from ..util.ProgressBar import ProgressBar
from scipy.spatial.transform import Rotation
from ..util.Misc import numeric_types, get_hash, get_pop_info
from ..physics.Constants import sqdeg_per_std, cm_per_mpc, cm_per_m, \
    erg_per_s_per_nW, c, s_per_myr

try:
    from astropy.io import fits
except ImportError:
    pass

try:
    from astropy.modeling.models import Sersic2D
except ImportError:
    pass

#try:
#    from numba import njit, prange
#except ImportError:
#    pass

angles_90 = 90 * np.arange(4)

class LightCone(object): # pragma: no cover
    """
    This should be inherited by the other classes in this submodule.
    """

    def build_directory_structure(self, fov, logmlim=None, dryrun=False):
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
        #if dryrun:
        #    print(f"# Creating {self.base_dir}/fov_{fov:.1f}/pix_{pix:.1f}")
        #elif not os.path.exists(f"{self.base_dir}/fov_{fov:.1f}/pix_{pix:.1f}"):
        #    os.mkdir(f"{self.base_dir}/fov_{fov:.1f}/pix_{pix:.1f}")

        sofar = f"{self.base_dir}/fov_{fov:.1f}"#/pix_{pix:.1f}"

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
        # where the layers will be.
        if dryrun:
            print(f"# Creating {sofar}/checkpoints")
        elif not os.path.exists(f"{sofar}/checkpoints"):
            os.mkdir(f"{sofar}/checkpoints")

        chck = f"{sofar}/checkpoints"

        # For each redshift layer, make a new subdirectory in checkpoints
        # Add a README in checkpoints as well that indicates layer properties.
        layers = self.get_redshift_layers(self.zlim)
        fn_R = f"{chck}/README"

        if dryrun:
            print(f"# Creating {fn_R}")
            for i, (zlo, zhi) in enumerate(layers):
                print(f"# Creating {chck}/z_{zlo:.3f}_{zhi:.3f}/")
        else:
            with open(fn_R, 'w') as f:
                f.write('# co-eval layer number; z lower edge; z upper edge\n')
                for i, (zlo, zhi) in enumerate(layers):
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

        Parameters
        ----------
        zlim: tuple
            Redshift range of interest.

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
        the z-axis for each layer of our lightcone.
        """

        ze, zc, Re = self.get_domain_info()

        te = self.sim.cosm.t_of_z(ze) / s_per_myr

        return np.diff(te)

    def get_pixels(self, fov, pix=1, hdr=None):
        """
        For a given field of view and pixel scale get pixel centers and edges.

        .. note :: We assume the center of the image is at RA=DEC=0, so pixel
            coordinates span the domain [-1/2, -1/2] * FOV.

        Parameters
        ----------
        fov : int, float
            Field of view (assumed square) in degrees.
        pix : int, float
            Pixel scale in arcseconds.

        Returns
        -------
        Tuple containing the (RA pixel edges, RA pixel centers, DEC pixel
        edges, DEC pixel centers), all in degrees.
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

    @property
    def _cache_domain(self):
        if not hasattr(self, '_cache_domain_'):
            self._cache_domain_ = {}
        return self._cache_domain_

    def get_domain_info(self, zlim=None, Lbox=None):
        """
        Figure out how the domain will be divided up along the line of sight.

        Parameters
        ----------
        zlim : tuple
            Redshift range of interest.
        Lbox : int, float
            Co-eval box size in cMpc / h. If not provided, we'll use the
            value in `self.Lbox`.

        Returns
        -------
        A tuple containing (layer edges in redshift, layer midpoints in redshift,
            layer edges in comoving Mpc [NOT cMpc / h, despite input `Lbox`
            being in cMpc/h!]).

        """

        if (zlim, Lbox) in self._cache_domain.keys():
            return self._cache_domain[(zlim, Lbox)]

        if self.zlayers is not None:
            dofz = [self.sim.cosm.get_dist_los_comoving(0, z) \
                for z in self.zlayers[:,0]]
            dofz.append(self.sim.cosm.get_dist_los_comoving(0, self.zlayers[-1,1]))
            Re = np.array(dofz) / cm_per_mpc
            return np.mean(self.zlayers, axis=1), self.zlayers, Re

        if Lbox is None:
            Lbox = self.Lbox

        if zlim is None:
            zlim = self.zlim

        ze, zmid, Re = self.sim.cosm.get_lightcone_boundaries(zlim, Lbox)

        self._cache_domain[(zlim, Lbox)] = ze, zmid, Re

        return ze, zmid, Re

    @property
    def _cache_zlayers(self):
        if not hasattr(self, '_cache_zlayers_'):
            self._cache_zlayers_ = {}
        return self._cache_zlayers_

    def get_redshift_layers(self, zlim):
        """
        Return the edges of each co-eval cube as positioned along the LoS.

        .. note :: Similar to output of `get_domain_info`, except redshift bins
            are reported as 2-D array (series of bin edge pairs).

        """

        if self.zlayers is not None:
            return self.zlayers
        if zlim in self._cache_zlayers.keys():
            return self._cache_zlayers[zlim]

        ze, zmid, Re = self.get_domain_info(zlim)

        layers = [(zlo, ze[i+1]) for i, zlo in enumerate(ze[0:-1])]

        self._cache_zlayers[zlim] = np.array(layers)

        return np.array(self._cache_zlayers[zlim])

    def get_mass_layers(self, logmlim, dlogm):
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
            dlogm=0.5, we will generate maps or catalogs in mass layers 0.5 dex
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
        For a given redshift, return the index of the layer that contains it
        in the LoS direction.
        """
        zall = self.get_redshift_layers()
        zlo, zhi = np.array(zall).T
        iz = np.argmin(np.abs(z - zlo))
        if zlo[iz] > z:
            iz -= 1

        return iz

    def get_seed_kwargs(self, layer, logmlim, popid):
        """
        Deterministically adjust the random seeds for the given redshift layer,
        mass range, and population.

        Parameters
        ----------
        layer : int
            ID number for given co-eval redshift `layer`.
        logmlim : tuple
            Min/mass log10(halo mass / Msun) range of interest.
        popid : int
            Population ID number.

        Returns
        -------
        Dictionary of random seeds to use for halo masses, positions,
        occupation, orientation, Sersic index....

        """


        if not hasattr(self, '_seeds'):
            # ARES ID, ARES parent ID, str representation of popid (e.g., '2a')
            pid, pid_par, pid_str = get_pop_info(popid)

            fmh = int(logmlim[0] + (logmlim[1] - logmlim[0]) / 0.1)

            ze, zmid, Re = self.get_domain_info(zlim=self.zlim, Lbox=self.Lbox)

            seed_rho = self.seed_rho \
                * np.arange(1, len(zmid)+1)
            seed_mh  = self.seed_halo_mass \
                * np.arange(1, len(zmid)+1) * fmh
            seed_xyz = self.seed_halo_pos \
                * np.arange(1, len(zmid)+1) * fmh
            seed_focc = self.seed_halo_occ \
                * np.arange(1, len(zmid)+1) * fmh

            # These seeds uniquely determine the locations and masses
            # of star-forming and quiescent centrals.
            self._seeds = {'seed_box': seed_rho,
                'seed': seed_mh, 'seed_pos': seed_xyz,
                'seed_occ': seed_focc}

            ##
            # [optional] resolved galaxies
            # Need `popid` here to ensure we use different seeds for the
            # surface brightness profiles of quiescent galaxies.
            if self.seed_profile is not None:
                seed_prof = (self.seed_profile + popid) \
                    * np.arange(1, len(zmid)+1) * fmh
                self._seeds['seed_profile'] = seed_prof

            ##
            # [optional] seeds for satellites
            if self.seed_sats is not None:
                seed_sats = self.seed_sats \
                    * np.arange(1, len(zmid)+1) * fmh
                self._seeds['seed_sats'] = seed_sats

        i = layer
        # Done
        return {key:self._seeds[key][i] for key in self._seeds.keys()}

    def _get_flux_catalog(self, zlim, logmlim, red, Mh, channel, pid):
        """
        Compute flux from catalog of sources in given redshift range.

        Parameters
        ----------
        zlim : tuple
            Redshift range in which to sum fluxes. This is probably the
            boundaries of a co-eval chunk.
        red : np.ndarray
            Redshifts of galaxies in catalog.
        Mh : np.ndarray
            Halo masses [Msun] of galaxies in catalog.
        channel : tuple
            Spectral channel edges in microns.

        Returns
        -------
        An array of fluxes corresponding to the halos in `red` and `Mh`, the
        units are erg/s/cm^2/Angstrom.

        """
        zlo, zhi = zlim
        zsub_lo = 1 * zlo

        flux = np.zeros_like(Mh)
        while zsub_lo < zhi:

            zsub_hi = min(zsub_lo + self.dz_max, zhi)

            zsub_mid = np.mean([zsub_lo, zsub_hi])

            band = channel[0] * 1e4 / (1. + zsub_mid), \
                   channel[1] * 1e4 / (1. + zsub_mid)

            okzsub = np.logical_and(red >= zsub_lo, red < zsub_hi)

            _flux_ = self.sim.pops[pid].get_lum(zsub_mid, x=None,
                Mh=Mh[okzsub==1], units='Ang',
                units_out='erg/s/Ang', band=tuple(band))

            # Frequency "squashing", i.e., our 'per Angstrom' interval is
            # different in the observer frame by a factor of 1+z.
            corr = 1. / 4. / np.pi \
                / (np.interp(zsub_mid, self.tab_z, self.tab_dL) * cm_per_mpc)**2
            flux[okzsub==1] = _flux_ * corr / (1. + zsub_mid)

            zsub_lo += self.dz_max

        return flux

    def _get_size_catalog(self, zlim, logmlim, red, Mh, pid):
        """
        Return sizes and surface brightness profile info for a galaxy catalog.

        Parameters
        ----------
        zlim : tuple
            Redshift range in which to sum fluxes.
        red : np.ndarray
            Redshifts of galaxies in catalog.
        Mh : np.ndarray
            Halo masses [Msun] of galaxies in catalog.

        Returns
        -------
        A tuple containing the:
        - Half-light radii of galaxies (in arcseconds)
        - sersic indices
        - ellipcities
        - position angles

        """
        Ms = self.sim.pops[pid].get_smhm(z=red, Mh=Mh) * Mh
        Rkpc = self.pops[pid].get_size(z=red, Ms=Ms)

        # Much faster to interpolate from table than generate angle/pMpc
        # on the fly. Interpolant automatically used if provided R is 1
        arcsec_per_pmpc = np.array([60 * self.sim.cosm.get_angle_from_length_proper(
            zz, 1.
        ) for zz in red])
        
        R_sec = arcsec_per_pmpc * Rkpc * 1e-3

        zlo, zhi = zlim
        zall = self.get_redshift_layers(zlim=self.zlim)

        ##
        # Make sure `zlim` is in provided redshift layers.
        # This is mostly to prevent users from doing something they shouldn't.
        ilayer = np.argmin(np.abs(zlim[0] - zall[:,0]))

        seed_kw = self.get_seed_kwargs(ilayer, logmlim, pid)

        # `R_sec` is the angular size of each galaxy in the model in arcsec.
        # Note: the size is defined as the stellar half-light radius.

        # Uniform for now.
        np.random.seed(seed_kw['seed_profile'])

        # Sersic indices and position angles
        pop_s = 'sfg' if self.pops[pid].is_star_forming else 'qg'

        # First, identify redshift interval to use.
        zoptions = self.profile_info[f'{pop_s}_z']
        z1, z2 = np.array(zoptions).T

        # Make sure `iz` gets redshift within appropriate window
        iz = np.argmin(np.abs(zlo - z1))
        if zlo < z1[iz]:
            iz -= 1

        # If provided redshift is > max redshift in profile_info, just use
        # highest available redshift.
        if zlo > z2.max():
            iz = -1

        key = zoptions[iz]

        # Axis ratios first
        ba_loc, ba_scale = self.profile_info[f'{pop_s}_ba'][key]

        ba_trunc_lo = 0.1
        ba_trunc_hi = 1
        ba_t_lo = (ba_trunc_lo - ba_loc) / ba_scale
        ba_t_hi = (ba_trunc_hi - ba_loc) / ba_scale

        rv_ba = truncnorm(ba_t_lo, ba_t_hi, loc=ba_loc, scale=ba_scale)
        b_over_a = rv_ba.rvs(size=Rkpc.size)

        # Now Sersic indices
        n_loc, n_scale = self.profile_info[f'{pop_s}_n'][key]

        n_trunc_lo = 0.2
        n_trunc_hi = 7
        n_t_lo = (n_trunc_lo - n_loc) / n_scale
        n_t_hi = (n_trunc_hi - n_loc) / n_scale

        rv_n = truncnorm(n_t_lo, n_t_hi, loc=n_loc, scale=n_scale)
        nsers = rv_n.rvs(size=Rkpc.size)

        # Ellipticity = 1 - b/a
        ellip = 1 - b_over_a

        pa = np.random.random(size=Rkpc.size) * 360

        return R_sec, nsers, ellip, pa

    def _get_postage_stamp_pix(self, R, psize):
        """
        Determine the pixel indices for a postage stamp image.

        Parameters
        ----------
        R : int, float
            Size of object in pixels.
        psize : int, float
            Size of postage stamp in units of `R`, which is probably a
            half-light radius or virial radius.

        Returns
        -------
        Essentially the results of a meshgrid call, with a third quantity
        that indicates the radius of the postage stamp in number of pixels.
        """

        if not hasattr(self, '_cache_pstamp_pix_'):
            self._cache_pstamp_pix_ = {}

        # Determine how big of a postage stamp image to make in
        # number of pixels (just scale R_eff by `psize`)
        # (Force to be odd)
        _r_ = np.ceil(psize * R)
        if _r_ % 2 == 0:
            _r_ += 1

        # Load from cache if possible. numpy's `meshgrid` can be slow.
        if _r_ in self._cache_pstamp_pix_:
            return self._cache_pstamp_pix_[_r_]

        # Pixel coordinates
        xy = np.arange(-_r_, _r_ + 1, 1, dtype=int)
        xx, yy = np.meshgrid(xy, xy, indexing='ij')

        self._cache_pstamp_pix_[_r_] = xx, yy, _r_

        return xx, yy, _r_

    def _get_ihl_postage_stamp(self, _r_, Rarr, Rtab, Stab, iM):
        """
        Because the projected NFW profile is tabulated, we use this simple
        wrapper to first check if we've already interpolated to a postage
        stamp of size `_r_`
        """
        if not hasattr(self, '_cache_ihl_pstamp_'):
            self._cache_ihl_pstamp_ = {}

        if (_r_, iM) in self._cache_ihl_pstamp_:
            return self._cache_ihl_pstamp_[(_r_, iM)]

        I = np.interp(np.log10(Rarr), np.log10(Rtab), Stab[iM,:])

        self._cache_ihl_pstamp_[(_r_, iM)] = I

        return I

    def _get_postage_stamp_slices(self, pstamp, buffer, i, j):
        nx, ny = pstamp.shape

        # OK, now we need to figure out how to slot this postage
        # stamp into the entire image. Mostly just tedium like
        # worrying about sources near the edge of the frame.

        # `i` and `j` refer to pixels in the full frame image
        # Here, we're figuring out the chunk of the full frame
        # into which we'll drop our postage stamp
        slcx = slice(max(i-(nx-1)//2, 0), i+(nx-1)//2 + 1)
        slcy = slice(max(j-(ny-1)//2, 0), j+(ny-1)//2 + 1)
        # i.e., this is where we're sticking the postage stamp
        # If we're unlucky and near the edge, we need to also
        # slice the `pstamp`.

        # If source spills off x-axis, adjust postage stamp
        # accordingly (i.e., remove a few columns)
        if (slcx.start == 0):
            xlo = abs(i-(nx-1)//2)
        else:
            xlo = 0
        if (slcx.stop > buffer.shape[0]):
            xhi = -(slcx.stop - buffer.shape[0])
        else:
            xhi = None

        if (slcy.start == 0):
            ylo = abs(j-(ny-1)//2)
        else:
            ylo = 0

        if (slcy.stop > buffer.shape[1]):
            yhi = -(slcy.stop - buffer.shape[1])
        else:
            yhi = None

        slcx2 = slice(xlo, xhi)
        slcy2 = slice(ylo, yhi)

        return slcx, slcy, slcx2, slcy2

    def get_pix_mesh(self, fov, pix, in_mpc=0):
        """
        Get
        """
        if not hasattr(self, '_cache_pix_mesh_'):
            self._cache_pix_mesh_ = {}

        if not in_mpc:
            if (fov, pix, in_mpc) in self._cache_pix_mesh_.keys():
                return self._cache_pix_mesh_[(fov, pix, in_mpc)]

            ra_e, ra_c, dec_e, dec_c = self.get_pixels(fov, pix=pix)
            pix_deg = pix / 3600.


            rr, dd = np.meshgrid(ra_c / pix_deg, dec_c / pix_deg,
                indexing='ij')

            self._cache_pix_mesh_[(fov, pix, in_mpc)] = rr, dd
            return rr, dd

        ##
        # Slightly harder case

            mpc_per_arcmin = self.sim.cosm.get_angle_from_length_comoving(zmid,
                pix / 60.)

            rr, dd = np.meshgrid(ra_c * 60 * mpc_per_arcmin,
                                dec_c * 60 * mpc_per_arcmin,
                                indexing='ij')

    #@njit(parallel=True)
    def get_map(self, fov, pix, channel, logmlim, zlim, popid=0,
        include_galaxy_sizes=False, null_beyond_size=np.inf, size_cut=0.5, dlam=20.,
        use_pbar=True, verbose=False, wave_units='um',
        logmlim_sats=(11,15), buffer=None, nthreads=None, batch_size=10,
        postage_stamp=5, **kwargs):
        """
        Get a map for a single channel, redshift layer, mass layer, and
        source population.

        .. note :: To get a 'full' map, containing contributions from multiple
            redshift and mass layers, and potentially populations, see the
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
        postage_stamp : int, float
            If provided, and `include_galaxy_sizes==True`, this is the size of
            image (in units of R_eff) on which we'll create each galaxy's
            surface brightness profile, to then be slotted into the full image.

        Returns
        -------
        If `buffer` is None, will return a map in our internal erg/s/cm^2/sr. If
        `buffer` is supplied, will increment that array, same units.
        Any conversion of units (using `map_units`) takes place *only* in the
        `generate_maps` routine.
        """

        pix_deg = pix / 3600.

        assert fov * 3600 / pix % 1 == 0, \
            "FOV must be integer number of pixels wide!"

        # In degrees
        if type(fov) in numeric_types:
            fov_2d = np.array([fov]*2)
        else:
            fov_2d = fov

        assert np.diff(fov_2d) == 0, "Only square FOVs allowed right now."

        zall = self.get_redshift_layers(zlim=self.zlim)

        ##
        # Make sure `zlim` is in provided redshift layers.
        # This is mostly to prevent users from doing something they shouldn't.
        ilayer = np.argmin(np.abs(zlim[0] - zall[:,0]))

        assert np.allclose(zlim, zall[ilayer])

        # Figure out the edges of the domain in RA and DEC (degrees)
        # Pixel coordinates
        ra_e, ra_c, dec_e, dec_c = self.get_pixels(fov, pix=pix)

        Npix = [ra_c.size, dec_c.size]

        # Unpack popid more [as of March 2025]
        # (id number in ARES, parent ID number [if satellite], name as str)
        pid, pid_par, pid_str = get_pop_info(popid)

        zlo, zhi = zlim
        zmid = np.mean([zlo, zhi])

        seed_kw = self.get_seed_kwargs(ilayer, logmlim, pid)

        # Initialize empty map
        img = buffer

        ##
        # First, check for a pre-existing catalog in this channel.
        fn_cat_ch = self.get_cat_fn(fov, channel, popid,
            logmlim=logmlim, zlim=(zlo, zhi), wave_units=wave_units)
        
        if os.path.exists(fn_cat_ch):
            
            ra, dec, red, flux = self._load_cat(fn_cat_ch)

            # Figure out what pixel each source is in
            ra_bin = np.searchsorted(ra_e, ra, side='right')
            dec_bin = np.searchsorted(dec_e, dec, side='right')
            ra_ind = ra_bin - 1
            de_ind = dec_bin - 1

            # Internally, these fluxes are always in
            # erg/s/cm^2/Ang, but then integrated over channel.
            # Will need channel width in Hz to recover specific
            # intensities averaged over band.
            nu = c * 1e4 / np.mean(channel)
            dnu = c * 1e4 * (channel[1] - channel[0]) / np.mean(channel)**2

            _dat = self._get_flux_catalog(zlayer, logmlim, _red, _Mh,
                channel, pid)
            flux *= 1. / (self.get_map_norm(cat_units) / dnu)
        else:
            # Run fresh if we didn't find anything
            ra, dec, red, Mh, parents = self.get_catalog_halos(
                zlim=(zlo, zhi), logmlim=logmlim, popid=popid, verbose=verbose,
                satellites=self.sim.pops[pid].is_satellite_pop,
                logmlim_sats=logmlim_sats)

            # Could be empty layers for very massive halos and/or early times.
            if ra is None:
                return #None, None, None

            # Correct for field position. Always (0,0) for log-normal boxes,
            # may not be for halo catalogs from sims.
            ra -= self.fxy[0]
            dec -= self.fxy[1]

            ##
            # Figure out which bin each galaxy is in.
            # Slightly faster than np.digitize
            ra_bin = np.searchsorted(ra_e, ra, side='right')
            dec_bin = np.searchsorted(dec_e, dec, side='right')
            mask_ra = np.logical_or(ra_bin == 0, ra_bin == Npix[0]+1)
            mask_de = np.logical_or(dec_bin == 0, dec_bin == Npix[1]+1)
            ra_ind = ra_bin - 1
            de_ind = dec_bin - 1

            # Mask out galaxies that aren't in our desired image plane.
            okp = np.logical_not(np.logical_or(mask_ra, mask_de))

            # Filter out galaxies outside specified redshift range.
            # [usually don't do this within layer, but hey, functionality there]
            if zlim is not None:
                okz = np.logical_and(red >= zlo, red < zhi)
                ok = np.logical_and(okp, okz)
            else:
                okz = None
                ok = okp

            # May have empty layers, e.g., very massive halos and/or very
            # high redshifts.
            if not np.any(ok):
                return #None, None, None

            ##
            # Isolate OK entries.
            ra = ra[ok==1]
            dec = dec[ok==1]
            red = red[ok==1]
            Mh = Mh[ok==1]
            ra_ind = ra_ind[ok==1]
            de_ind = de_ind[ok==1]

            # Need to filter `parents` also
            if self.sim.pops[pid].is_satellite_pop:
                parents = parents[ok==1]
            else:
                # parents is None in this case
                pass

            # Shape of (ra, dec, red) is just (Ngalaxies)

            # Get flux from each object. Units = erg/s/cm^2/Ang.
            flux = self._get_flux_catalog((zlo, zhi), logmlim, red, Mh, channel, pid)

        ##
        # Need some extra info to do more sophisticated modeling...
        ##
        mpc_per_arcmin = self.sim.cosm.get_angle_from_length_comoving(zmid, 1)

        resolved_sources = False

        # Extended emission from IHL
        if self.sim.pops[pid].is_diffuse and include_galaxy_sizes:
            resolved_sources = True

            Rall = self.sim.pops[0].halos.tab_R_nfw
            Rvir = self.sim.pops[0].halos.get_Rvir(zmid, Mh) / 1e3 # kpc->Mpc
            _iz = np.argmin(np.abs(zmid - self.sim.pops[pid].halos.tab_z))

            # Remaining dimensions (Mh, R)
            Sall = self.sim.pops[pid].halos.tab_Sigma_nfw[_iz,:,:]
            Mall = self.sim.pops[pid].halos.tab_M

            R_pix = R_X = Rvir * 60 / mpc_per_arcmin / pix

            # Pixel coordinates in RA and DEC
            if postage_stamp is None:
                rr, dd = np.meshgrid(ra_c * 60 * mpc_per_arcmin,
                                dec_c * 60 * mpc_per_arcmin,
                                indexing='ij')


        elif include_galaxy_sizes:  
            resolved_sources = True

            assert self.profile_info is not None, \
                "Must supply `profile_info` at initialization!"

            R_sec, nsers, ellip, pa = self._get_size_catalog(zlim, logmlim,
                red, Mh, pid)

            Rvir = self.sim.pops[0].halos.get_Rvir(zmid, Mh) / 1e3 # kpc->Mpc

            ##
            # Next, impose effective stopping criterion in size where we
            # stop painting on Sersic profiles and just dump all photons
            # in a single pixel.
            #

            # Will paint anything with a half-light radius greater than a pixel
            if size_cut == 0.5:
                R_X = R_sec
            # General option: paint anything with size, defined as the
            # radius containing `size_cut` fraction of the light, that
            # exceeds a pixel.
            elif size_cut == 1:
                R_X = np.inf # ensures detailed model for every galaxy
            else:
                # e.g., if size_cut == 0.9, we'll find the radius containing
                # 90% of the light for a given galaxy, and if that radius is
                # bigger than a pixel, we'll model its profile.
                rcut = [self.sim.pops[pid].get_sersic_r_containing_lightfrac(
                    size_cut, nsers[h]) for h in range(R_sec.size)]

                # `rcut` is in units of the half-light radius, so we need
                # to multiply by `R_sec` to obtain the size in arcseconds.
                R_X = np.array(rmax) * R_sec

            ##
            # R_X here is still in arcseconds, will get converted to pixels
            # below.

            # Size in degrees
            R_deg = R_sec / 3600.
            # Size in pixels (`pix_deg` is the pixel scale in degrees)
            R_pix = R_deg / pix_deg

            # R_X is the threshold size of an object we'll model in detail.
            #
            R_X /= (3600 * pix_deg)

            # All in degrees
            x0, y0 = ra, dec
            a, b = R_deg, R_deg

            # Pixel coordinates in RA and DEC
            if postage_stamp is None:
                rr, dd = np.meshgrid(ra_c / pix_deg, dec_c / pix_deg,
                    indexing='ij')

            ##
            # Shorthand for later
            x_0 = ra / pix_deg
            y_0 = dec / pix_deg
            theta = pa * np.pi / 180.

            b_n = gammaincinv(2. * nsers, 0.5)
            a, b = R_pix, (1 - ellip) * R_pix
            cos_theta, sin_theta = np.cos(theta), np.sin(theta)
            #

        ##
        # Accelerated approach if not doing resolved sources
        if (not resolved_sources):
            _flux_ = None
            _img_, _xe_, _ye_ = np.histogram2d(ra, dec, 
                bins=(ra_e, dec_e), weights=flux)

            # Recall that `img` is a buffer to be incremented
            img += _img_
        else:

            ##
            # Actually sum fluxes from all objects in image plane.
            for h in range(ra.size):
    
                # Where this galaxy lives in pixel coordinates
                i, j = ra_ind[h], de_ind[h]
    
                # Grab the flux
                _flux_ = flux[h]
    
                # HERE: account for fact that galaxies aren't point sources.
                # [optional]
                if self.sim.pops[pid].is_diffuse and include_galaxy_sizes and (R_X[h] >= 1):
                    # Interpolate between tabulated solutions.
                    iM = np.argmin(np.abs(Mh[h] - Mall))
    
                    if postage_stamp is not None:
                        xx, yy, _r_ = self._get_postage_stamp_pix(R_pix[h], postage_stamp)
    
                        # This is in pixels, need to convert to cMpc before
                        # interpolating
                        Rarr = np.sqrt(xx**2 + yy**2) * (pix / 60.) \
                            * mpc_per_arcmin
    
                        I = self._get_ihl_postage_stamp(_r_, Rarr, Rall, Sall, iM)
    
                        # OK, now need to drop into full image
                        slcx, slcy, slcx2, slcy2 = \
                            self._get_postage_stamp_slices(I, img, i, j)
    
                    else:
                        # Image of distances from halo center
                        r0 = ra_c[i] * 60 * mpc_per_arcmin
                        d0 = dec_c[j] * 60 * mpc_per_arcmin
                        Rarr = np.sqrt((rr - r0)**2 + (dd - d0)**2)
    
                        # In Msun/cMpc^3
                        I = np.interp(np.log10(Rarr), np.log10(Rall), Sall[iM,:])
    
                    # Optional: hard cut at large radius.
                    I[Rarr >= null_beyond_size * Rvir[h]] = 0
    
                    tot = I.sum()
    
                    if postage_stamp is not None:
                        img[slcx,slcy] += _flux_ * I[slcx2,slcy2] \
                            / I[slcx2,slcy2].sum()
                    elif tot == 0:
                        img[i,j] += _flux_
                    else:
                        img[:,:] += _flux_ * I / tot
    
                elif include_galaxy_sizes and (R_X[h] >= 1):
    
                    if postage_stamp is not None:
    
                        xx, yy, _r_ = self._get_postage_stamp_pix(R_pix[h], postage_stamp)
    
                        # This is in pixels, need to convert to cMpc before
                        # interpolating
                        Rarr = np.sqrt(xx**2 + yy**2) * (pix / 60.) \
                            * mpc_per_arcmin
    
                        # Put galaxies at the center of the postage stamp, hence
                        # no (xx - x_0) factors, just xx
                        x_maj = xx * cos_theta[h] + yy * sin_theta[h]
                        x_min = -xx * sin_theta[h] + yy * cos_theta[h]
                        #z = np.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)
                        zsq = (x_maj / a[h])**2 + (x_min / b[h])**2
    
                        # Fractional contribution to total flux
                        pstamp = np.exp(-b_n[h] * (zsq**(1. / nsers[h] / 2.) - 1))
    
                        slcx, slcy, slcx2, slcy2 = \
                            self._get_postage_stamp_slices(pstamp, img, i, j)
    
                        I = pstamp
    
                    else:
                        Rarr = np.sqrt((rr - x_0[h])**2 + (dd - y_0[h])**2)
    
                        x_maj =  (rr - x_0[h]) * cos_theta[h] \
                              + (dd - y_0[h]) * sin_theta[h]
                        x_min = -(rr - x_0[h]) * sin_theta[h] \
                              + (dd - y_0[h]) * cos_theta[h]
                        #z = np.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)
                        zsq = (x_maj / a[h])**2 + (x_min / b[h])**2
    
                        # Fractional contribution to total flux
                        I = np.exp(-b_n[h] * (zsq**(1. / nsers[h] / 2.) - 1))
    
                    # Optional: hard cut at large radius.
                    I[Rarr >= null_beyond_size * Rvir[h]] = 0
    
                    # Get total flux
                    tot = I.sum()
    
                    if postage_stamp is not None:
                        img[slcx,slcy] += _flux_ * pstamp[slcx2,slcy2] \
                            / pstamp[slcx2,slcy2].sum()
                    elif tot == 0 or R_X[h] < 1:
                        img[i,j] += _flux_
                    else:
                        img[:,:] += _flux_ * I / tot
    
                ##
                # Otherwise just add flux to single pixel
                else:
                    img[i,j] += _flux_
    
        ##
        # Clear out some memory sheesh
        del flux, _flux_, ra, dec, red, Mh, ok, okp, okz, ra_ind, de_ind, \
            mask_ra, mask_de
        if self.mem_concious:
            gc.collect()

    def _get_map_from_cat(self, fov, pix, ra, dec, red, flux, pid,
        include_galaxy_sizes):
        ##
        # Need some extra info to do more sophisticated modeling...
        ##
        raise NotImplemented('help')

        ra_e, ra_c, dec_e, dec_c = self.get_pixels(fov, pix=pix)

        # Extended emission from IHL
        if self.sim.pops[pid].is_diffuse and include_galaxy_sizes:

            Rmi, Rma = -3, 1
            dlogR = 0.25
            Rall = 10**np.arange(Rmi, Rma+dlogR, dlogR)

            _iz = np.argmin(np.abs(zmid - self.sim.pops[pid].halos.tab_z))

            # Remaining dimensions (Mh, R)
            Sall = self.sim.pops[pid].halos.tab_Sigma_nfw[_iz,:,:]
            Mall = self.sim.pops[pid].halos.tab_M

            mpc_per_arcmin = self.sim.cosm.get_angle_from_length_comoving(zmid,
                pix / 60.)

            rr, dd = np.meshgrid(ra_c * 60 * mpc_per_arcmin,
                                dec_c * 60 * mpc_per_arcmin,
                                indexing='ij')


        elif include_galaxy_sizes:

            assert self.profile_info is not None, \
                "Must supply `profile_info` at initialization!"

            R_sec, nsers, ellip, pa = self._get_size_catalog(zlim, logmlim,
                red, Mh, pid)

            ##
            # Next, impose effective stopping criterion in size where we
            # stop painting on Sersic profiles and just dump all photons
            # in a single pixel.
            #

            # Will paint anything half-light radius greater than a pixel
            if size_cut == 0.5:
                R_X = R_sec
            # General option: paint anything with size, defined as the
            # radius containing `size_cut` fraction of the light, that
            # exceeds a pixel.
            else:
                rmax = [self.sim.pops[pid].get_sersic_rmax(size_cut,
                    nsers[h]) for h in range(R_sec.size)]

                R_X = np.array(rmax) * R_sec

            #R_sec = Rkpc * self.cosmo.arcsec_per_kpc_proper(red).to_value()

            # Size in degrees
            R_deg = R_sec / 3600.
            R_pix = R_deg / pix_deg

            R_X /= (3600 * pix_deg)

            # All in degrees
            x0, y0 = ra, dec
            a, b = R_deg, R_deg

            rr, dd = np.meshgrid(ra_c / pix_deg, dec_c / pix_deg,
                indexing='ij')

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
            if self.sim.pops[pid].is_diffuse and include_galaxy_sizes:

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

                #print(f"doing IHL, _flux_={_flux_}, tot={tot}")

            elif include_galaxy_sizes and R_X[h] >= 1:

                model_SB = Sersic2D(amplitude=1., r_eff=R_pix[h],
                    x_0=ra[h] / pix_deg, y_0=dec[h] / pix_deg,
                    n=nsers[h], theta=pa[h] * np.pi / 180.,
                    ellip=ellip[h])

                # Fractional contribution to total flux
                I = model_SB(rr, dd)
                tot = I.sum()

                ##
                # Test: null flux from beyond 4 R_e
                #dr = np.sqrt((rr - ra[h] / pix_deg)**2 \
                #   +         (dd - dec[h] / pix_deg)**2)
                #beyond_edges = dr > 8 * R_pix[h]
                #I[beyond_edges==1] = 0

                #print('hi', h, R_pix[h], I.sum())

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
        del flux, _flux_, ra, dec, red, Mh, ok, okp, okz, ra_ind, de_ind, \
            mask_ra, mask_de
        if self.mem_concious:
            gc.collect()

    def get_output_dir(self, fov, zlim, logmlim=None, force_chunk=False):
        fn = f"{self.base_dir}/fov_{fov:.1f}"
        fn += f"/box_{self.Lbox:.0f}/dim_{self.dims:.0f}"
        fn += f"/{self.model_name}"
        fn += f"/zmin_{self.zmin:.3f}"

        # Need directory for zmax, logmlim range
        final = (zlim[0] == self.zlim[0]) and (zlim[1] == self.zlim[1])

        # [new] Check if this redshift range spans more than one layer.
        # BUT: don't count if final=True, since the definition of final
        # is 100% of the layers
        all_zchunks = self.get_redshift_layers(self.zlim)
        ilo = np.argmin(np.abs(zlim[0] - all_zchunks[:,0]))
        ihi = np.argmin(np.abs(zlim[1] - all_zchunks[:,1]))
        is_chunk = (force_chunk or (ihi > ilo)) and (not final)

        #
        if final or is_chunk:
            fn += f"/zmax_{self.zlim[1]:.3f}"
            if logmlim is not None:
                fn += f"/m_{logmlim[0]:.2f}_{logmlim[1]:.2f}"
        else:
            fn += f'/checkpoints/z_{zlim[0]:.3f}_{zlim[1]:.3f}'
            if logmlim is not None:
                fn += f'/m_{logmlim[0]:.2f}_{logmlim[1]:.2f}'

        #
        if is_chunk:
            fn += f'/z_{zlim[0]:.3f}_{zlim[1]:.3f}'

        # Everything should exist up to the m_??.??_??.?? subdirectory
        if not os.path.exists(fn):
            path = Path(fn)
            path.mkdir(parents=True)

        return fn

    def get_map_fn(self, fov, pix, channel, popid, logmlim=None, zlim=None,
        fmt='fits', wave_units='um', force_chunk=False, suffix=None,
        include_galaxy_sizes=False):
        """
        Return filename expected for map with given properties.
        """

        save_dir = self.get_output_dir(fov=fov, 
            zlim=zlim, logmlim=logmlim, force_chunk=force_chunk)

        pid, pid_parent, pid_str = get_pop_info(popid)

        fn = f'{save_dir}/map_pix_{pix:.1f}_{channel[0]:.3f}_{channel[1]:.3f}_{wave_units}_pop_{pid_str}'

        if include_galaxy_sizes:
            if popid in [4, '4']:
                fn += '_prof_nfw'
            else:
                fn += '_prof_sers'
        else:
            fn += '_prof_delt'

        if suffix is not None:
            fn += f'_{suffix}'

        return fn + '.' + fmt

    def get_cat_fn(self, fov, channel, popid, logmlim=None, zlim=None,
        fmt='fits', wave_units='um', suffix=None):
        """
        Return filename expected for catalog with given properties.
        """

        save_dir = self.get_output_dir(fov=fov, 
            zlim=zlim, logmlim=logmlim)

        pid, pid_parent, pid_str = get_pop_info(popid)

        if type(channel) in [tuple, list, np.ndarray]:
            fn = f'{save_dir}/cat_{channel[0]:.3f}_{channel[1]:.3f}_{wave_units}_pop_{pid_str}'
        else:
            fn = f'{save_dir}/cat_{channel}_pop_{pid_str}'

        if suffix is not None:
            fn += f'_{suffix}'

        return fn + '.' + fmt

    def get_README(self, fov, zlim=None, logmlim=None,
        is_map=True, verbose=False):
        """

        """

        assert is_map

        base_dir = self.get_output_dir(fov, zlim=zlim, logmlim=logmlim)

        hdr = "#" * 78
        hdr += '\n# README\n'
        hdr += "#" * 78
        hdr +=  "\n# This is an automatically-generated file! \n"
        hdr += "# It contains some basic metadata for maps once they are available.\n"
        hdr += "# Note: all wavelengths here are in microns.\n"
        hdr += "#" * 78
        hdr += "\n"
        hdr += "# channel name; central wavelength; "
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

    def _filter_by_fov(self, ok):
        """

        """

        ids_in = np.arange(ok.size, dtype=int)
        ids_out = []

        ct = 0
        for id in ids_in:
            if ok[id]:
                ids_out.append((id, ct))
            else:
                continue

            ct += 1

        ids_out = np.array(ids_out, dtype=int)

        return ids_out

    def _refresh_sat_ids(self, ids_in, ids_out, parents_in):
        """
        Initially we record the parent ID of satellites as the index of the
        parent in a particular layer BEFORE any FoV filtering. After filtering,
        we must adjust the indices accordingly. This routine figures out the
        mapping between indices before and after FoV filtering.

        Parameters
        ----------
        ids_in : np.ndarray
            Indices of central halos BEFORE filtering on FoV.
        ids_out : np.ndarray
            Final indices of central halos.
        parents_in : np.ndarray
            Indices corresponding to parent ID of each satellite BEFORE
            the FoV filter.

        Returns
        -------
        Tuple containing: (new parent IDs AFTER FoV filter, mask indicating which
        centrals in original catalog were filtered out by FoV cut). Note that
        the length of these two arrays will be different anytime some > 0
        number of halos are filtered out by the FoV cut.
        """

        p_out = []
        cen_ok = []
        for i, p_in in enumerate(parents_in):
            # Means that the parent of this satellite ended up outside the FoV
            if p_in not in ids_in:
                cen_ok.append(0)
                continue

            i_out = np.argwhere(p_in == ids_in).squeeze()
            new_id = ids_out[i_out]
            p_out.append(new_id)
            cen_ok.append(1)

        return np.array(p_out, dtype=int), np.array(cen_ok)

    def generate_cats(self, fov, channels, logmlim, dlogm=0.5, zlim=None,
        include_galaxy_sizes=False, dlam=20, path='.', channel_names=None,
        suffix=None, fmt='fits', hdr={}, wave_units='um',
        cat_units='uJy', keep_layers=False, logmlim_sats=(11,15),
        include_pops=[0], clobber=False, verbose=False, dryrun=False,
        use_pbar=True, **kwargs):
        """
        Generate galaxy catalogs.

        Parameters
        ----------
        fov : int, float
            Field of view (single dimension) in degrees, so total area is
            FOV^2/deg^2.
        

        """

        # Create root directory if it doesn't already exist.
        self.build_directory_structure(fov, dryrun=False)

        # Create root directory if it doesn't already exist.
        base_dir = self.get_output_dir(fov, zlim=self.zlim, logmlim=logmlim)

        # At least save halo mass since we get it for free.
        if (channels is None):
            channels = ['Mh']
            # Override cat_units
            cat_units = 'Msun'

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

        zlayers = self.get_redshift_layers(self.zlim)
        zcent, ze, Re = self.get_domain_info(self.zlim)
        mlayers = self.get_mass_layers(logmlim, dlogm)

        all_layers = self.get_layers(channels, logmlim, dlogm=dlogm,
            include_pops=include_pops, channel_names=channel_names)

        # Progress bar
        pb = ProgressBar(len(all_layers),
            name="cat(Mh>={:.1f}, Mh<{:.1f}, z>={:.3f}, z<{:.3f})".format(
                logmlim[0], logmlim[1], zlim[0], zlim[1]),
            use=use_pbar)
        pb.start()

        ##
        # Start doing work.
        ct = 0

        Nlayers = len(zlayers) * len(mlayers)

        # The `tracker` keeps track of central halos. This is because we 
        # use indices to keep track of the parents of satellites, so from
        # one iteration to the next we need a running tally to get our 
        # indices right.
        tracker = {}
        tracker_flat = {}
        for popid in include_pops:
            pid, pid_par, pid_str = get_pop_info(popid)
            if pid not in tracker:
                tracker[pid] = np.zeros((len(zlayers), len(mlayers)), dtype=int)
                tracker_flat[pid] = [None] * Nlayers
                tracker_flat[pid][0] = 0
            if pid_par not in tracker:
                tracker[pid_par] = np.zeros((len(zlayers), len(mlayers)), dtype=int)
                tracker_flat[pid_par] = [None] * Nlayers
                tracker_flat[pid_par][0] = 0

        #tracker = {str(pid): np.zeros((len(zlayers), len(mlayers)), dtype=int) \
        #    for pid in include_pops}

        #tracker_flat = {str(pid): [None] * Nlayers for pid in include_pops}
        #for pid in include_pops:
        #    tracker_flat[str(pid)][0] = 0

        ra = []
        dec = []
        red = []
        dat = []
        parh = []
        for h, layer in enumerate(all_layers):

            # Unpack info about this layer
            popid, channel, chname, zlayer, mlayer = layer

            # Need channel in microns for internal routines
            chan_mic = self.convert_chan_to_micron(channel, wave_units)

            # Just used for file naming
            field_names = ['ra', 'dec', 'z', channel]
            field_units = ['deg', 'deg', '', cat_units]

            # Retrieve info about population:
            # ARES ID, parent ID (in ARES), `popid` as string
            pid, pid_par, pid_str = get_pop_info(popid)

            # Short-hand needed below
            zlo, zhi = zlayer

            # Get number of z layer
            iz = np.digitize(zlayer.mean(), bins=zlayers[:,0]) - 1

            # Get number of M layer
            im = np.argmin(np.abs(mlayer[0] - mlayers[:,0]))

            izm = iz * len(mlayers) + im

            # See if we already finished this map.
            # Note that if this file exists, it's guaranteed that the
            # corresponding ra, dec, and redshift catalogs are done too.
            fn = self.get_cat_fn(fov, channel, popid,
                logmlim=mlayer, zlim=zlayer, wave_units=wave_units)

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
                _ra, _dec, _red, _Mh, _parents = \
                    self.get_catalog_halos(zlim=zlayer,
                    logmlim=mlayer, popid=popid, verbose=verbose,
                    satellites=self.sim.pops[pid].is_satellite_pop,
                    logmlim_sats=logmlim_sats)

                # Should be able to cache this, no? Just until we get
                # to the next redshift and/or mass bin?
                # Or, read from catalog? I/O can be slow...

                # Could be empty layers for very massive halos and/or early times.
                if (_ra is None) or (len(_ra) == 0):
                    # You might think: let's `continue` to the next iteration!
                    # BUT, if we do that, and we're really unlucky and this
                    # happens on the last layer of work for a given channel,
                    # then no checkpoint will be written below :/
                    # Hence the use of `pass` here intead.
                    if (izm < Nlayers - 1):
                        tracker_flat[pid_par][izm+1] = tracker_flat[pid_par][izm]

                    tracker[pid_par][iz,im] = 0

                    _parents = []
                else:

                    # Correct for field position. Always (0,0) for log-normal boxes,
                    # may not be for halo catalogs from sims.
                    _ra -= self.fxy[0]
                    _dec -= self.fxy[1]

                    # Hack out galaxies outside our requested lightcone.
                    ok = np.logical_and(np.abs(_ra)  < fov / 2.,
                                        np.abs(_dec) < fov / 2.)

                    # Isolate OK entries.
                    _ra = _ra[ok==1]
                    _dec = _dec[ok==1]
                    _red = _red[ok==1]
                    _Mh = _Mh[ok==1]

                    # Handle satellites
                    if self.sim.pops[pid].is_satellite_pop:
                        if ok.sum():
                            _parents = _parents[ok==1]

                            _ra_c, _dec_c, _red_c, _Mh_c, _parents_c = \
                                self.get_catalog_halos(zlim=zlayer,
                                logmlim=mlayer, popid=popid, verbose=verbose)

                            # Problem: `_parents` are indices generated within
                            # each layer, need to be incremented so that
                            # elements point to the right central in the
                            # FINAL halo catalog. So, we need to increment by
                            # the number of halos up to but NOT including
                            # this layer.

                            ok_c = np.logical_and(np.abs(_ra_c)  < fov / 2.,
                                                  np.abs(_dec_c) < fov / 2.)

                            tracker[pid_par][iz,im] = ok_c.sum()

                            if izm == 0:
                                Ncen = 0
                            else:
                                Ncen = tracker_flat[pid_par][izm]

                            # Prep for next iteration
                            if (izm < Nlayers - 1) and (tracker_flat[pid_par][izm+1] is None):
                                tracker_flat[pid_par][izm+1] = ok_c.sum() \
                                    + tracker_flat[pid_par][izm]

                            if ok_c.sum():
                                ids_in, ids_out = self._filter_by_fov(ok_c).T

                                # Need to worry about satellites being ok
                                # but their centrals being not OK.
                                _parents, cen_ok = \
                                    self._refresh_sat_ids(ids_in, ids_out, _parents)

                                if not np.all(cen_ok):
                                    _ra = _ra[cen_ok==1]
                                    _dec = _dec[cen_ok==1]
                                    _red = _red[cen_ok==1]
                                    _Mh = _Mh[cen_ok==1]

                                if ok_c.sum() > 0:
                                    _parents += Ncen
                            else:
                                _parents = _ra = _dec = _red = _Mh = []

                            # Done dealing with scenario in which >0 satellites
                            # are (at least initially) `ok`.
                        else:
                            # This means there aren't any satellites
                            # in the FoV.
                            _parents = _ra = _dec = _red = _Mh = []
                            if (izm < Nlayers - 1):
                                tracker_flat[pid_par][izm+1] = \
                                    tracker_flat[pid_par][izm]
                            tracker[pid_par][iz,im] = 0

                        ##
                        # Done with satellites
                        if len(_parents) != len(_ra):
                            print('problem with _parents 1', popid, izm, len(_parents), len(_ra))
                            input('<enter>')

                    ct += ok.sum()

                    if len(_ra) > 0:
                        ra.extend(list(_ra))
                        dec.extend(list(_dec))
                        red.extend(list(_red))

                        if self.sim.pops[pid].is_satellite_pop:
                            parh.extend(list(_parents))

                            if len(_parents) != len(_ra):
                                print('problem with _parents 2', popid, izm, len(_parents), len(_ra))

                        ##
                        # Unpack channel info
                        # Could be name of field, e.g., 'Mh', 'SFR', 'Mstell',
                        # photometric info, e.g., ('roman', 'F087'),
                        # or special quantities like Ly-a EW or luminosity.
                        # Note: if pops[popid] is a GalaxyEnsemble object
                        if type(channel) in [tuple, list, np.ndarray]:
                            # Internally, these fluxes are always in
                            # erg/s/cm^2/Ang, but then integrated over channel.
                            # Will need channel width in Hz to recover specific
                            # intensities averaged over band.
                            nu = c * 1e4 / np.mean(chan_mic)
                            dnu = c * 1e4 * (chan_mic[1] - chan_mic[0]) / np.mean(chan_mic)**2

                            _dat = self._get_flux_catalog(zlayer, logmlim, _red, _Mh,
                                chan_mic, pid)
                            _dat *= self.get_map_norm(cat_units) / dnu
                        elif channel in ['Mh']:
                            _dat = _Mh
                        elif channel in ['parents']:
                            _dat = _parents
                        elif channel.lower().startswith('ew'):
                            raise NotImplemented('help')
                        elif channel.lower() == 'sfr':
                            _dat = self.sim.pops[pid].get_sfr(z=_red, Mh=_Mh)
                        elif channel.lower() in ['ms', 'mstell']:
                            raise NotImplemented('help')
                        elif channel.lower() in ['ellip', 'nsers', 'pa', 'r50']:
                            R_sec, nsers, ellip, pa = self._get_size_catalog(zlim,
                                logmlim, _red, _Mh, pid)

                            _dat_dict = {'r50': R_sec, 'nsers': nsers,
                                'ellip': ellip, 'pa': pa}

                            _dat = _dat_dict[channel.lower()]
                        else:
                            cam, filt = channel.split('_')

                            #raise NotImplemented('do we need to do this anymore?')

                            ##
                            # Once again, in general need to sub-cycle through z
                            # to preserve accuracy.
                            zsub_lo = 1 * zlo

                            mags = np.inf * np.ones(_Mh.size)
                            while zsub_lo < zhi:

                                zsub_hi = min(zsub_lo + self.dz_max, zhi)

                                zsub_mid = np.mean([zsub_lo, zsub_hi])

                                okzsub = np.logical_and(_red >= zsub_lo,
                                                        _red < zsub_hi)

                                _filt, out = \
                                    self.sim.pops[pid].get_mags(zsub_mid,
                                    absolute=False, cam=cam, filters=[filt],
                                    Mh=_Mh[okzsub==1])

                                # There's a meaningless second dimension here
                                # because get_mags can report mags for multiple
                                # filters at once, we're just not doing that here.
                                mags[okzsub==1] = out[:,0]
                                zsub_lo += self.dz_max

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
                        if keep_layers:

                            for ff, field in enumerate([_ra, _dec, _red, _dat]):
                                # e.g., `parents` field for centrals is None
                                if field in [[], None]:
                                    continue

                                fn_ff = self.get_cat_fn(fov, field_names[ff],
                                    popid, logmlim=mlayer, zlim=zlayer,
                                    wave_units=wave_units)
                                self.save_cat(fn_ff, field, field_names[ff],
                                    zlayer, mlayer, fov, fmt=fmt, hdr=hdr,
                                    cat_units=field_units[ff],
                                    clobber=clobber, verbose=verbose)

                        ##
                        # This is just because all datasets will be arrays if
                        # they contain entries. If there are no entries, _dat
                        # will be either an empty list or None. The latter
                        # case is what we're trying to avoid here since
                        # len(None) = error.
                        if (type(_dat) == np.ndarray):
                            dat.extend(list(_dat))
                        else:
                            pass
                    ##
                    # len(_ra) == 0, i.e., no halos to do anything with
                    else:
                        pass

                # End of else block that generates new catalog if one isn't found.

            # Back to level of loop over layers of work.

            ##
            # Figure out if we're done with all the layers
            if h == len(all_layers) - 1:
                done_w_chan_or_pop = True
            else:
                done_w_chan_or_pop = np.logical_or(
                    channel != all_layers[h+1][1],
                    popid != all_layers[h+1][0])

            # If we're done with this channel, save file containing
            # full redshift and mass range.
            # Only reason we do np.all here is because a spectral channel will
            # be a 2-element tuple.
            if np.all(done_w_chan_or_pop):
                #_fn = self.get_cat_fn(fov, pix, channel, popid,
                #    logmlim=logmlim, zlim=self.zlim, fmt=fmt)


                for ff, field in enumerate([ra, dec, red, dat]):
                    # e.g., `parents` field for centrals is None
                    if field in [[], None]:
                        continue
                    
                    if type(field_names[ff]) == str:
                        if field_names[ff] == 'parents':
                            if len(field) != len(ra):
                                print('problem with _parents 3', popid, logmlim, len(parents), len(ra))
                                #input('<enter>')

                    _fn_ff = self.get_cat_fn(fov, field_names[ff], popid,
                        logmlim=logmlim, zlim=self.zlim, fmt=fmt, wave_units=wave_units,
                        suffix=suffix)

                    self.save_cat(_fn_ff, field,
                        field_names[ff], self.zlim, logmlim,
                        fov, fmt=fmt, hdr=hdr, cat_units=field_units[ff],
                        clobber=clobber, verbose=verbose)

                del ra, dec, red, dat, parh
                dat = []
                ra = []
                dec = []
                red = []
                parh = []

        pb.finish()

        ##
        # Done
        return

    def get_layers(self, channels, logmlim, dlogm=0.5, include_pops=[0],
        channel_names=None):
        """
        Take a list of channels, populations, and bounds in halo mass,
        and construct a list of layers of work to do of the form:

        all_layers = [
            (popid, channel, chname, zlayer, mlayer),
            (popid, channel, chname, zlayer, mlayer),
            (popid, channel, chname, zlayer, mlayer),
            (popid, channel, chname, zlayer, mlayer),
          ...
        ]

        Basically this allows us to 'flatten' a series of for loops over
        spectral channels, populations, redshift, and mass layers into
        a single loop. Just unpack as, e.g.,

        >>> all_layers = self.get_layers(channels, logmlim, dlogm=dlogm,
        >>>    include_pops=include_pops)
        >>> for layer in all_layers:
        >>>    popid, channel, chname, zlayer, mlayer = layer
        >>>    <do cool stuff>

        """

        zlayers = self.get_redshift_layers(self.zlim)
        mlayers = self.get_mass_layers(logmlim, dlogm)
        players = include_pops

        if channel_names is None:
            channel_names = [None] * len(channels)

        all_layers = []
        for h, popid in enumerate(players):
            for i, channel in enumerate(channels):
                for j, zlayer in enumerate(zlayers):

                    # Option to limit redshift range.
                    zlo, zhi = zlayer

                    for k, mlayer in enumerate(mlayers):
                        all_layers.append((popid, channel, channel_names[i],
                            zlayer, mlayer))

        return all_layers

    def _check_for_corrupted_files(self, fov, pix, channels, logmlim, dlogm,
        include_pops, channel_names=None, include_galaxy_sizes=False):
        """
        When running on a cluster, occasionally we get really unlucky and an
        output file will be corrupted, (probably) because we hit the wallclock
        time limit on the job while the file is being written. This routine
        does a cursory check that pre-existing files all have the same size, as
        a quick-and-dirty way of rooting out corrupted files.
        """


        # Assemble list of map layers to run.
        all_layers = self.get_layers(channels, logmlim, dlogm=dlogm,
            include_pops=include_pops, channel_names=channel_names)

        all_zlayers = np.array(self.get_redshift_layers(self.zlim))
        all_mlayers = np.array(self.get_mass_layers(logmlim, dlogm))

        # Check status before we start
        all_sizes = np.zeros(len(all_layers))
        all_fn = []

        for h, layer in enumerate(all_layers):

            # Unpack info about this layer
            popid, channel, chname, zlayer, mlayer = layer

            # See if we already finished this map.
            fn = self.get_map_fn(fov, pix, channel, popid,
                logmlim=mlayer, zlim=zlayer,
                include_galaxy_sizes=include_galaxy_sizes)

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

                print(f"! Problem file for layer={h}: {fn}.")

            ##
            # Consistent with failed write as job is killed
            if len(probs) == 1:
                #os.remove(probs[0])
                print(f"! Removed corrupted file {fn}.")
            else:
                raise IOError('! {len(probs)} corrupted files detected. Help?')

        elif np.all(all_sizes == 0):
            # Means this is the first time the mock is being run.
            pass
        else:
            ##
            # Made it here? All good
            print(f"! No corrupted files detected! All {len(all_layers)} layers look good.")

    def get_map_norm(self, map_units, pix=None):
        """
        Remember: we're using cgs units internally. This method determines the
        conversion factor to user's favorite `map_units` (within reason).

        Parameters
        ----------
        map_units : str
            Current options are 'si' (nW/m^2/sr^1), 'cgs' (erg/s/cm^2),
            or 'MJy/sr'. Case insensitive.
        pix : int, float
            Pixel scale [arcseconds]. This is just in here because we are
            generating fluxes *per pixel* first and so much convert to
            per solid angle units.

        Returns
        -------
        Normalization factor, i.e., if you multiply by this number it will
        convert intensities *from* cgs *to* `map_units`.
        """
        if (map_units.lower() == 'si') or ('nw/m^2' in map_units.lower()):
            # aka (1e2)^2 / 0.01 = 1e6
            f_norm = cm_per_m**2 / erg_per_s_per_nW
        elif map_units.lower() == 'cgs':
            f_norm = 1.
        elif 'mjy' in map_units.lower() :
            # 1 MJy = 1e6 Jy = 1e6 * 1e-23 erg/s/cm^2/sr -> 1e17 MJy / cgs units
            f_norm = 1e17
        elif 'ujy' in map_units.lower() :
            # 1 micro-Jy = 1e-6 Jy = 1e-6 * 1e-23 erg/s/cm^2/sr -> 1e29 uJy / cgs units
            f_norm = 1e29
        else:
            raise ValueErorr(f"Unrecognized option `map_units={map_units}`")

        if pix is not None:
            pix_deg = pix / 3600.
            if '/sr' in map_units.lower():
                sr_per_pix = pix_deg**2 / sqdeg_per_std
                f_norm /= sr_per_pix

        return f_norm

    def _check_chunks(self, keep_chunks):
        """
        Go through user-provided `keep_chunks`, check to see that their demands
        can be met, and offer up slightly modified chunk edges if they've
        strayed from what's actually available. We'll also return a list of
        custom redshift layers that are needed in order to construct the
        desired chunks in post processing.

        Returns
        -------
        Tuple containing: (keep_chunks -> closest available redshifts,
            bounding indices of redshift layers in chunks,
            list of custom redshift layers needed to be able to construct
            the requested chunks)



        """

        if keep_chunks is None:
            return None, None, None

        zlayers = self.get_redshift_layers(self.zlim)

        chunks_edges = []
        chunks_edges_ids = []
        zlayers_minimal = []

        for (zlo, zhi) in keep_chunks:

            i = np.argmin(np.abs(zlo - zlayers[:,0]))
            j = np.argmin(np.abs(zhi - zlayers[:,1]))

            zlayers_minimal.extend(list(np.arange(i,j+1)))

            chunks_edges.append((zlayers[i,0], zlayers[j,1]))
            chunks_edges_ids.append((i, j))

        return chunks_edges, chunks_edges_ids, list(np.sort(zlayers_minimal))

    def convert_chan_to_micron(self, channel, wave_units='um'):
        """

        """

        if wave_units == 'um':
            return channel
        elif wave_units == 'ghz':
            lam_obs = c * 1e4 / np.array(channel) / 1e9
            return tuple(lam_obs[-1::-1])

    def generate_maps(self, fov, pix, channels, logmlim, dlogm=1,
        include_galaxy_sizes=False, null_beyond_size=np.inf, size_cut=0.5, dlam=20,
        suffix=None, fmt='fits', hdr={}, map_units='MJy/sr', channel_names=None,
        include_pops=[0], clobber=False, wave_units='um',
        load_if_found=True, keep_layers_custom_z=None, keep_layers=False,
        keep_chunks=None, use_pbar=False, verbose=False, dryrun=False,
        logmlim_sats=(11,15),
        postage_stamp=5, nthreads=None, **kwargs):

        """
        Write maps in one or more spectral channels to disk.

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
        include_galaxy_sizes : bool
            If True, use empirical mass-size relations to paint on galaxy
            surface brightness profiles (assume Sersic). Relies on parameter
            `pop_msr`, a function of argument `z` and `Ms`.
        size_cut : float
            It is computationally expensive to generate galaxy sizes. So, for
            sufficiently small galaxies, we revert to the point source treatment.
            `size_cut` determines when we revert -- if size_cut=0.5, it means
            that any galaxy with half-light radius >= 1 pixel will be modeled
            in detail. If `size_cut=0.9`, it means any galaxy whose 90%-light
            radius is bigger than a pixel will be modeled. Bigger numbers mean
            more expensive calculations.
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
        keep_layers : bool
            If True, individual mass and redshift 'layers' will be saved to
            disk in the `checkpoints` subdirectory. This can get heavy for
            big mocks -- see next parameter for another option.
        keep_layers_custom_z : list
            If provided, this is a list of individual layers to save (i.e.,
            not all of them). Note that these need to be integers for now, so
            you have to kind of know what you're doing. See the method
            `get_redshift_layers` to reveal the co-eval redshift layers
            that are available.

        Returns
        -------
        Right now, nothing. Just saves files to disk.

        """

        pix_deg = pix / 3600.

        # Create root directory if it doesn't already exist.
        self.build_directory_structure(fov, dryrun=False)

        # Must do this after building the directory tree otherwise
        # we'll get errors.
        if not clobber:
            self._check_for_corrupted_files(fov, pix, channels,
                logmlim=logmlim, dlogm=dlogm,
                include_pops=include_pops, channel_names=channel_names,
                include_galaxy_sizes=include_galaxy_sizes)

        ##
        # Initialize a README file / see what's in it.
        README = self.get_README(fov=fov, zlim=self.zlim,
            logmlim=logmlim)

        # For final outputs
        final_dir = self.get_output_dir(fov=fov, zlim=self.zlim,
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

        assert fov * 3600 / pix % 1 == 0, \
            "FOV must be integer number of pixels wide!"

        npix = int(fov * 3600 / pix)

        # Converts from cgs [internal units] to `map_units`
        f_norm = self.get_map_norm(map_units, pix)

        # Assemble list of map layers to run.
        all_layers = self.get_layers(channels, logmlim, dlogm=dlogm,
            include_pops=include_pops, channel_names=channel_names)

        all_zlayers = np.array(self.get_redshift_layers(self.zlim))
        all_mlayers = np.array(self.get_mass_layers(logmlim, dlogm))

        # Users can keep custom chunks (i.e., sums over layers)
        if keep_chunks is not None:
            assert keep_layers, "Must set keep_layers=True to `keep_chunks`."
            chunks_edges, chunks_edges_ids, chunks_zlayers_needed = \
                self._check_chunks(keep_chunks)
        else:
            chunk_edges = chunk_edges_ids = chunks_zlayers_needed = None

        # User can custom define subset of redshift layers to save
        # (this is a computational choice: saving all can be ~TBs of images)
        if keep_layers:
            if (keep_layers_custom_z == None):
                _keep_layers_custom = list(np.arange(0, len(all_zlayers)))
            else:
                _keep_layers_custom = list(keep_layers_custom_z)

            # Make sure we save the layers needed to build provided chunks
            if keep_chunks is not None:
                for layer_id in chunks_zlayers_needed:
                    if layer_id not in _keep_layers_custom:
                        _keep_layers_custom.append(layer_id)
                        if verbose:
                            print(f"! Added layer {layer_id} to list of layers to keep.")

                _keep_layers_custom = list(np.sort(_keep_layers_custom))
        else:
            if keep_layers_custom_z is not None:
                raise ValueError('You set keep_layers_custom_z but not keep_layers! Set latter to True (probably).')

        # Array telling us which layers were already done and which
        # we ran from scratch so at the end we know whether to update
        # the channel maps.
        # Recall that if we changed zmax, final maps will go in a new
        # subdirectory.
        status_done_pre = np.zeros((len(include_pops), len(channels),
            len(all_zlayers), len(all_mlayers)))
        status_done_now = status_done_pre.copy()

        ##
        # Check status before we start
        for h, layer in enumerate(all_layers):

            # Unpack info about this layer
            popid, channel, chname, zlayer, mlayer = layer

            # Identify indices of each (channel, z, m) layer
            ichan = np.argmin(np.abs(channel[0] - channels[:,0]))
            iz = np.argmin(np.abs(zlayer[0] - all_zlayers[:,0]))
            im = np.argmin(np.abs(mlayer[0] - all_mlayers[:,0]))
            ip = include_pops.index(popid)

            if np.all(status_done_pre[ip,ichan,:,:]) == 1:
                continue

            # Check first for final map.
            fn = self.get_map_fn(fov, pix, channel, popid,
                logmlim=logmlim, zlim=self.zlim,
                wave_units=wave_units, suffix=suffix,
                include_galaxy_sizes=include_galaxy_sizes)

            if os.path.exists(fn) and (not clobber):
                status_done_pre[ip,ichan,:,:] = 1
                print(f"! Final map for popid={ip} and channel={channel} exists.")
                continue

            # See if we already finished this map.
            fn = self.get_map_fn(fov, pix, channel, popid,
                logmlim=mlayer, zlim=zlayer, wave_units=wave_units,
                suffix=suffix,
                include_galaxy_sizes=include_galaxy_sizes)

            if os.path.exists(fn) and (not clobber):
                status_done_pre[ip,ichan,iz,im] = 1

        ##
        # If all maps done, exit.
        if np.all(status_done_pre == 1):
            return

        # Progress bar
        pb = ProgressBar(len(all_layers),
            name="img(Mh>={:.1f}, Mh<{:.1f}, z>={:.3f}, z<{:.3f})".format(
                logmlim[0], logmlim[1], zlim[0], zlim[1]),
            use=use_pbar)
        pb.start()

        # Make preliminary buffer for channel map (hence 'c' + 'img')
        cimg = np.zeros([npix]*2)

        if verbose:
            print(f"# Generating {len(all_layers)} individual map layers...")

        ##
        # Start doing work.
        # The way this works is we treat each layer: (z, M, pop, lambda)
        # separately. We'll keep a running tally of the "final" flux in any
        # given channel map as we go, and only create a new buffer when we
        # finish all the work for a single channel and a given population.
        for h, layer in enumerate(all_layers):

            # Unpack info about this layer
            popid, channel, chname, zlayer, mlayer = layer

            # Unpack popid more [as of March 2025]
            # (id number in ARES, parent ID number [if satellite], name as str)
            pid, pid_par, pid_str = get_pop_info(popid)

            # Identify indices of each (channel, z, m) layer
            ichan = np.argmin(np.abs(channel[0] - channels[:,0]))
            iz = np.argmin(np.abs(zlayer[0] - all_zlayers[:,0]))
            im = np.argmin(np.abs(mlayer[0] - all_mlayers[:,0]))
            ip = include_pops.index(popid)

            # Can only move on if ALL layers are already done, otherwise
            # it means the user has added z or m layers since the last run,
            # and so the final channel map (saved into new subdirectory
            # to reflect new zmax, logmlim range) must be incremented.
            if np.all(status_done_pre[ip,ichan,:,:]):
                continue

            # See if we already finished this map.
            fn = self.get_map_fn(fov, pix, channel, popid,
                logmlim=mlayer, zlim=zlayer, wave_units=wave_units,
                suffix=suffix,
                include_galaxy_sizes=include_galaxy_sizes)

            pb.update(h)

            if dryrun:
                print(f"# Dry run: would run map {fn}")
                continue

            chan_mic = self.convert_chan_to_micron(channel, wave_units)

            # Will need channel width in Hz to recover specific intensities
            # averaged over band.
            nu = c * 1e4 / np.mean(chan_mic)
            dnu = c * 1e4 * (chan_mic[1] - chan_mic[0]) / np.mean(chan_mic)**2

            # What buffer should we increment?
            if (not keep_layers):
                buffer = cimg
            else:
                buffer = np.zeros([npix]*2)

            ran_new = True
            if os.path.exists(fn) and (not clobber):
                # Load map
                if load_if_found:
                    _buffer, _hdr = self._load_map(fn)

                    # Might need to adjust units before incrementing
                    if _hdr['BUNIT'] == map_units:
                        _buffer *= (f_norm / dnu)**-1.
                    else:
                        raise NotImplemented('help')

                    # Increment map for this z layer
                    cimg += _buffer

                    if verbose:
                        print(f"# Loaded map {fn}.")
                else:
                    print(f"# Elected not to load {fn} since load_if_found=False.")
                    print(f"# Be sure to re-run `generate_maps` once all checkpoints are done with load_if_found=True.")

                ran_new = False
            else:
                if verbose:
                    print(f"# Generating map {fn}...")

                # Make sure user gave us info needed to generate surface
                # brightness profiles. Note that IHL is exempt from this as
                # we only have one option (projected NFW treatment).
                if include_galaxy_sizes and (not self.sim.pops[pid].is_diffuse):
                    assert self.sim.pops[pid].pf['pop_msr'] is not None, \
                        "Must provide `pop_msr` if include_galaxy_sizes=True!"

                # Generate map -> buffer
                # Internal flux units are cgs [erg/s/cm^2/Hz/sr]
                # but get_map returns a channel-integrated flux, erg/s/cm^2/sr
                self.get_map(fov, pix, chan_mic,
                    logmlim=mlayer, zlim=zlayer, popid=popid,
                    wave_units=wave_units,
                    include_galaxy_sizes=include_galaxy_sizes,
                    null_beyond_size=null_beyond_size,
                    size_cut=size_cut,
                    dlam=dlam, use_pbar=False,
                    logmlim_sats=logmlim_sats,
                    buffer=buffer, nthreads=nthreads, verbose=verbose,
                    postage_stamp=postage_stamp,
                    **kwargs)

                status_done_now[ip,ichan,iz,im] = 1

            # Save every mass layer within every redshift layer if the user
            # says so.
            if keep_layers and ran_new:

                if iz in _keep_layers_custom:
                    _fn = self.get_map_fn(fov, pix, channel, popid,
                        logmlim=mlayer, zlim=zlayer, wave_units=wave_units,
                        suffix=suffix,
                        fmt=fmt, include_galaxy_sizes=include_galaxy_sizes)
                    self.save_map(_fn, buffer * f_norm / dnu,
                        channel, zlayer, logmlim, fov,
                        pix=pix, fmt=fmt, hdr=hdr, map_units=map_units,
                        verbose=verbose, clobber=clobber)

            # Increment map for this z layer
            # (a new `cimg` gets created later once full mass range is done)
            #if ran_new:
                cimg += buffer
            #else:
                # Already incremented above after loaded
            #    pass

            ##
            # Otherwise, figure out what (if anything) needs to be
            # written to disk now.
            done_w_chan = np.all(
                status_done_pre[ip,ichan,:,:] +
                status_done_now[ip,ichan,:,:]
                )

            # This probably means our re-run only added channels, not
            # z layers or mass layers.
            was_done_already = np.all(status_done_pre[ip,ichan,:,:] == 1) \
                and (not clobber)

            ##
            # Need to know:
            # Did we do any work to fill out this spectral channel, e.g.,
            # augmenting the redshift or mass range? If so, we need to save
            # a new channel map. If not, we don't need to write anything to
            # disk, but we do need to clear 'cimg' since the next iteration
            # will be a new channel.

            # Also: for mass layers, we might run, e.g., (11,12) in one call,
            # (12,13) next, and then later decide to do (11,13), in which case
            # all the work is done already *except* creating the final
            # channel map. That's why below we'll either write the final map
            # if we can tell the work wasn't done before OR if we can't find
            # an output file.

            # Filename for the final channel map
            # (note use of self.zlim, not zlayer, and logmlim, not mlayer)
            _fn = self.get_map_fn(fov, pix, channel, popid,
                logmlim=logmlim, zlim=self.zlim, fmt=fmt, wave_units=wave_units,
                suffix=suffix,
                include_galaxy_sizes=include_galaxy_sizes)

            _fn_exists = os.path.exists(_fn)

            # If we're done with the channel and population, time to write
            # a final "channel map". Afterward, we'll zero-out `cimg` to be
            # incremented starting on the next iteration.
            if done_w_chan and ((not was_done_already) or (not _fn_exists)) \
                and load_if_found:

                self.save_map(_fn, cimg * f_norm / dnu,
                    channel, self.zlim, logmlim, fov,
                    pix=pix, fmt=fmt, hdr=hdr, map_units=map_units,
                    verbose=verbose, clobber=clobber)

                del cimg, buffer
                if self.mem_concious:
                    gc.collect()

                base_dir = self.get_output_dir(fov, zlim=self.zlim, logmlim=logmlim)

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
                s_ch  = f'{chname}; {np.mean(channel):.6f}; '
                s_ch += f'{channel[0]:.5f}; {channel[1]:.6f}; '
                s_ch += f'{popid}; {_fn} \n'

                ##
                # # Append to README to indicate channel map is complete
                if write_README:
                    with open(f'{base_dir}/README', 'a') as f:
                        f.write(s_ch)
            elif done_w_chan and ((not was_done_already) or (not _fn_exists)):
                print(f"! Done with map {_fn} but did not write because load_if_found=False.")

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

        ##
        # Stitch together z slices?
        self.post_process_z_layers(fov, pix, channels,
            logmlim=logmlim, dlogm=dlogm,
            clobber=clobber, channel_names=channel_names,
            include_pops=include_pops, verbose=verbose,
            map_units=map_units,
            include_galaxy_sizes=include_galaxy_sizes,
            keep_layers=keep_layers,
            keep_layers_custom_z=keep_layers_custom_z,
            keep_chunks=keep_chunks)

        return

    def post_process_z_layers(self, fov, pix, channels, logmlim, dlogm=1,
        clobber=False, include_pops=[0], verbose=True, channel_names=None,
        keep_layers=False, keep_layers_custom_z=None, keep_chunks=None,
        include_galaxy_sizes=False,
        map_units='MJy/sr', hdr={}, fmt='fits'):
        """
        If we decided to save redshift layers, we may still need to sum
        together the individual mass layers.

        .. note :: Generalize this to automatically sum over source populations
            as well?

        """

        if (not keep_layers) and (keep_chunks is None):
            return

        chunks_edges_z, chunks_edges_ids, chunks_zlayers_needed = \
            self._check_chunks(keep_chunks)

        # Full list of map layers to run.
        all_layers = self.get_layers(channels, logmlim, dlogm=dlogm,
            include_pops=include_pops, channel_names=channel_names)

        all_zlayers = np.array(self.get_redshift_layers(self.zlim))
        all_mlayers = np.array(self.get_mass_layers(logmlim, dlogm))

        # User can custom define subset of redshift layers to save
        # (this is a computational choice: saving all can be ~TBs of images)
        if (keep_layers_custom_z == None):
            _keep_layers_custom = list(np.arange(0, len(all_zlayers)))
        else:
            _keep_layers_custom = list(keep_layers_custom_z)

        # A few last things we need
        f_norm = self.get_map_norm(map_units, pix)
        npix = int(fov * 3600 / pix)

        ##
        # loop through redshift layers of interest
        for ichan, channel in enumerate(channels):

            nu = c * 1e4 / np.mean(channel)
            dnu = c * 1e4 * (channel[1] - channel[0]) / np.mean(channel)**2

            for popid in include_pops:

                for iz in _keep_layers_custom:

                    cimg = np.zeros([npix, npix])
                    for im, mlayer in enumerate(all_mlayers):

                        # See if we already finished this map.
                        fn = self.get_map_fn(fov, pix, channel, popid,
                            logmlim=mlayer, zlim=all_zlayers[iz],
                            wave_units=wave_units,
                            include_galaxy_sizes=include_galaxy_sizes)

                        _buffer, _hdr = self._load_map(fn)

                        # Might need to adjust units before incrementing
                        if _hdr['BUNIT'] == map_units:
                            _buffer *= (f_norm / dnu)**-1.
                        else:
                            raise NotImplemented('help')

                        # Increment map for this z layer
                        cimg += _buffer

                    ##
                    # Done with mass slices. Save redshift slice.
                    _fn = self.get_map_fn(fov, pix, channel, popid,
                        logmlim=logmlim, zlim=all_zlayers[iz],
                        wave_units=wave_units,
                        include_galaxy_sizes=include_galaxy_sizes)

                    self.save_map(_fn, cimg * f_norm / dnu,
                        channel, all_zlayers[iz], logmlim, fov,
                        pix=pix, fmt=fmt, hdr=hdr, map_units=map_units,
                        verbose=verbose, clobber=clobber)

                if chunks_edges_ids is None:
                    continue

                ##
                # Now, [optionally] sum over redshift layers to form 'chunks'
                # like "EoR", "cosmic noon", etc.
                for k, chunk_edge_id in enumerate(chunks_edges_ids):

                    cimg = np.zeros([npix, npix])
                    for iz in range(chunk_edge_id[0], chunk_edge_id[1]+1):
                        # Load z layer summed over mass (`logmlim` is whole range)
                        fn = self.get_map_fn(fov, pix, channel, popid,
                            logmlim=logmlim, wave_units=wave_units,
                            zlim=all_zlayers[iz],
                            include_galaxy_sizes=include_galaxy_sizes)

                        _buffer, _hdr = self._load_map(fn)

                        # Might need to adjust units before incrementing
                        if _hdr['BUNIT'] == map_units:
                            _buffer *= (f_norm / dnu)**-1.
                        else:
                            raise NotImplemented('help')

                        # Increment map for this z layer
                        cimg += _buffer

                    ##
                    # Done with mass slices. Save redshift slice.
                    _fn = self.get_map_fn(fov, pix, channel, popid,
                        logmlim=logmlim, zlim=chunks_edges_z[k],
                        wave_units=wave_units,
                        force_chunk=True, include_galaxy_sizes=include_galaxy_sizes)

                    self.save_map(_fn, cimg * f_norm / dnu,
                        channel, chunks_edges_z[k], logmlim, fov,
                        pix=pix, fmt=fmt, hdr=hdr, map_units=map_units,
                        verbose=verbose, clobber=clobber)

    def save_cat(self, fn, cat, channel, zlim, logmlim, fov, fmt='fits',
        hdr={}, clobber=False, verbose=False, cat_units=''):
        """
        Save galaxy catalog.

        Parameters
        ----------
        fn : str
            Output filename.
        cat : np.array
            1-D Array containing the quantity to be saved.
        channel : str
            Name of the field being saved.

        """

        # Should just figure out `fmt` from filename in future
        assert fn.endswith(fmt)

        if os.path.exists(fn) and (not clobber):
            if verbose:
                print(f"# {fn} exists! Set clobber=True to overwrite.")
            return

        if fmt == 'hdf5':
            with h5py.File(fn, 'w') as f:
                #f.create_dataset('ra', data=ra)
                #f.create_dataset('dec', data=dec)
                #f.create_dataset('z', data=red)
                f.create_dataset(channel, data=cat)

                # Save hdr
                grp = f.create_group('hdr')
                for key in hdr:
                    grp.create_dataset(key, data=hdr[key])

        elif fmt == 'fits':
            #col1 = fits.Column(name='ra', format='D', unit='deg', array=ra)
            #col2 = fits.Column(name='dec', format='D', unit='deg', array=dec)
            #col3 = fits.Column(name='z', format='D', unit='', array=red)
            if type(channel) in [list, tuple, np.ndarray]:
                col4 = fits.Column(name='flux', format='D', unit=cat_units,
                    array=np.array(cat, dtype=float))
            else:
                col4 = fits.Column(name=channel, format='D', unit=cat_units,
                    array=np.array(cat, dtype=float))
            coldefs = fits.ColDefs([col4])

            hdu = fits.BinTableHDU.from_columns(coldefs)
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

            t1 = time.time()
            with fits.open(fn) as hdu:
                # In whatever `map_units` user supplied.
                img = hdu[0].data
                hdr = hdu[0].header

            t2 = time.time()
            print(f"! Loaded {fn} [took {(t2-t1):.2f} sec].")

        else:
            raise NotImplementedError(f'No support for fmt={fmt}!')

        return img, hdr

    def _load_cat(self, fn, skip_pos=False):
        """
        Load a catalog from disk.

        Parameters
        ----------
        fn : str
            Filename.
        skip_pos : bool
            If True, will not (re-)load (ra, dec, z) from file. This an be
            advantageous for big catalogs if you already have the galaxy
            positions loaded in memory.

        Returns
        -------
        A tuple containing (ra, dec, redshift, catalog, catalog_units), unless
        skip_pos==True, in which case it will just be (catalog, catalog_units).
        """
        if fn.endswith('hdf5'):
            raise NotImplemented('hdf5 option needs updating')
            with h5py.File(fn, 'r') as f:
                ra = np.array(f[('ra')])
                dec = np.array(f[('dec')])
                red = np.array(f[('z')])
                X = np.array(f[('Mh')])
                Xunit = None
        elif fn.endswith('fits'):

            with fits.open(fn) as f:
                data = f[1].data

            # Determine field name from column header
            name = data.columns[0].name
            X = np.array(data[name], dtype=float)
            Xunit = f[1].header['TUNIT1']

            out = []
            for field in ['ra', 'dec', 'z']:
                if skip_pos or name in ['ra', 'dec', 'z']:
                    break

                with fits.open(fn.replace(name, field)) as f:
                    data = np.array(f[1].data, dtype=float)

                out.append(data)

            out.extend([X, Xunit])

        else:
            raise NotImplemented('Unrecognized file format `{}`'.format(
                fn[fn.rfind('.'):]))

        return tuple(out)

    def read_maps(self, fov, channels, pix=1, logmlim=None, dlogm=0.5,
        prefix=None, suffix=None, save_dir=None, keep_layers=False, fmt='fits'):
        """
        Assemble an array of maps.
        """

        raise NotImplemented('needs fixing')

        if save_dir is None:
            save_dir = '.'

        npix = int(fov * 3600 / pix)
        zlayers = self.get_redshift_layers(self.zlim)
        mlayers = self.get_mass_layers(logmlim, dlogm)

        if keep_layers:
            layers = np.zeros((len(channels), len(zlayers), len(mlayers), npix, npix))
        else:
            layers = np.zeros((len(channels), npix, npix))

        ra_e, ra_c, dec_e, dec_c = self.get_pixels(fov, pix=pix)

        Nloaded = 0
        for i, channel in enumerate(channels):

            for j, (zlo, zhi) in enumerate(zlayers):

                for k, (mlo, mhi) in enumerate(mlayers):

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

        return channels, zlayers, mlayers, ra_c, dec_c, layers
