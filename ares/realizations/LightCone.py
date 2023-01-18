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
import numpy as np
from scipy.integrate import cumtrapz
from ..util.Stats import bin_e2c, bin_c2e
from ..util.ProgressBar import ProgressBar
from ..util.Misc import numeric_types, get_hash
from ..physics.Constants import sqdeg_per_std, cm_per_mpc, cm_per_m, \
    erg_per_s_per_nW, c

try:
    import h5py
except ImportError:
    pass

class LightCone(object): # pragma: no cover
    """
    This should be inherited by the other classes in this submodule.
    """

    def get_pixels(self, fov, pix=1):
        # Figure out the edges of the domain in RA and DEC (degrees)
        ra0, ra1 = fov * 0.5 * np.array([-1, 1])
        dec0, dec1 = fov * 0.5 * np.array([-1, 1])

        pix_deg = pix / 3600.

        # Pixel coordinates
        ra_e = np.arange(ra0, ra1 + pix_deg, pix_deg)
        ra_c = ra_e[0:-1] + 0.5 * pix_deg
        dec_e = np.arange(dec0, dec1 + pix_deg, pix_deg)
        dec_c = dec_e[0:-1] + 0.5 * pix_deg

        return ra_e, ra_c, dec_e, dec_c

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
            self._tab_dL = np.array([self.sim.cosm.LuminosityDistance(z) \
                for z in self.tab_z]) / cm_per_mpc
        return self._tab_dL

    def get_map(self, fov, channel, zlim=None, logmlim=None, pix=1, idnum=0,
        include_galaxy_sizes=False, save_intermediate=True, dlam=20.,
        use_pbar=True, **kwargs):
        """
        Get a map in the user-supplied spectral channel.

        Parameters
        ----------
        fov : int, float
            Field of view (single dimension) in degrees.
        channel : tuple, list, np.ndarray
            Edges of the spectral channel of interest [microns].
        zlim : tuple, list, np.ndarray
            Optional redshift range. If None, will include all objects in the
            catalog.
        pix : int, float
            Pixel scale in arcseconds.

        """

        #if not hasattr(self, '_cache_maps'):
        #    self._cache_maps = {}

        #kwtup = tuple(kwargs.items())

        #if (channel, zlim, pix, include_galaxy_sizes, kwtup) in self._cache_maps:
        #    return self._cache_maps[(channel, zlim, pix, include_galaxy_sizes, kwtup)]

        pix_deg = pix / 3600.
        sr_per_pix = pix_deg**2 / sqdeg_per_std

        # In degrees
        if type(fov) in numeric_types:
            fov = np.array([fov]*2)

        assert np.diff(fov) == 0, "Only square FOVs allowed right now."

        zall = self.get_redshift_chunks(zlim=self.zlim)

        if zlim is None:
            zlim = self.zlim
        if logmlim is None:
            logmlim = -np.inf, 16

        # Figure out the edges of the domain in RA and DEC (degrees)
        # Pixel coordinates
        ra_e, ra_c, dec_e, dec_c = self.get_pixels(fov, pix=pix)

        Npix = [ra_c.size, dec_c.size]

        # Initialize empty map
        if save_intermediate:
            img = np.zeros([len(zall)] + Npix, dtype=np.float64)
        else:
            img = np.zeros([1] + Npix, dtype=np.float64)

        ##
        # Might take awhile.
        pb = ProgressBar(len(zall),
            name="img(z; Mh>={:.1f}, Mh<{:.1f})".format(logmlim[0], logmlim[1]),
            use=use_pbar)
        pb.start()

        ##
        # Loop over redshift chunks and assemble image.
        for _iz_, (zlo, zhi) in enumerate(zall):

            if (zhi <= zlim[0]) or (zlo >= zlim[1]):
                continue

            _z_ = np.mean([zlo, zhi])

            if save_intermediate:
                iz = _iz_
            else:
                iz = 0

            #ra = self.get_field(z=_z_, name='ra', isolate_chunk=True,
            #    **kwargs)
            #dec = self.get_field(z=_z_, name='dec', isolate_chunk=True,
            #    **kwargs)
            #red = self.get_field(z=_z_, name='redshift', isolate_chunk=True,
            #    **kwargs)
            #Mh = self.get_field(z=_z_, name='mass', isolate_chunk=True,
            #    **kwargs)

            ra, dec, red, Mh = self.get_catalog(zlim=(zlo, zhi),
                logmlim=logmlim)

            # Could be empty chunks for very massive halos and/or early times.
            if ra is None:
                continue

            # Check that (RA, DEC) range in galaxy catalog is within the
            # requested FOV.
            theta_cat = self.sim.cosm.ComovingLengthToAngle(zhi, 1) \
                * (self.Lbox / self.sim.cosm.h70) / 60.

            assert theta_cat >= fov[0], \
                f"Catalog FoV ({theta_cat:.2f}) smaller than requested FoV at z={zhi:.2f}!"

            #dra = ra.max() - ra.min()
            #dde = dec.max() - dec.min()
            #assert dra >= fov[0], \
            #    f"Catalog spans RA range ({dra}) smaller than requested FoV!"
            #assert dde >= dec[0], \
            #    f"Catalog spans DEC range ({dde}) smaller than requested FoV!"

            # Get luminosity distance to each object.
            d_L = np.interp(red, self.tab_z, self.tab_dL) * cm_per_mpc
            corr = 1. / 4. / np.pi / d_L**2

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
            seds = self.sim.pops[idnum].get_sed(_z_, Mh, waves,
                stellar_mass=False)
            #waves, seds = self.get_seds(z=_z_)
            owaves = waves[None,:] * (1. + red[:,None])

            # Frequency "squashing", i.e., our 'per Angstrom' interval is
            # different in the observer frame by a factor of 1+z.
            flux = corr[:,None] * seds[:,:] / (1. + red[:,None]) / sr_per_pix

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
            if zlim is not None:
                okz = np.logical_and(red >= zlim[0], red < zlim[1])
                ok = np.logical_and(okp, okz)
            else:
                ok = okp

            #if self.verbose:
            #    print("Masked fraction: {:.5f}".format((ok.size - ok.sum()) / float(ok.size)))

            ##
            # Need some extra info to do more sophisticated modeling...
            if include_galaxy_sizes:

                # Should have routine built into `cat` that returns
                # structural parameters.
                # For Sersic option, returns Re, n, PA.
                # Potentially other options in the future.

                ##
                # NEED TO GENERALIZE THIS
                Rkpc = self.cat.get_field(_z_, 'Re_maj', isolate_chunk=True,
                    **kwargs)
                nsers = self.cat.get_field(_z_, 'sersic_n', isolate_chunk=True,
                    **kwargs)
                pa = self.cat.get_field(_z_, 'position_angle', isolate_chunk=True,
                    **kwargs)
                R_sec = Rkpc * self.cosmo.arcsec_per_kpc_proper(red).to_value()

                # Size in degrees
                R_deg = R_sec / 3600.
                R_pix = R_deg / pix_deg

                # All in degrees
                x0, y0 = ra, dec
                a, b = R_deg, R_deg

                rr, dd = np.meshgrid(ra_c / pix_deg, dec_c / pix_deg)



            # May have empty chunks, e.g., very massive halos and/or very
            # high redshifts.
            if not np.any(ok):
                continue

            ##
            # Actually sum fluxes from all objects in image plane.
            for h in range(ra.size):

                if not ok[h]:
                    continue

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
                if include_galaxy_sizes and R_pix[h] > 0.5:

                    model_SB = Sersic2D(amplitude=1., r_eff=R_pix[h],
                        x_0=ra[h] / pix_deg, y_0=dec[h] / pix_deg,
                        n=nsers[h], theta=pa[h] * np.pi / 180.)

                    # Fractional contribution to total flux
                    I = model_SB(rr, dd)
                    tot = I.sum()

                    if tot == 0:
                        img[iz,i,j] += _flux_
                    else:
                        img[iz,:,:] += _flux_ * I / tot

                ##
                # Otherwise just add flux to single pixel
                else:
                    img[iz,i,j] += _flux_

            pb.update(_iz_)

            ##
            # Clear out some memory sheesh
            del seds, flux, _flux_, ra, dec, red, Mh
            gc.collect()

        pb.finish()

        ##
        # Hmmm
        if np.any(np.isnan(img)):
            print("* WARNING: {:.4f}% of pixels are NaN! Removing...".format(
                100 * np.isnan(img).sum() / float(img.size)
            ))
            img[np.isnan(img)] = 0.0

        #self._cache_maps[(channel, zlim, pix, include_galaxy_sizes, kwtup)] = \
        #    ra_e, dec_e, img * cm_per_m**2 / erg_per_s_per_nW

        return ra_e, dec_e, img * cm_per_m**2 / erg_per_s_per_nW

    def get_fn(self, fov, channel, pix=1, zlim=None, logmlim=None,
        prefix=None, suffix=None, fmt='hdf5'):

        if zlim is None:
            zlim = self.zlim

        if logmlim is None:
            logmlim = -np.inf, 16

        pref = 'fov_{:.2f}_pix_{:.1f}'.format(fov, pix)
        pref += '_L{:.0f}_N{:.0f}'.format(self.Lbox, self.dims)
        if zlim is not None:
            pref += '_z_{:.2f}_{:.2f}'.format(*zlim)
        if logmlim is not None:
            pref += '_M_{:.1f}_{:.1f}'.format(*logmlim)
        pref += '_ch_{:.2f}_{:.2f}'.format(*channel)

        if prefix is None:
            prefix = ''
        else:
            prefix = f'{prefix}_'

        if suffix is None:
            suffix = ''
        else:
            suffix = f'_{suffix}'

        return prefix + pref + suffix + '.' + fmt

    def generate_maps(self, fov, channels, logmlim, dlogm=0.5, zlim=None, pix=1,
        include_galaxy_sizes=False, dlam=20, clobber=False,
        save_dir=None, prefix=None, suffix=None, fmt='hdf5', hdr={},
        **kwargs):
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

        if np.array(channels).ndim == 1:
            channels = np.array([channels])

        if zlim is None:
            zlim = self.zlim

        if save_dir is None:
            save_dir = '.'
        else:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

        npix = int(fov / (pix / 3600.))
        zchunks = self.get_redshift_chunks(self.zlim)
        mbins = np.arange(logmlim[0], logmlim[1], dlogm)
        mchunks = np.array([(mbin, mbin+dlogm) for mbin in mbins])

        for channel in channels:

            tot = np.zeros([npix]*2)

            for (zlo, zhi) in zchunks:

                totz = np.zeros([npix]*2)

                if (zhi <= zlim[0]) or (zlo >= zlim[1]):
                    continue

                for (mlo, mhi) in mchunks:

                    fn = self.get_fn(fov, channel, pix=pix, zlim=(zlo, zhi),
                        logmlim=(mlo, mhi), prefix=prefix, suffix=suffix, fmt=fmt)

                    fn = save_dir + '/' + fn

                    # Try to read from disk.
                    if os.path.exists(fn) and (not clobber):
                        print(f"Loaded {fn}")
                        continue

                    print(f"# Will save EBL mock to {fn}.")

                    # Will just be for a single chunk so save_intermediate is
                    # irrelevant.
                    ra_e, dec_e, img = self.get_map(fov, channel, zlim=(zlo, zhi),
                        logmlim=(mlo, mhi),
                        pix=pix, save_intermediate=False,
                        include_galaxy_sizes=False,
                        dlam=dlam, use_pbar=False, **kwargs)

                    nu = c * 1e4 / np.mean(channel)
                    dnu = (c * 1e4 / channel[0]) - (c * 1e4 / channel[1])

                    # `img` is a band-integrated power, convert to band-averaged
                    # intensity so that map has units of nW m^-2 Hz^-1 sr^-1.
                    Inu = img[0,:,:] / dnu

                    # Save
                    self.save_map(fn, Inu, channel, (zlo, zhi), (mlo, mhi), fov,
                        pix=pix, fmt=fmt, hdr=hdr)

                    totz += Inu

                ##
                # Save image summed over mass
                fnz = self.get_fn(fov, channel, pix=pix, zlim=(zlo, zhi),
                    logmlim=logmlim, prefix=prefix, suffix=suffix, fmt=fmt)
                fnz = save_dir + '/' + fnz

                self.save_map(fnz, totz, channel, (zlo, zhi), logmlim, fov,
                    pix=pix, fmt=fmt, hdr=hdr)

                # Increment map for this z chunk
                tot += totz

            ##
            # Save image summed over mass and redshift.
            fnt = self.get_fn(fov, channel, pix=pix,
                zlim=(self.zlim[0], self.zlim[1]),
                logmlim=logmlim, prefix=prefix, suffix=suffix, fmt=fmt)

            fnt = save_dir + '/' + fnt

            ##
            # Save image summed over mass
            self.save_map(fnt, tot, channel, self.zlim, logmlim, fov,
                pix=pix, fmt=fmt, hdr=hdr)



    def save_map(self, fn, img, channel, zlim, logmlim, fov, pix=1, fmt='hdf5',
        hdr={}):
        """
        Save map to disk.
        """

        ra_e, ra_c, dec_e, dec_c = self.get_pixels(fov, pix=pix)

        nu = c * 1e4 / np.mean(channel)

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

            # Save as MJy/sr in this case.

            hdr = fits.Header(hdr)
            #_hdr.update(hdr)
            #hdr = _hdr
            hdr['DATE'] = time.ctime()

            hdr['NAXIS'] = 2
            hdr['BUNIT'] = 'MJy/sr'
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
            hdr['ARES'] = get_hash().decode('utf-8')

            hdr.update(hdr)

            hdu = fits.PrimaryHDU(data=img/1e-17, header=hdr)
            hdul = fits.HDUList([hdu])
            hdul.writeto(fn)
        else:
            raise NotImplementedError(f'No support for fmt={fmt}')

        print(f"# Wrote {fn}.")

    def read_maps(self, fov, channels, pix=1, logmlim=None, dlogm=0.5,
        prefix=None, suffix=None, save_dir=None, fmt='hdf5'):
        """
        Assemble an array of maps.
        """

        if save_dir is None:
            save_dir = '.'

        npix = int(fov / (pix / 3600.))
        zchunks = self.get_redshift_chunks(self.zlim)
        mbins = np.arange(logmlim[0], logmlim[1], dlogm)
        mchunks = np.array([(mbin, mbin+dlogm) for mbin in mbins])

        layers = np.zeros((len(channels), len(zchunks), len(mbins), npix, npix))

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

                    ##
                    # Read!
                    if fmt == 'hdf5':
                        with h5py.File(fn, 'r') as f:
                            layers[i,j,k,:,:] = np.array(f[('ebl')])
                    elif fmt == 'fits':
                        from astropy.io import fits
                        with fits.open(fn) as hdu:
                            # Convert back to cgs
                            layers[i,j,k,:,:] = hdu[0].data * 1e-17
                    else:
                        raise NotImplementedError(f'No support for fmt={fmt}!')

                    print(f"# Loaded {fn}.")
                    Nloaded += 1

        if Nloaded == 0:
            raise IOError("Did not find any files! Are prefix, suffix, and save_dir set appropriately?")

        return channels, zchunks, mchunks, ra_c, dec_c, layers
