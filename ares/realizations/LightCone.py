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
from ..util.Stats import bin_e2c, bin_c2e
from scipy.integrate import cumtrapz, quad
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

    def create_directory_structure(self, path='.', suffix=None):
        """
        Our model is:

        ->ebl_fov_<FOV in deg>_pix_<pixel scale in arcsec>_L<box/cMpc/h>_N<dims>/
        ->  README
        ->  <parent directory name>.tar.gz
                *(contains) final mock for each channel summed over all z, M, pops.
        ->  ch_<spectral channel edges>
        ->    z_<redshift lower edge, upper edge>_M_<halo mass lower edge, upper>_pid_<pop ID number>.fits
        ->

        """

        base_dir = '{}/ebl_fov_{:.1f}_pix_{:.1f}'.format(path,
            self.fov, self.pix, self.Lbox)

    def get_base_dir(self, fov, pix, path='.', suffix=None):
        """
        Generate the name for the root directory where all mocks for a given
        model will go.

        Our model is:

        ->ebl_fov_<FOV/deg>_pix_<pixel scale / arcsec>_L<box/cMpc/h>_N<dims>/
        ->  README

        Inside this directory, there will be many subdirectories: one for each
        spectral channel of interest.

        There will also be a series of .fits (or .hdf5) files, which represent
        "final" maps, i.e., those that are summed over redshift and mass chunks,
        and also summed over all source populations.

        """


        s = '{}/ebl_fov_{:.1f}_pix_{:.1f}_L{:.0f}_N{:.0f}'.format(path,
            fov, pix, self.Lbox, self.dims)

        if suffix is None:
            print("# WARNING: might be worth providing `suffix` as additional identifier.")
        else:
            s += f'_{suffix}'

        return s

    def get_map(self, fov, pix, channel, logmlim=None, zlim=None, idnum=0,
        include_galaxy_sizes=False, save_intermediate=True, dlam=20.,
        use_pbar=True, map_units='si', **kwargs):
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
        map_units : str
            Options include "si" (nW/m^2/Hz/sr), 'cgs' (erg/s/cm^2/sr), or
            'Mjy' (MJy/sr)

        """

        #if not hasattr(self, '_cache_maps'):
        #    self._cache_maps = {}

        #kwtup = tuple(kwargs.items())

        #if (channel, zlim, pix, include_galaxy_sizes, kwtup) in self._cache_maps:
        #    return self._cache_maps[(channel, zlim, pix, include_galaxy_sizes, kwtup)]

        pix_deg = pix / 3600.
        sr_per_pix = pix_deg**2 / sqdeg_per_std

        assert fov * 3600 / pix % 1 == 0, "FOV must be integer number of pixels wide!"

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
                logmlim=logmlim, idnum=idnum)

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
            seds = self.sim.pops[idnum].get_spec(_z_, waves, M=Mh,
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

            # May have empty chunks, e.g., very massive halos and/or very
            # high redshifts.
            if not np.any(ok):
                continue

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

        ##
        # Remember: using cgs units internally.
        # 1 Jy = 1e-23 erg/s/cm^2/sr
        if map_units.lower() == 'si':
            final_map = img * cm_per_m**2 / erg_per_s_per_nW
        elif map_units.lower() == 'cgs':
            final_map = img
        elif map_units.lower() == 'mjy':
            final_map = img * 1e17
        else:
            raise ValueErorr(f"Unrecognized option `map_units={map_units}`")

        return ra_e, dec_e, final_map

    def get_fn(self, channel, logmlim, zlim=None, fmt='hdf5', final=False):
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
            try:
                fn += '{:.2f}_{:.2f}'.format(*channel)
            except:
                fn += f'{channel}'

        return fn + '.' + fmt

    def get_README(self, fov, pix, channels, logmlim, zlim, path='.',
        suffix=None, fmt='fits', save=False, is_map=True):
        """

        """

        s = "#" * 78
        s += '\n# README\n'
        s += "#" * 78
        s +=  "\n# This is an automatically-generated file! \n"
        s += "# In it, we list the names of all maps created with a given set"
        s += " of parameters.\n"
        s += "# Note: this is a listing of files that will exist when the "
        s += "map-making is \n# COMPLETE, i.e., they may not all exist yet.\n"
        s += "#" * 78
        s += "\n"
        s += "# central wavelength (microns); channel lower edge (microns) ; "
        s += "channel upper edge (microns) ; filename \n"

        # Loop over channels
        if is_map:
            final_dir = 'final_maps'
            rstr = 'maps'
        else:
            final_dir = 'final_cats'
            rstr = 'cats'

        if channels in [None, 'Mh', ['Mh']]:
            channels = ['Mh']

        # Loop over channels and record filenames. If not supplied, must be
        # galaxy catalog containing only positions (no photometry).
        for channel in channels:
            fn = self.get_fn(channel, logmlim, zlim=zlim,
                fmt=fmt, final=True)

            if is_map:
                s += "{:.4f} ; {:.4f} ; {:.4f} ; {}/{} \n".format(
                    np.mean(channel), channel[0], channel[1], final_dir, fn)
            else:
                s += "{} ; {}/{}".format(channel, final_dir, fn)
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

    def generate_cats(self, fov, pix, channels, logmlim, dlogm=0.5, zlim=None,
        include_galaxy_sizes=False, dlam=20, path='.',
        suffix=None, fmt='hdf5', hdr={},
        include_pops=None, clobber=False, save_catalogs=False, **kwargs):
        """
        Generate galaxy catalogs.
        """

        # Create root directory if it doesn't already exist.
        base_dir = self.get_base_dir(fov=fov, pix=pix, path=path, suffix=suffix)
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        if not os.path.exists(base_dir + '/final_cats'):
            os.mkdir(base_dir + '/intermediate_cats')
            os.mkdir(base_dir + '/final_cats')

        #
        if channels is None:
            run_phot = False
            channels = ['Mh']
        else:
            run_phot = True

        ##
        # Write a README file that says what all the final products are
        README = self.get_README(fov=fov, pix=pix, channels=channels,
            logmlim=logmlim, zlim=zlim, path=path, fmt=fmt,
            suffix=suffix, save=True, is_map=False)

        if zlim is None:
            zlim = self.zlim

        if include_pops is None:
            include_pops = range(0, len(self.sim.pops))

        assert fov * 3600 / pix % 1 == 0, \
            "FOV must be integer number of pixels wide!"

        npix = int(fov * 3600 / pix)
        zchunks = self.get_redshift_chunks(self.zlim)
        mbins = np.arange(logmlim[0], logmlim[1], dlogm)
        mchunks = np.array([(mbin, mbin+dlogm) for mbin in mbins])

        final_dir = base_dir + '/final_cats'
        _final_sub = self.get_fn(None, logmlim, zlim=zlim, fmt=fmt,
            final=True)
        final_sub = _final_sub[0:_final_sub.rfind('/')]
        if not os.path.exists(f'{final_dir}/{final_sub}'):
            os.mkdir(f'{final_dir}/{final_sub}')

        ##
        # Loop over populations, make catalogs.
        for channel in channels:
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
                for (zlo, zhi) in zchunks:

                    # Look for pre-existing file.

                    if (zhi <= zlim[0]) or (zlo >= zlim[1]):
                        continue

                    ra_z = []
                    dec_z = []
                    red_z = []
                    dat_z = []

                    # Go from high to low in mass
                    for (mlo, mhi) in mchunks[-1::-1]:

                        fn = self.get_fn(channel, (mlo, mhi), (zlo, zhi),
                            fmt=fmt, final=False)

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

                        # Check that (RA, DEC) range in galaxy catalog is within the
                        # requested FOV.
                        theta_cat = self.sim.cosm.ComovingLengthToAngle(zhi, 1) \
                            * (self.Lbox / self.sim.cosm.h70) / 60.


                        # Hack out galaxies outside our requested lightcone.
                        ok = np.logical_and(np.abs(ra) < fov / 2.,
                                            np.abs(dec) < fov / 2.)
                        ra = ra[ok==1]
                        dec = dec[ok==1]
                        red = red[ok==1]
                        Mh = Mh[ok==1]

                        self.save_cat(fn, (ra, dec, red, Mh),
                            None, (zlo, zhi), (mlo, mhi),
                            fov, pix=pix, fmt=fmt, hdr=hdr)

                        ra_z.extend(list(ra))
                        dec_z.extend(list(dec))
                        red_z.extend(list(red))


                        if not run_phot:
                            dat_z.extend(list(Mh))
                            continue

                        seds = self.sim.pops[popid].get_spec(_z_, waves, M=Mh,
                            stellar_mass=False)
                        #dat_z.extend(list(Mh))

                    ##
                    # Done with all mass chunks

                    # Save intermediate chunk: all masses, single redshift chunk
                    fnt = self.get_fn(channel, logmlim=logmlim,
                        zlim=(zlo, zhi), fmt=fmt, final=False)

                    fnt = intmd_dir + '/' + fnt

                    self.save_cat(fnt, (ra_z, dec_z, red_z, dat_z),
                        None, (zlo, zhi), (mlo, mhi),
                        fov, pix=pix, fmt=fmt, hdr=hdr)

                    ra_allz.extend(ra_z)
                    dec_allz.extend(dec_z)
                    red_allz.extend(red_z)
                    dat_allz.extend(dat_z)

                    del ra_z, dec_z, red_z, dat_z

                ##
                # Done with all redshift chunks for this population.
                fnp = self.get_fn(channel, logmlim=logmlim,
                    zlim=zlim, fmt=fmt, final=False)

                fnp = intmd_dir + '/' + fnp

                self.save_cat(fnp, (ra_allz, dec_allz, red_allz, dat_allz),
                    None, zlim, logmlim,
                    fov, pix=pix, fmt=fmt, hdr=hdr)

                ra_allp.extend(ra_allz)
                dec_allp.extend(dec_allz)
                red_allp.extend(red_allz)
                dat_allp.extend(dat_allz)

            del ra_allz, dec_allz, red_allz, dat_allz

            ##
            # Combine catalogs over populations?
            fnf = self.get_fn(channel, logmlim=logmlim,
                zlim=zlim, fmt=fmt, final=True)

            fnf = final_dir + '/' + fnf

            print(f"Trying to save final catalog to {fnf}")
            print(f"final_dir={final_dir}, fnf={fnf}, channel={channel}")


            self.save_cat(fnf, (ra_allp, dec_allp, red_allp, dat_allp),
                channel, zlim, logmlim,
                fov, pix=pix, fmt=fmt, hdr=hdr)

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

        print(f"# Wrote {final_dir}/README")


    def generate_maps(self, fov, pix, channels, logmlim, dlogm=0.5, zlim=None,
        include_galaxy_sizes=False, dlam=20, path='.',
        suffix=None, fmt='hdf5', hdr={}, map_units='MJy',
        include_pops=None, clobber=False, save_catalogs=False, **kwargs):
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
            logmlim=logmlim, zlim=zlim, path=path,
            suffix=suffix, save=True)

        if np.array(channels).ndim == 1:
            channels = np.array([channels])

        if zlim is None:
            zlim = self.zlim

        if include_pops is None:
            include_pops = range(0, len(self.sim.pops))

        assert fov * 3600 / pix % 1 == 0, \
            "FOV must be integer number of pixels wide!"

        npix = int(fov * 3600 / pix)
        zchunks = self.get_redshift_chunks(self.zlim)
        mbins = np.arange(logmlim[0], logmlim[1], dlogm)
        mchunks = np.array([(mbin, mbin+dlogm) for mbin in mbins])

        final_dir = base_dir + '/final_maps'
        _final_sub = self.get_fn(None, logmlim, zlim=zlim, fmt=fmt,
            final=True)
        final_sub = _final_sub[0:_final_sub.rfind('/')]
        if not os.path.exists(f'{final_dir}/{final_sub}'):
            os.mkdir(f'{final_dir}/{final_sub}')

        totp = np.zeros([npix]*2)
        for popid, pop in enumerate(self.sim.pops):
            if popid not in include_pops:
                continue

            intmd_dir = base_dir + f'/intermediate_maps/pop_{popid}'
            if not os.path.exists(intmd_dir):
                os.mkdir(intmd_dir)

            for channel in channels:

                tot = np.zeros([npix]*2)

                ##
                # Look for pre-existing file

                # Go from low-z to high-z
                for (zlo, zhi) in zchunks:

                    # Look for pre-existing file.

                    totz = np.zeros([npix]*2)

                    if (zhi <= zlim[0]) or (zlo >= zlim[1]):
                        continue

                    # Go from high to low in mass
                    for (mlo, mhi) in mchunks[-1::-1]:

                        fn = self.get_fn(channel, (mlo, mhi), (zlo, zhi),
                            fmt=fmt, final=False)

                        fn = intmd_dir + '/' + fn

                        # Try to read from disk.
                        if os.path.exists(fn) and (not clobber):
                            print(f"Found {fn}. Set clobber=True to overwrite.")
                            _Inu = self._load_map(fn)
                            totz += _Inu
                            continue

                        print(f"# Will save EBL mock to {fn}.")


                        # Will just be for a single chunk so save_intermediate is
                        # irrelevant.
                        ra_e, dec_e, img = self.get_map(fov, pix, channel,
                            logmlim=(mlo, mhi), zlim=(zlo, zhi),
                            save_intermediate=False,
                            include_galaxy_sizes=False,
                            dlam=dlam, use_pbar=False, map_units=map_units, **kwargs)

                        nu = c * 1e4 / np.mean(channel)
                        dnu = (c * 1e4 / channel[0]) - (c * 1e4 / channel[1])

                        # `img` is a band-integrated power, convert to band-averaged
                        # intensity so that map has units of nW m^-2 Hz^-1 sr^-1.
                        Inu = img[0,:,:] / dnu

                        # Save
                        self.save_map(fn, Inu, channel, (zlo, zhi), (mlo, mhi),
                            fov, pix=pix, fmt=fmt, hdr=hdr, map_units=map_units)

                        totz += Inu

                    ##
                    # Save image summed over mass
                    fnz = self.get_fn(channel, logmlim=logmlim, zlim=(zlo, zhi),
                        fmt=fmt, final=False)
                    fnz = intmd_dir + '/' + fnz

                    # Always overwrite. Not really an issue, if a calculation
                    # terminated early then this file wouldn't have been written.
                    #if not os.path.exists(fnz):
                    self.save_map(fnz, totz, channel, (zlo, zhi), logmlim, fov,
                        pix=pix, fmt=fmt, hdr=hdr, map_units=map_units)

                    # Increment map for this z chunk
                    tot += totz

                ##
                # Save image summed over mass and redshift.
                fnt = self.get_fn(channel, logmlim=logmlim,
                    zlim=zlim, fmt=fmt, final=False)

                fnt = intmd_dir + '/' + fnt

                ##
                # Save image summed over mass
                #if not os.path.exists(fnt):
                self.save_map(fnt, tot, channel, zlim, logmlim, fov,
                    pix=pix, fmt=fmt, hdr=hdr, map_units=map_units)

                ##
                # Increment total image
                totp += tot

        ##
        # Wipe slate clean
        f = open(f"{final_dir}/README", 'w')
        f.write("# map [central wavelength/micron] ; populations included \n")
        f.close()

        ##
        # To finish: sum over populations for each channel.
        # Do we want to sum over redshift for halo mass sub-intervals too?
        for channel in channels:
            totp = np.zeros([npix]*2)

            for popid, pop in enumerate(self.sim.pops):
                if popid not in include_pops:
                    continue

                intmd_dir = base_dir + f'/intermediate_maps/pop_{popid}'

                ##
                # Save image summed over populations.
                fnp = self.get_fn(channel, logmlim=logmlim,
                    zlim=zlim, fmt=fmt, final=False)
                fnp = intmd_dir + '/' + fnp

                _Inu = self._load_map(fnp)
                totp += _Inu

            ##
            # Save final products
            fnp = self.get_fn(channel, logmlim=logmlim,
                zlim=zlim, fmt=fmt, final=True)
            fnp = final_dir + '/' + fnp

            self.save_map(fnp, totp, channel, zlim, logmlim, fov,
                pix=pix, fmt=fmt, hdr=hdr, map_units=map_units)

            # Make a note of which populations are included in the current
            # tally.

            s = "{:.4f} ; ".format(np.mean(channel))
            for pop in include_pops:
                s += f"{pop},"

            s = s.rstrip(',') + '\n'

            with open(f"{final_dir}/README", 'a') as f:
                f.write(s)

            print(f"# Wrote {final_dir}/README")

    def save_cat(self, fn, cat, channel, zlim, logmlim, fov, pix=1, fmt='hdf5',
        hdr={}, clobber=True):
        """
        Save galaxy catalog.
        """
        ra, dec, red, X = cat

        if fmt == 'hdf5':
            with h5py.File(fn, 'w') as f:
                f.create_dataset('ra', data=ra)
                f.create_dataset('dec', data=dec)
                f.create_dataset('z', data=red)
                f.create_dataset(channel if channel is not None else 'Mh',
                    data=X)

                # Save hdr
                grp = f.create_group('hdr')
                for key in hdr:
                    grp.create_dataset(key, data=hdr[key])

        else:
            raise NotImplemented(f'Unrecognized `fmt` option "{fmt}"')

        print(f"# Wrote {fn}.")

    def save_map(self, fn, img, channel, zlim, logmlim, fov, pix=1, fmt='hdf5',
        hdr={}, map_units='MJy', clobber=True):
        """
        Save map to disk.
        """

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
            hdr['ARES'] = get_hash().decode('utf-8')

            hdr.update(hdr)

            # Right now, image is in erg/s/cm^2/Hz
            # Convert to MJy/sr
            # Recall that 1 Jy = 1e-23 erg/s/cm^2/Hz

            hdu = fits.PrimaryHDU(data=img, header=hdr)
            hdul = fits.HDUList([hdu])
            hdul.writeto(fn, overwrite=clobber)
            hdul.close()
        else:
            raise NotImplementedError(f'No support for fmt={fmt}')

        print(f"# Wrote {fn}.")

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
        mbins = np.arange(logmlim[0], logmlim[1], dlogm)
        mchunks = np.array([(mbin, mbin+dlogm) for mbin in mbins])

        if keep_layers:
            layers = np.zeros((len(channels), len(zchunks), len(mbins), npix, npix))
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
