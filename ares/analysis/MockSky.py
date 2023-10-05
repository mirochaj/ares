"""

MockSky.py

Author: Jordan Mirocha
Affiliation: JPL / Caltech
Created on: Thu May 11 13:18:17 PDT 2023

Description:

"""

import os
import h5py
import numpy as np

try:
    from astropy.io import fits
except ImportError:
    pass

class MockSky(object):
    def __init__(self, fov=4, pix=1, logmlim=None, zlim=None,
        base_dir=None, Lbox=512, dims=128, prefix='ares_mock', suffix=None,
        fmt='fits'):
        """
        A set of routines to find appropriate mock maps and catalogs.

        Mocks are saved in directories of the form:

            <prefix>_fov_<fov/deg>_pix_<pixel scale / arcsec>_L<Lbox/cMpc/h>_N<dims>

        Within this directory, there will be a README file with information
        about the final maps, which are stored in the "final_maps"
        subdirectory. By "final maps," we mean those that are the sum total of
        emission over redshift, halo mass, and source populations.

        Intermediate maps, e.g., maps corresponding to emission from only a
        single source population, halo mass range, or redshift range, can be
        found in the "intermediate_maps" subdirectory.

        Parameters
        ----------
        fov : int, float
            Field-of-view, linear dimension [degrees]
        pix ; int, float
            Pixel scale [arcseconds]
        logmlim : tuple
            Bounds on halo masses included when generating the mock,
            in log10(M/Msun), e.g., logmlim=(10, 14) means halos with
            10^10 <= Mh/Msun < 10^14 were included.
        zlim : tuple
            Redshift range of mock, e.g., zlim=(0.2, 3) means the redshift range
            0.2 <= z < 3 was included.
        Lbox : int, float
            Linear dimension of co-eval cubes used to build mocks [cMpc / h].
        dims : int
            Resolution of co-eval cubes in linear dimension, i.e., the total
            number of resolution elements per box is dims^3.
        base_dir : str
            If provided, will override the values of `fov`, `pix`, `Lbox`, and
            `dims`.

        Usage
        -----
        Initialization of this class only requires a few numbers. For example:

            mock = MockSky(fov=4, pix=1, Lbox=512, dims=128,
                logmlim=(10, 14), zlim=(0.2, 3))

            # Find all available final maps, store the central wavelengths
            # in `chan`, channel edges in `chan_e`, filenames in `fn`,
            # and the list of which source populations are included in `pops`.
            chan, chan_e, fn, pops = simpars.get_final_maps()

        To see what intermediate products are available, you can do:

            zchunks, mchunks = mock.get_available_subintervals()
            popids = mock.get_available_pops()

        To then work with intermediate products, you could do something like:

            for i, zchunk in enumerate(zchunks):
                fn = mock.get_filenames(channel=chan[0], popid=0,
                    zlim=zchunk, logmlim=None)

                # Load file, do something cool here.
                # Note that `fn` might have multiple entries, e.g., if you
                # don't supply `popid` or `zlim` it will return all
                # files satisfying the provided criteria.

        """
        self.fov = fov
        self.pix = pix
        self.Lbox = Lbox
        self.dims = dims
        self.suffix = suffix
        self.fmt = fmt
        self.npix = self.fov * 3600 // pix
        self.shape = (self.npix, self.npix)
        self.prefix = prefix

        # Replace FOV, pix if base_dir is supplied
        if base_dir is not None:
            self.base_dir = base_dir

            # Be careful to account for prefix and suffix potentially having
            # underscores in them.
            post = base_dir[base_dir.find('fov'):]
            self.prefix = base_dir[0:base_dir.find('fov')-1]

            try:
                _fov, fov, _pix, pix, _L, _N = post.split('_')
            except ValueError:
                iN = post.find('_N')
                _suffix = post[iN+1:]
                i_ = _suffix.find('_')
                _fov, fov, _pix, pix, _L, _N = post[0:iN+i_+1].split('_')
                self.suffix = _suffix[i_+1:]

            self.fov = float(fov)
            self.pix = float(pix)
            self.Lbox = float(_L[1:])
            self.dims = int(_N[1:])


        else:
            self.base_dir = '{}_fov_{:.1f}_pix_{:.1f}_L{:.0f}_N{:.0f}'.format(
                self.prefix, self.fov, self.pix, self.Lbox, self.dims)
            if suffix is not None:
                self.base_dir += f'_{suffix}'

        # These need to be determined from file contents
        if zlim is None:
            zchunks, mchunks = self.get_available_subintervals()
            self.zlim = (zchunks.min(), zchunks.max())
        else:
            self.zlim = zlim

        if logmlim is None:
            zchunks, mchunks = self.get_available_subintervals()
            self.logmlim = (mchunks.min(), mchunks.max())
        else:
            self.logmlim = logmlim

    def get_pixels(self):
        """
        Returns the pixel edges and centers in both RA and DEC.

        .. note :: These are RELATIVE to the center of the map, which
            is specified in the map headers via `CRVAL1` and `CRVAL2`.

        """
        fov = self.fov
        pix = self.pix

        if type(fov) in [int, float, np.float64]:
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

    def get_available_subintervals(self):
        """
        Return a list of redshift and halo mass 'chunks' that we have saved as
        intermediate data products.

        .. note :: This is done for computational reasons in the map-building,
            but could also be a useful check that different contributions to the
            maps can be isolated.

        Returns
        -------
        A tuple containing (redshift chunks, halo mass chunks), where chunk is
        itself a two-element tuple containing the (lower bound, upper bound).
        The mass chunks are in log10(Mhalo/Msun).

        """

        pops = self.get_available_pops()

        _subdir = f"{self.base_dir}/intermediate_maps/pop_{pops[0]}"

        zchunks = []
        mchunks = []
        for fn in os.listdir(_subdir):

            # May save channel edges in filename or with channel name
            try:
                z, zlo, zhi, m, mlo, mhi, chlo, chi = fn.split('_')
                chi = chi[0:chi.rfind('.')]
            except ValueError:
                z, zlo, zhi, m, mlo, mhi, chname = fn.split('_')

            zchunk = (float(zlo), float(zhi))
            mchunk = (float(mlo), float(mhi))

            if zchunk not in zchunks:
                zchunks.append(zchunk)
            if mchunk not in mchunks:
                mchunks.append(mchunk)

        # Sort
        _zlo, _zhi = zip(*zchunks)
        zsort = np.argsort(_zlo)
        _mlo, _mhi = zip(*mchunks)
        msort = np.argsort(_mlo)

        return np.array(zchunks)[zsort], np.array(mchunks)[msort]

    def get_available_pops(self):
        """
        Returns list of source populations for which (at least some) data
        products exist.

        Natively, maps are made one source population at a time. Each population
        (e.g., star-forming centrals, quiescent centrals, IHL, dwarfs, etc) is
        given an ID number.
        """
        _subdirs = os.listdir(f"{self.base_dir}/intermediate_maps")

        # Check to see if there are any output files yet.
        subdirs = []
        for _subdir in _subdirs:
            cand = os.listdir(f"{self.base_dir}/intermediate_maps/{_subdir}")
            if len(cand) == 0:
                continue

            subdirs.append(_subdir)

        pops = [int(subdir.split('_')[-1]) for subdir in subdirs]
        return np.sort(pops)

    def get_available_channels(self):
        """
        Returns tuple containing (central wavelengths of channels, edges).
        """
        chan_c = np.loadtxt(f'{self.base_dir}/README', unpack=True,
            delimiter=';', dtype=float, usecols=[0])
        chan_e = np.loadtxt(f'{self.base_dir}/README', unpack=True,
            delimiter=';', dtype=float, usecols=[1,2])

        return chan_c, chan_e.T

    def get_filenames(self, channel=None, popid=None, zlim=None, logmlim=None):
        """
        Return list of files meeting criteria set by user input. Could be final
        files or intermediate products, e.g., if `popid` is supplied or if
        the `zlim` or `logmlim` ranges supplied are sub-intervals.

        If you're not sure what intermediate products are available, check
        the output of the `get_available_*` routines.

        Parameters
        ----------
        channel : int, float, np.ndarray
            Can be central wavelength for channel of interest or 2-element
            array containing the channel edges [microns].
        popid : int
            ID number for source population of interest. If you're not sure
            what's available, check `get_available_pops`.

        Returns
        -------
        Return filename or list of filenames matching user selection criteria.
        If user supplies nothing, will just return filenames of final maps,
        which are listed in the `self.base_dir/README`.

        """

        chan_n, chan_c, chan_e, fn_fin, pops = self.get_final_maps()

        if zlim is None:
            zlim = self.zlim
        if logmlim is None:
            logmlim = self.logmlim

        if channel is not None:
            if type(channel) in [int, float, np.float64]:
                k = np.argmin(np.abs(channel - chan_c))
            else:
                k = np.argmin(np.abs(np.mean(channel) - chan_c))

            if chan_c[k] != channel:
                print(f"# WARNING: closest channel to requested: {chan_c[k]}")

            ch_n = [chan_n[k]]
            ch_e = [chan_e[k]]
            ch_c = [chan_c[k]]
            fn = fn_fin[k]
        else:
            ch_n = chan_n
            ch_e = chan_e
            ch_c = chan_c
            fn = fn_fin

        # Return final files if no user input.
        if (zlim == self.zlim) and (logmlim == self.logmlim):
            if (popid is None) or (np.unique(pops).size == 1):
                # In this case, it's just a final map we're interested in.
                return ch_n, ch_c, ch_e, [fn]

        ##
        # If we made it here, we're interested in some intermediate data product.
        _subdir = f"{self.base_dir}/intermediate_maps"

        if popid is None:
            popids = self.get_available_pops()
        else:
            popids = [popid]

        all_fn = []
        for _popid in popids:
            subdir = f'{_subdir}/pop_{_popid}'
            fn = 'z_{:.2f}_{:.2f}'.format(*list(zlim))
            fn += '_M_{:.1f}_{:.1f}'.format(*list(logmlim))

            for j, _chan in enumerate(ch_e):
                if ch_n[j] is not None:
                    _fn = fn + '_{}'.format(ch_n[j])
                else:
                    _fn = fn + '_{:.2f}_{:.2f}'.format(*_chan)
                all_fn.append(f"{subdir}/{_fn}.{self.fmt}")

        return ch_n, ch_c, ch_e, all_fn

    def get_final_maps(self):
        """
        Reads README file to obtain listing of files corresponding to final maps.

        .. note :: "Final" maps are those that contain the total summed
            contributions over multiple source populations, redshifts, and
            halo masses.

        Returns
        -------
        Tuple containing (channels [central wavelength / microns],
            channel edges [microns], filenames, list of included source pops).

        """
        chan_n = np.loadtxt(f'{self.base_dir}/README_maps', unpack=True,
            delimiter=';', dtype=str, usecols=[0],
            converters=lambda s: s.strip(), ndmin=1)
        chan_c = np.loadtxt(f'{self.base_dir}/README_maps', unpack=True,
            delimiter=';', dtype=float, usecols=[1], ndmin=1)
        chan_e = np.loadtxt(f'{self.base_dir}/README_maps', unpack=True,
            delimiter=';', dtype=float, usecols=[2,3], ndmin=2)
        _fn = np.loadtxt(f'{self.base_dir}/README_maps', unpack=True,
            delimiter=';', dtype=str, usecols=[4],
            converters=lambda s: s.strip(), ndmin=1)

        # Account for case with only one output file
        fn = [self.base_dir+'/'+_fn_.strip() for _fn_ in _fn]

        # Read in which populations are included so far.

        if os.path.exists(f"{self.base_dir}/final_maps/README"):
            _chan = np.loadtxt(f'{self.base_dir}/final_maps/README',
                unpack=True, delimiter=';', dtype=float, usecols=[0], ndmin=1)
            _pops = np.loadtxt(f'{self.base_dir}/final_maps/README',
                unpack=True, delimiter=';', dtype=str, usecols=[1],
                converters=lambda s: s.strip(), ndmin=1)

            pops = [None for ch in chan_c]
            for j, _ch_ in enumerate(_chan):
                k = np.argmin(np.abs(_ch_ - chan_c))
                pops[k] = [int(pop) for pop in _pops[j].split(',')]
        else:
            # Shouldn't happen anymore.
            pops = None

        return chan_n, chan_c, chan_e.T, fn, pops

    def get_final_cats(self):
        """
        Reads README file to obtain listing of files corresponding to final
        catalogs.

        .. note :: "Final" catalogs are those that contain the full galaxy
            population over multiple source populations, redshifts, and
            halo masses.

        Returns
        -------
        Tuple containing (channels [central wavelength / microns],
            channel edges [microns], filenames, list of included source pops).

        """
        filt = np.loadtxt(f'{self.base_dir}/README_cats', unpack=True,
            delimiter=';', dtype=str, usecols=[0],
            converters=lambda s: s.strip(), ndmin=1)
        _fn = np.loadtxt(f'{self.base_dir}/README_cats', unpack=True,
            delimiter=';', dtype=str, usecols=[1],
            converters=lambda s: s.strip(), ndmin=1)

        # Account for case with only one output file
        fn = [self.base_dir+'/'+_fn_.strip() for _fn_ in _fn]

        # Read in which populations are included so far.

        if os.path.exists(f"{self.base_dir}/final_cats/README"):
            _chan = np.loadtxt(f'{self.base_dir}/final_cats/README',
                unpack=True, delimiter=';', dtype=str, usecols=[0],
                converters=lambda s: s.strip(), ndmin=1)
            _pops = np.loadtxt(f'{self.base_dir}/final_cats/README',
                unpack=True, delimiter=';', dtype=str, usecols=[1],
                converters=lambda s: s.strip(), ndmin=1)

            pops = [None for ch in filt]
            for j, _ch_ in enumerate(_chan):
                k = list(filt).index(_ch_)
                pops[k] = [int(pop) for pop in _pops[j].split(',')]
        else:
            # Shouldn't happen anymore.
            pops = None

        return filt, fn, pops

    def load_map(self, fn=None, channel=None, popid=None, zlim=None,
        logmlim=None, as_fits=False):
        """

        """

        if fn is None:
            assert channel is not None, "Must provide channel if not `fn`."
            ch_n, ch_c, ch_e, all_fn = self.get_filenames(channel=channel,
                popid=popid, zlim=zlim, logmlim=logmlim)

            if len(all_fn) > 1:
                raise ValueError(f"File selection criteria not unique! all_fn={all_fn}")

            fn = all_fn[0]

        if as_fits:
            return fits.open(fn)

        try:

            with fits.open(fn) as hdu:
                # In whatever `map_units` user supplied.
                img = hdu[0].data
                hdr = hdu[0].header
        except FileNotFoundError:
            img = hdr = None
            print(f"# No file {fn} found.")

        return img, hdr

    def load_cat(self, fn):
        if fn.endswith('hdf5'):
            with h5py.File(fn, 'r') as f:
                ra = np.array(f[('ra')])
                dec = np.array(f[('dec')])
                red = np.array(f[('z')])
                X = np.array(f[('Mh')])
        elif fn.endswith('fits'):
            with fits.open(fn) as f:
                data = f[1].data
                ra = data['ra']
                dec = data['dec']
                red = data['z']

                # Hack for now.
                name = data.columns[3].name
                X = data[name]
        else:
            raise NotImplemented('Unrecognized file format `{}`'.format(
                fn[fn.rfind('.'):]))

        return ra, dec, red, X

    def get_galaxy_map(self, z, dz=0.1, maglim=None, magfilt='Mh', pops=None):
        """
        Return an image containing the number of galaxies in each pixel at
        the specified redshift, (z - dz/2, z+dz/2), with optional magnitude
        cut [not yet implemented].
        """

        filt, fn_cat, pops = self.get_final_cats()

        k = list(filt).index(magfilt)

        _ra, _dec, red, X = self.load_cat(fn_cat[k])

        ok = np.logical_and(red >= z-0.5*dz, red < z+0.5*dz)

        if maglim is not None:
            ok = np.logical_and(ok, X < maglim)

        # Grab info about available maps just so we know what the image
        # dimensions should be and how to map RA/DEC to pixels.
        ch_n, ch_c, ch_e, all_fn, pops_maps = self.get_final_maps()

        _img_, _hdr_ = self.load_map(channel=ch_c[0])

        #ra_0 = _hdr_['CRVAL1']
        #dec_0 = _hdr_['CRVAL2']
        #x0, y0 = _hdr_['CRPIX1'], _hdr_['CRPIX2']

        #w = WCS(_hdr_)
        #w.pixel_to_world(30, 40)

        ra_e, ra_c, dec_e, dec_c = self.get_pixels()

        ra = _ra #+ ra_0
        dec = _dec #+ dec_0

        i = np.digitize(ra[ok==1], ra_e) - 1
        j = np.digitize(dec[ok==1], dec_e) - 1

        img = np.zeros(_img_.shape, dtype=int)
        for k in range(ok.sum()):
            img[i[k],j[k]] += 1

        return img
