"""

SpectralSynthesis.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Sat 25 May 2019 09:58:14 EDT

Description:

"""

import time
import numpy as np
from ..obs import Survey
from ..util import ProgressBar
from ..obs import Madau1995
from ..util import ParameterFile
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from ..physics.Cosmology import Cosmology
from scipy.interpolate import RectBivariateSpline
from ..physics.Constants import s_per_myr, c, h_p, erg_per_ev, flux_AB, \
    lam_LL, lam_LyA

nanoJ = 1e-23 * 1e-9

tiny_lum = 1e-8
all_cameras = ['wfc', 'wfc3', 'nircam', 'roman', 'irac']

def _powlaw(x, p0, p1):
    return p0 * (x / 1.)**p1

class SpectralSynthesis(object):
    def __init__(self, **kwargs):
        self.pf = ParameterFile(**kwargs)

    @property
    def cosm(self):
        if not hasattr(self, '_cosm'):
            self._cosm = Cosmology(pf=self.pf, **self.pf)
        return self._cosm

    @property
    def src(self):
        return self._src

    @src.setter
    def src(self, value):
        self._src = value

    @property
    def oversampling_enabled(self):
        if not hasattr(self, '_oversampling_enabled'):
            self._oversampling_enabled = True
        return self._oversampling_enabled

    @oversampling_enabled.setter
    def oversampling_enabled(self, value):
        self._oversampling_enabled = value

    @property
    def oversampling_below(self):
        if not hasattr(self, '_oversampling_below'):
            self._oversampling_below = 30.
        return self._oversampling_below

    @oversampling_below.setter
    def oversampling_below(self, value):
        self._oversampling_below = value

    @property
    def force_perfect(self):
        if not hasattr(self, '_force_perfect'):
            self._force_perfect = False
        return self._force_perfect

    @force_perfect.setter
    def force_perfect(self, value):
        self._force_perfect = value

    @property
    def careful_cache(self):
        if not hasattr(self, '_careful_cache_'):
            self._careful_cache_ = True
        return self._careful_cache_

    @careful_cache.setter
    def careful_cache(self, value):
        self._careful_cache_ = value

    @property
    def cameras(self):
        if not hasattr(self, '_cameras'):
            self._cameras = {}
            for cam in all_cameras:
                self._cameras[cam] = Survey(cam=cam,
                    force_perfect=self.force_perfect,
                    cache=self.pf['pop_synth_cache_phot'])

        return self._cameras

    @property
    def hydr(self):
        if not hasattr(self, '_hydr'):
            from ..physics.Hydrogen import Hydrogen
            self._hydr = Hydrogen(pf=self.pf, cosm=self.cosm, **self.pf)
        return self._hydr

    @property
    def madau1995(self):
        if not hasattr(self, '_madau1995'):
            self._madau1995 = Madau1995(hydr=self.hydr, cosm=self.cosm,
                **self.pf)
        return self._madau1995

    def OpticalDepth(self, z, owaves):
        """
        Compute Lyman series line blanketing following Madau (1995).

        Parameters
        ----------
        zobs : int, float
            Redshift of object.
        owaves : np.ndarray
            Observed wavelengths in microns.

        """

        if self.pf['tau_clumpy'] is None:
            return 0.0

        assert self.pf['tau_clumpy'] in ['madau1995', 1, True, 2], \
            "tau_clumpy in [1,2,'madau1995'] are currently the sole options!"

        tau = np.zeros_like(owaves)
        rwaves = owaves * 1e4 / (1. + z)

        # Scorched earth option: null all flux at < 912 Angstrom
        if self.pf['tau_clumpy'] == 1:
            tau[rwaves < lam_LL] = np.inf
        # Or all wavelengths < 1216 A (rest)
        elif self.pf['tau_clumpy'] == 2:
            tau[rwaves < lam_LyA] = np.inf
        else:
            tau = self.madau1995(z, owaves)

        return tau

    def L_of_Z_t(self, wave):

        if not hasattr(self, '_L_of_Z_t'):
            self._L_of_Z_t = {}

        if wave in self._L_of_Z_t:
            return self._L_of_Z_t[wave]

        tarr = self.src.times
        Zarr = np.sort(list(self.src.metallicities.values()))
        L = np.zeros((tarr.size, Zarr.size))
        for j, Z in enumerate(Zarr):
            L[:,j] = self.src.L_per_sfr_of_t(wave, Z=Z)

        # Interpolant
        self._L_of_Z_t[wave] = RectBivariateSpline(np.log10(tarr),
            np.log10(Zarr), np.log10(L), kx=3, ky=3)

        return self._L_of_Z_t[wave]

    def Slope(self, zobs=None, tobs=None, spec=None, waves=None,
        sfh=None, zarr=None, tarr=None, hist={}, idnum=None,
        cam=None, rest_wave=None, band=None,
        return_norm=False, filters=None, filter_set=None, dlam=20.,
        method='linear', window=1, extras={}, picky=False, return_err=False):
        """
        Compute slope in some wavelength range or using photometry.

        Parameters
        ----------
        zobs : int, float
            Redshift of observation.
        rest_wave: tuple
            Rest-wavelength range in which slope will be computed (Angstrom).
        dlam : int
            Sample the spectrum with this wavelength resolution (Angstrom).
        window : int
            Can optionally operate on a smoothed version of the spectrum,
            obtained by convolving with a boxcar window function if this width.
        """

        assert (tobs is not None) or (zobs is not None)
        if tobs is not None:
            zobs = self.cosm.z_of_t(tobs * s_per_myr)

        # If no camera supplied, operate directly on spectrum
        if cam is None:

            func = lambda xx, p0, p1: p0 * (xx / 1.)**p1

            if waves is None:
                waves = np.arange(rest_wave[0], rest_wave[1]+dlam, dlam)

            owaves, oflux = self.ObserveSpectrum(zobs, spec=spec, waves=waves,
                sfh=sfh, zarr=zarr, tarr=tarr, flux_units='Ang', hist=hist,
                extras=extras, idnum=idnum, window=window)

            rwaves = waves
            ok = np.logical_and(rwaves >= rest_wave[0], rwaves <= rest_wave[1])

            x = owaves[ok==1]

            if oflux.ndim == 2:
                batch_mode = True
                y = oflux[:,ok==1].swapaxes(0, 1)

                ma = np.max(y, axis=0)
                sl = -2.5 * np.ones(ma.size)
                guess = np.vstack((ma, sl)).T
            else:
                batch_mode = False
                y = oflux[ok==1]
                guess = np.array([oflux[np.argmin(np.abs(owaves - 1.))], -2.4])

        else:

            if filters is not None:
                assert rest_wave is None, \
                    "Set rest_wave=None if filters are supplied"

            if type(cam) not in [list, tuple]:
                cam = [cam]

            filt = []
            xphot = []
            dxphot = []
            ycorr = []
            for _cam in cam:
                _filters, _xphot, _dxphot, _ycorr = \
                    self.Photometry(sfh=sfh, hist=hist, idnum=idnum, spec=spec,
                    cam=_cam, filters=filters, filter_set=filter_set, waves=waves,
                    dlam=dlam, tarr=tarr, tobs=tobs, extras=extras, picky=picky,
                    zarr=zarr, zobs=zobs, rest_wave=rest_wave, window=window)

                filt.extend(list(_filters))
                xphot.extend(list(_xphot))
                dxphot.extend(list(_dxphot))
                ycorr.extend(list(_ycorr))

            # No matching filters? Return.
            if len(filt) == 0:
                if idnum is not None:
                    N = 1
                elif sfh is not None:
                    N = sfh.shape[0]
                else:
                    N = 1

                if return_norm:
                    return -99999 * np.ones((N, 2))
                else:
                    return -99999 * np.ones(N)

            filt = np.array(filt)
            xphot = np.array(xphot)
            dxphot = np.array(dxphot)
            ycorr = np.array(ycorr)

            # Sort arrays in ascending wavelength
            isort = np.argsort(xphot)

            _x = xphot[isort]
            _y = ycorr[isort]

            # Recover flux to do power-law fit
            xp, xm = dxphot.T
            dx = xp + xm

            # Need flux in units of A^-1
           #dnphot = c / ((xphot-xm) * 1e-4) - c / ((xphot + xp) * 1e-4)
           #dwdn = dx * 1e4 / dnphot
            _dwdn = (_x * 1e4)**2 / (c * 1e8)

            if rest_wave is not None:
                r = _x * 1e4 / (1. + zobs)
                ok = np.logical_and(r >= rest_wave[0], r <= rest_wave[1])
                x = _x[ok==1]
            else:
                ok = np.ones_like(_x)
                x = _x

            # Be careful in batch mode!
            if ycorr.ndim == 2:
                batch_mode = True
                _f = 10**(_y / -2.5) * flux_AB / _dwdn[:,None]
                y = _f[ok==1]
                ma = np.max(y, axis=0)
                sl = -2.5 * np.ones(ma.size)
                guess = np.vstack((ma, sl)).T
            else:
                batch_mode = False
                _f = 10**(_y / -2.5) * flux_AB / _dwdn
                y = _f[ok==1]
                ma = np.max(y)
                guess = np.array([ma, -2.])

            if ok.sum() == 2 and self.pf['verbose']:
                print("WARNING: Estimating slope at z={} from only two points: {}".format(zobs,
                    filt[isort][ok==1]))

        ##
        # Fit a PL to points.
        if method == 'fit':

            if len(x) < 2:
                if self.pf['verbose']:
                    print("Not enough points to estimate slope")

                if batch_mode:
                    corr = np.ones(y.shape[1])
                else:
                    corr = 1

                if return_norm:
                    return -99999 * corr, -99999 * corr
                else:
                    return -99999 * corr

            if batch_mode:
                N = y.shape[1]

                popt = -99999 * np.ones((2, N))
                pcov = -99999 * np.ones((2, 2, N))

                for i in range(N):

                    if not np.any(y[:,i] > 0):
                        continue

                    try:
                        popt[:,i], pcov[:,:,i] = curve_fit(_powlaw, x, y[:,i],
                            p0=guess[i], maxfev=10000)
                    except RuntimeError:
                        popt[:,i], pcov[:,:,i] = -99999, -99999

            else:
                try:
                    popt, pcov = curve_fit(_powlaw, x, y, p0=guess, maxfev=10000)
                except RuntimeError:
                    popt, pcov = -99999 * np.ones(2), -99999 * np.ones(2)

        elif method == 'linear':

            logx = np.log10(x)
            logy = np.log10(y)

            A = np.vstack([logx, np.ones(len(logx))]).T

            if batch_mode:
                N = y.shape[1]

                popt = -99999 * np.ones((2, N))
                pcov = -99999 * np.ones((2, 2, N))

                for i in range(N):
                    popt[:,i] = np.linalg.lstsq(A, logy[:,i],
                        rcond=None)[0][-1::-1]

            else:
                popt = np.linalg.lstsq(A, logy, rcond=None)[0]
                pcov = -99999 * np.ones(2)

        elif method == 'diff':

            assert cam is None, "Should only use to skip photometry."

            # Remember that galaxy number is second dimension
            logL = np.log(y)
            logw = np.log(x)

            if batch_mode:
                # Logarithmic derivative = beta
                beta = (logL[-1,:] - logL[0,:]) / (logw[-1,None] - logw[0,None])
            else:
                beta = (logL[-1] - logL[0]) / (logw[-1] - logw[0])

            popt = np.array([-99999, beta])

        else:
            raise NotImplemented('help me')

        if return_norm:
            return popt
        else:
            if return_err:
                return popt[1], np.sqrt(pcov[1,1])
            else:
                return popt[1]

    def ObserveSpectrum(self, zobs, **kwargs):
        return self.get_spec_obs(zobs, **kwargs)

    def get_spec_obs(self, zobs, spec=None, sfh=None, waves=None,
        flux_units='Hz', tarr=None, tobs=None, zarr=None, hist={},
        idnum=None, window=1, extras={}, nthreads=1, load=True):
        """
        Take an input spectrum and "observe" it at redshift z.

        Parameters
        ----------
        zobs : int, float
            Redshift of observation.
        waves : np.ndarray
            Simulate spectrum at these rest wavelengths [Angstrom]
        spec : np.ndarray
            Specific luminosities in [erg/s/A]


        Returns
        -------
        Observed wavelengths in microns, observed fluxes in erg/s/cm^2/Hz.

        """

        if spec is None:
            spec = self.get_spec_rest(waves, sfh=sfh, tarr=tarr, zarr=zarr,
                zobs=zobs, tobs=None, hist=hist, idnum=idnum,
                extras=extras, window=window, load=load)

        dL = self.cosm.LuminosityDistance(zobs)

        if waves is None:
            waves = self.src.wavelengths
            dwdn = self.src.dwdn
            assert len(spec) == len(waves)
        else:
            #freqs = c / (waves / 1e8)
            dwdn = waves**2 / (c * 1e8)
            #tmp = np.abs(np.diff(waves) / np.diff(freqs))
            #dwdn = np.concatenate((tmp, [tmp[-1]]))

        # Flux at Earth in erg/s/cm^2/Hz
        f = spec / (4. * np.pi * dL**2)

        # Correct for redshifting and change in units.
        if flux_units == 'Hz':
            f *= (1. + zobs)
        else:
            f /= dwdn
            f /= (1. + zobs)

        owaves = waves * (1. + zobs) / 1e4

        tau = self.OpticalDepth(zobs, owaves)
        T = np.exp(-tau)

        return owaves, f * T

    def Photometry(self, **kwargs):
        return self.get_photometry(**kwargs)

    def get_photometry(self, spec=None, sfh=None, cam='wfc3', filters='all',
        filter_set=None, dlam=20., rest_wave=None, extras={}, window=1,
        tarr=None, zarr=None, waves=None, zobs=None, tobs=None, band=None,
        hist={}, idnum=None, flux_units=None, picky=False, lbuffer=200.,
        ospec=None, owaves=None, load=True):
        """
        Just a wrapper around `Spectrum`.

        Returns
        -------
        Tuple containing (in this order):
            - Names of all filters included
            - Midpoints of photometric filters [microns]
            - Width of filters [microns]
            - Apparent magnitudes corrected for filter transmission.

        """

        assert (tobs is not None) or (zobs is not None)

        if zobs is None:
            zobs = self.cosm.z_of_t(tobs * s_per_myr)

        # Might be stored for all redshifts so pick out zobs
        if type(filters) == dict:
            assert zobs is not None
            filters = filters[round(zobs)]

        # Get transmission curves
        if cam in self.cameras.keys():
            filter_data = self.cameras[cam].read_throughputs(filter_set=filter_set,
                filters=filters)
        else:
            # Can supply spectral windows, e.g., Calzetti+ 1994, in which
            # case we assume perfect transmission but otherwise just treat
            # like photometric filters.
            assert type(filters) in [list, tuple, np.ndarray]

            #print("Generating photometry from {} spectral ranges.".format(len(filters)))

            wraw = np.array(filters)
            x1 = wraw.min()
            x2 = wraw.max()
            x = np.arange(x1-1, x2+1, 1.) * 1e-4 * (1. + zobs)

            # Note that in this case, the filter wavelengths are in rest-
            # frame units, so we convert them to observed wavelengths before
            # photometrizing everything.

            filter_data = {}
            for _window in filters:
                lo, hi = _window

                lo *= 1e-4 * (1. + zobs)
                hi *= 1e-4 * (1. + zobs)

                y = np.zeros_like(x)
                y[np.logical_and(x >= lo, x <= hi)] = 1
                mi = np.mean([lo, hi])
                dx = np.array([hi - mi, mi - lo])
                Tavg = 1.
                filter_data[_window] = x, y, mi, dx, Tavg

        _all_filters = list(filter_data.keys())

        # Sort filters in ascending wavelength
        _waves = []
        for _filter_ in _all_filters:
            _waves.append(filter_data[_filter_][2])

        sorter = np.argsort(_waves)
        all_filters = [_all_filters[ind] for ind in sorter]

        # Figure out spectral range we need to model for these filters.
        # Find bluest and reddest filters, set wavelength range with some
        # padding above and below these limits.
        lmin = np.inf
        lmax = 0.0
        ct = 0
        for filt in filter_data:
            x, y, cent, dx, Tavg = filter_data[filt]

            # If we're only doing this for the sake of measuring a slope, we
            # might restrict the range based on wavelengths of interest,
            # i.e., we may not use all the filters.

            # Right now, will include filters as long as their center is in
            # the requested band. This results in fluctuations in slope
            # measurements, so to be more stringent set picky=True.
            if rest_wave is not None:

                if picky:
                    l = (cent - dx[1]) * 1e4 / (1. + zobs)
                    r = (cent + dx[0]) * 1e4 / (1. + zobs)

                    if (l < rest_wave[0]) or (r > rest_wave[1]):
                        continue

                cent_r = cent * 1e4 / (1. + zobs)
                if (cent_r < rest_wave[0]) or (cent_r > rest_wave[1]):
                    continue

            lmin = min(lmin, cent - dx[1] * 1.2)
            lmax = max(lmax, cent + dx[0] * 1.2)
            ct += 1

        # No filters in range requested
        if ct == 0:
            return [], [], [], []

        # Here's our array of REST wavelengths
        if waves is None:
            # Convert from microns to Angstroms, undo redshift.
            lmin = lmin * 1e4 / (1. + zobs)
            lmax = lmax * 1e4 / (1. + zobs)

            lmin = max(lmin, self.src.wavelengths.min())
            lmax = min(lmax, self.src.wavelengths.max())

            # Force edges to be multiples of dlam
            l1 = lmin - lbuffer
            l1 -= l1 % dlam
            l2 = lmax + lbuffer

            waves = np.arange(l1, l2+dlam, dlam)

        # Get spectrum first.
        if (spec is None) and (ospec is None):
            spec = self.get_spec_rest(waves, sfh=sfh, tarr=tarr, tobs=tobs,
                zarr=zarr, zobs=zobs, band=band, hist=hist,
                idnum=idnum, extras=extras, window=window, load=load)

            # Observed wavelengths in micron, flux in erg/s/cm^2/Hz
            wave_obs, flux_obs = self.ObserveSpectrum(zobs, spec=spec,
                waves=waves, extras=extras, window=window)

        elif ospec is not None:
            flux_obs = ospec
            wave_obs = owaves
        else:
            raise ValueError('This shouldn\'t happen')

        # Might be running over lots of galaxies
        batch_mode = False
        if flux_obs.ndim == 2:
            batch_mode = True

        # Convert microns to cm. micron * (m / 1e6) * (1e2 cm / m)
        freq_obs = c / (wave_obs * 1e-4)

        # Why do NaNs happen? Just nircam.
        flux_obs[np.isnan(flux_obs)] = 0.0

        # Loop over filters and re-weight spectrum
        fphot = []
        xphot = []      # Filter centroids
        wphot = []      # Filter width
        yphot_corr = [] # Magnitudes corrected for filter transmissions.

        # Loop over filters, compute fluxes in band (accounting for
        # transmission fraction) and convert to observed magnitudes.
        for filt in all_filters:

            if filters != 'all':
                if filt not in filters:
                    continue

            x, T, cent, dx, Tavg = filter_data[filt]

            if rest_wave is not None:
                cent_r = cent * 1e4 / (1. + zobs)
                if (cent_r < rest_wave[0]) or (cent_r > rest_wave[1]):
                    continue

            # Re-grid transmission onto provided wavelength axis.
            T_regrid = np.interp(wave_obs, x, T, left=0, right=0)
            #func = interp1d(x, T, kind='cubic', fill_value=0.0,
            #    bounds_error=False)
            #T_regrid = func(wave_obs)

            #T_regrid = np.interp(np.log(wave_obs), np.log(x), T, left=0.,
            #    right=0)

            # Remember: observed flux is in erg/s/cm^2/Hz

            # Integrate over frequency to get integrated flux in band
            # defined by filter.
            if batch_mode:
                integrand = -1. * flux_obs * T_regrid[None,:]
                _yphot = np.sum(integrand[:,0:-1] * np.diff(freq_obs)[None,:],
                    axis=1)
            else:
                integrand = -1. * flux_obs * T_regrid
                _yphot = np.sum(integrand[0:-1] * np.diff(freq_obs))

                #_yphot = np.trapz(integrand, x=freq_obs)

            corr = np.sum(T_regrid[0:-1] * -1. * np.diff(freq_obs), axis=-1)

            fphot.append(filt)
            xphot.append(cent)
            yphot_corr.append(_yphot / corr)
            wphot.append(dx)

        xphot = np.array(xphot)
        wphot = np.array(wphot)
        yphot_corr = np.array(yphot_corr)

        # Convert to magnitudes
        mphot = -2.5 * np.log10(yphot_corr / flux_AB)

        # We're done
        return fphot, xphot, wphot, mphot

    def Spectrum(self, waves, **kwargs):
        return self.get_spec_rest(waves, **kwargs)

    def get_spec_rest(self, waves, sfh=None, tarr=None, zarr=None, window=1,
        zobs=None, tobs=None, band=None, idnum=None, units='Hz', hist={},
        extras={}, load=True):
        """
        This is just a wrapper around `Luminosity`.
        """

        # Select single row of SFH array if `idnum` provided.
        if sfh.ndim == 2 and idnum is not None:
            sfh = sfh[idnum,:]

        batch_mode = sfh.ndim == 2
        time_series = (zobs is None) and (tobs is None)

        # Shape of output array depends on some input parameters.
        shape = []
        if batch_mode:
            shape.append(sfh.shape[0])
        if time_series:
            shape.append(tarr.size)
        shape.append(len(waves))

        # Do kappa up front?

        pb = ProgressBar(waves.size, name='l(nu)', use=self.pf['progress_bar'])
        pb.start()

        ##
        # Can thread this calculation
        ##
        if (self.pf['nthreads'] is not None):

            try:
                import pymp
                have_pymp = True
            except ImportError:
                have_pymp = False

            assert have_pymp, "Need pymp installed to run with nthreads!=None!"

            pymp.config.num_threads = self.pf['nthreads']

            if self.pf['verbose']:
                print("Setting nthreads={} for spectral synthesis.".format(
                        self.pf['nthreads']))

            spec = pymp.shared.array(shape, dtype='float64')
            with pymp.Parallel(self.pf['nthreads']) as p:
                for i in p.xrange(0, waves.size):
                    slc = (Ellipsis, i) if (batch_mode or time_series) else i

                    spec[slc] = self.get_lum(wave=waves[i],
                        sfh=sfh, tarr=tarr, zarr=zarr, zobs=zobs, tobs=tobs,
                        band=band, hist=hist, idnum=idnum,
                        extras=extras, window=window, load=load)

                    pb.update(i)

        else:

            spec = np.zeros(shape)
            for i, wave in enumerate(waves):
                slc = (Ellipsis, i) if (batch_mode or time_series) else i

                spec[slc] = self.get_lum(wave=wave,
                    sfh=sfh, tarr=tarr, zarr=zarr, zobs=zobs, tobs=tobs,
                    band=band, hist=hist, idnum=idnum,
                    extras=extras, window=window, load=load)

                pb.update(i)

        pb.finish()

        if units in ['A', 'Ang']:
            dwdn = waves**2 / (c * 1e8)
            spec /= dwdn

        return spec

    #def Magnitude(self, wave=1600., sfh=None, tarr=None, zarr=None, window=1,
    #    zobs=None, tobs=None, band=None, idnum=None, hist={}, extras={}):
#
    #    L = self.get_lum(wave=wave, sfh=sfh, tarr=tarr, zarr=zarr,
    #        zobs=zobs, tobs=tobs, band=band, idnum=idnum, hist=hist,
    #        extras=extras, window=window)
#
    #    MAB = self.magsys.L_to_MAB(L)
#
    #    return MAB

    def _oversample_sfh(self, ages, sfh, i):
        """
        Over-sample time axis while stellar populations are young if the time
        resolution is worse than 1 Myr / grid point.
        """

        batch_mode = sfh.ndim == 2

        # Use 1 Myr time resolution for final stretch.
        # final stretch is determined by `oversampling_below` attribute.
        # This loop determines how many elements at the end of
        # `ages` are within the `oversampling_below` zone.

        ct = 0
        while ages[-1-ct] < self.oversampling_below:
            ct += 1

            if ct + 1 == len(ages):
                break

        ifin = -1 - ct
        ages_x = np.arange(ages[-1], ages[ifin], 1.)[-1::-1]

        # `ages_x` is an array of ages at higher resolution than native data
        # to-be-tagged on the end of supplied `ages`.

        # Must augment ages and dt accordingly
        _ages = np.hstack((ages[0:ifin], ages_x))
        _dt = np.abs(np.diff(_ages) * 1e6)

        if batch_mode:
            xSFR = np.ones((sfh.shape[0], ages_x.size-1))
        else:
            xSFR = np.ones(ages_x.size-1)

        # Must allow non-constant SFR within over-sampled region
        # as it may be tens of Myr.
        # Walk back from the end and fill in SFR
        N = int((ages_x.size - 1) / ct)
        for _i in range(0, ct):

            if batch_mode:
                slc = Ellipsis, slice(-1 * N * _i-1, -1 * N * (_i + 1) -1, -1)
            else:
                slc = slice(-1 * N * _i-1, -1 * N * (_i + 1) -1, -1)

            if batch_mode:
                _sfh_rs = np.array([sfh[:,-_i-2]]*N).T
                xSFR[slc] = _sfh_rs * np.ones(N)[None,:]
            else:
                xSFR[slc] = sfh[-_i-2] * np.ones(N)

        # Need to tack on the SFH at ages older than our
        # oversampling approach kicks in.
        if batch_mode:
            if ct + 1 == len(ages):
                _SFR = np.hstack((sfh[:,0][:,None], xSFR))
            else:
                _SFR = np.hstack((sfh[:,0:i+1][:,0:ifin+1], xSFR))
        else:

            if ct + 1 == len(ages):
                _SFR = np.hstack((sfh[0], xSFR))
            else:
                _SFR = np.hstack((sfh[0:i+1][0:ifin+1], xSFR))

        return _ages, _SFR

    @property
    def _cache_lum_ctr(self):
        if not hasattr(self, '_cache_lum_ctr_'):
            self._cache_lum_ctr_ = 0
        return self._cache_lum_ctr_

    def _cache_kappa(self, wave):
        if not hasattr(self, '_cache_kappa_'):
            self._cache_kappa_ = {}

        if wave in self._cache_kappa_:
            return self._cache_kappa_[wave]

        return None

    def _cache_lum(self, kwds):
        """
        Cache object for spectral synthesis of stellar luminosity.
        """
        if not hasattr(self, '_cache_lum_'):
            self._cache_lum_ = {}

        notok = -1

        t1 = time.time()

        # If we set order by hand, it greatly speeds things up because
        # more likely than not, the redshift and wavelength are the only
        # things that change and that's an easy logical check to do.
        # Checking that SFHs, histories, etc., is more expensive.
        ok_keys = ('wave', 'zobs', 'tobs', 'idnum', 'sfh', 'tarr', 'zarr',
            'window', 'band', 'hist', 'extras', 'load', 'energy_units')

        ct = -1

        # Loop through keys to do more careful comparison for unhashable types.
        #all_waves = self._cache_lum_waves_

        all_keys = self._cache_lum_.keys()

        # Search in reverse order since we often the keys represent different
        # wavelengths, which are generated in ascending order.
        for keyset in all_keys:

            ct += 1

            # Remember: keyset is just a number.
            kw, data = self._cache_lum_[keyset]

            # Check wavelength first. Most common thing.

            # If we're not being as careful as possible, retrieve cached
            # result so long as wavelength and zobs match requested values.
            # This should only be used when SpectralSynthesis is summoned
            # internally! Likely to lead to confusing behavior otherwise.
            if (self.careful_cache == 0) and ('wave' in kw) and ('zobs' in kw):
                if (kw['wave'] == kwds['wave']) and (kw['zobs'] == kwds['zobs']):
                    notok = 0
                    break

            notok = 0
            # Loop over cached keywords, compare to those supplied.
            for key in ok_keys:

                if key not in kwds:
                    notok += 1
                    break

                #if isinstance(kw[key], collections.Hashable):
                #    if kwds[key] == kw[key]:
                #        continue
                #    else:
                #        notok += 1
                #        break
                #else:
                # For unhashable types, must work on case-by-case basis.
                if type(kwds[key]) != type(kw[key]):
                    notok += 1
                    break
                elif type(kwds[key]) == np.ndarray:
                    if np.array_equal(kwds[key], kw[key]):
                        continue
                    else:
                        # This happens when, e.g., we pass SFH by hand.
                        notok += 1
                        break
                elif type(kwds[key]) == dict:
                    if kwds[key] == kw[key]:
                        continue
                    else:

                        #for _key in kwds[key]:
                        #    print(_key, kwds[key][_key] == kw[key][_key])
                        #
                        #raw_input('<enter>')

                        notok += 1
                        break
                else:
                    if kwds[key] == kw[key]:
                        continue
                    else:
                        notok += 1
                        break

            if notok > 0:
                #print(keyset, key)
                continue

            # If we're here, load this thing.
            break

        t2 = time.time()

        if notok < 0:
            return kwds, None
        elif notok == 0:
            if (self.pf['verbose'] and self.pf['debug']):
                print("Loaded from cache! Took N={} iterations, {} sec to find match".format(ct, t2 - t1))
            # Recall that this is (kwds, data)
            return self._cache_lum_[keyset]
        else:
            return kwds, None

    def Luminosity(self, **kwargs):
        return self.get_lum(**kwargs)

    def get_lum(self, wave=1600., sfh=None, tarr=None, zarr=None,
        window=1,
        zobs=None, tobs=None, band=None, idnum=None, hist={}, extras={},
        load=True, energy_units=True):
        """
        Synthesize luminosity of galaxy with given star formation history at a
        given wavelength and time.

        Parameters
        ----------
        sfh : np.ndarray
            Array of SFRs. If 1-D, should be same shape as time or redshift
            array. If 2-D, first dimension should correspond to galaxy number
            and second should be time.
        tarr : np.ndarray
            Array of times in ascending order [Myr].
        zarr : np.ndarray
            Array of redshift in ascending order (so decreasing time). Only
            supply if not passing `tarr` argument.
        wave : int, float
            Wavelength of interest [Angstrom]
        window : int
            Average over interval about `wave`. [Angstrom]
        zobs : int, float
            Redshift of observation.
        tobs : int, float
            Time of observation (will be computed self-consistently if `zobs`
            is supplied).
        hist : dict
            Extra information we may need, e.g., metallicity, dust optical
            depth, etc. to compute spectrum.

        Returns
        -------
        Luminosity at wavelength=`wave` in units of erg/s/Hz.

        """

        setup_1 = (sfh is not None) and \
            ((tarr is not None) or (zarr is not None))
        setup_2 = hist != {}

        do_all_time = False
        if (tobs is None) and (zobs is None):
            do_all_time = True
        #assert (tobs is not None) or (zobs is not None), \
        #    "Must supply time or redshift of observation, `tobs` or `zobs`!"

        assert setup_1 or setup_2

        if setup_1:
            assert (sfh is not None)
        elif setup_2:
            assert ('z' in hist) or ('t' in hist), \
                "`hist` must contain redshifts, `z`, or times, `t`."
            sfh = hist['SFR'] if 'SFR' in hist else hist['sfr']
            if 'z' in hist:
                zarr = hist['z']
            else:
                tarr = hist['t']

        kw = {'sfh':sfh, 'zobs':zobs, 'tobs':tobs, 'wave':wave, 'tarr':tarr,
            'zarr':zarr, 'band':band, 'idnum':idnum, 'hist':hist,
            'extras':extras, 'window': window, 'energy_units': energy_units}

        if load:
            _kwds, cached_result = self._cache_lum(kw)
        else:
            self._cache_lum_ = {}
            cached_result = None

        if cached_result is not None:
            return cached_result

        if sfh.ndim == 2 and idnum is not None:
            sfh = sfh[idnum,:]
            if 'Z' in hist:
                Z = hist['Z'][idnum,:]

            # Don't necessarily need Mh here.
            if 'Mh' in hist:
                Mh = hist['Mh'][idnum,:]
        else:
            if 'Mh' in hist:
                Mh = hist['Mh']
            if 'Z' in hist:
                Z = hist['Z']

        # If SFH is 2-D it means we're doing this for multiple galaxies at once.
        # The first dimension will be number of galaxies and second dimension
        # is time/redshift.
        batch_mode = sfh.ndim == 2

        # Parse time/redshift information
        if tarr is not None:
            zarr = self.cosm.z_of_t(tarr * s_per_myr)
        else:
            assert tarr is None

            tarr = self.cosm.t_of_z(zarr) / s_per_myr

        assert np.all(np.diff(tarr) > 0), \
            "Must supply SFH in time-ascending (i.e., redshift-descending) order!"

        # Convert tobs to redshift.
        if tobs is not None:
            zobs = self.cosm.z_of_t(tobs * s_per_myr)
            if type(tobs) == np.ndarray:
                assert (tobs.min() >= tarr.min()) and (tobs.max() <= tarr.max()), \
                    "Requested times of observation (`tobs={}-{}`) not in supplied range ({}, {})!".format(tobs.min(),
                        tobs.max(), tarr.min(), tarr.max())
            else:
                assert tarr.min() <= tobs <= tarr.max(), \
                    "Requested time of observation (`tobs={}`) not in supplied range ({}, {})!".format(tobs,
                        tarr.min(), tarr.max())

        # Prepare slice through time-axis.
        if zobs is None:
            slc = Ellipsis
            izobs = None
        else:
            # Need to be sure that we grab a grid point exactly at or just
            # below the requested redshift (?)
            izobs = np.argmin(np.abs(zarr - zobs))
            if zarr[izobs] > zobs:
                izobs += 1

            if batch_mode:
                #raise NotImplemented('help')
                # Need to slice over first dimension now...
                slc = Ellipsis, slice(0, izobs+1)
            else:
                slc = slice(0, izobs+1)

            if not (zarr.min() <= zobs <= zarr.max()):
                if batch_mode:
                    return np.ones(sfh.shape[0]) * -99999
                else:
                    return -99999

        fill = np.zeros(1)
        tyr = tarr * 1e6
        dt = np.hstack((np.diff(tyr), fill))

        # Figure out if we need to over-sample the grid we've got to more
        # accurately solve for young stellar populations.
        oversample = self.oversampling_enabled and (dt[-2] > 1.01e6)
        # Used to also require zobs is not None. Why?

        ##
        # Done parsing time/redshift

        # Is this luminosity in some bandpass or monochromatic?
        if band is not None:
            # Will have been supplied in Angstroms
            b = h_p * c / (np.array(band) * 1e-8) / erg_per_ev

            Loft = self.src.IntegratedEmission(b[1], b[0],
                energy_units=energy_units)

            # Need to get Hz^-1 units back
            #db = b[0] - b[1]
            #Loft = Loft / (db * erg_per_ev / h_p)

            #raise NotImplemented('help!')
        else:
            Loft = self.src.L_per_sfr_of_t(wave=wave, avg=window, raw=False)

            assert energy_units
        #print("Synth. Lum = ", wave, window)
        #

        # Setup interpolant for luminosity as a function of SSP age.
        Loft[Loft == 0] = tiny_lum
        _func = interp1d(np.log(self.src.times), np.log(Loft),
            kind=self.pf['pop_synth_age_interp'], bounds_error=False,
            fill_value=(Loft[0], Loft[-1]))

        # Extrapolate linearly at times < 1 Myr
        _m = (Loft[1] - Loft[0]) / (self.src.times[1] - self.src.times[0])
        L_small_t = lambda age: _m * age + Loft[0]

        if not (self.src.pf['source_aging'] or self.src.pf['source_ssp']):
            L_asympt = np.exp(_func(np.log(self.src.pf['source_tsf'])))

        #L_small_t = lambda age: Loft[0]

        # Extrapolate as PL at t < 1 Myr based on first two
        # grid points
        #m = np.log(Loft[1] / Loft[0]) \
        #  / np.log(self.src.times[1] / self.src.times[0])
        #func = lambda age: np.exp(m * np.log(age) + np.log(Loft[0]))

        #if zobs is None:
        Lhist = np.zeros(sfh.shape)
        #if hasattr(self, '_sfh_zeros'):
        #    Lhist = self._sfh_zeros.copy()
        #else:
        #    Lhist = np.zeros_like(sfh)
        #    self._sfh_zeros = Lhist.copy()
        #else:
        #    pass
            # Lhist will just get made once. Don't need to initialize

        ##
        # Loop over the history of object(s) and compute the luminosity of
        # simple stellar populations of the corresponding ages (relative to
        # zobs).
        ##

        # Start from initial redshift and move forward in time, i.e., from
        # high redshift to low.

        for i, _tobs in enumerate(tarr):

            # If zobs is supplied, we only have to do one iteration
            # of this loop. This is just a dumb way to generalize this function
            # to either do one redshift or return a whole history.
            if not do_all_time:
                if (zarr[i] > zobs):
                    continue

            ##
            # Life if easy for constant SFR models
            if not (self.src.pf['source_aging'] or self.src.pf['source_ssp']):

                if not do_all_time:
                    Lhist = L_asympt * sfh[:,i]
                    break

                raise NotImplemented('does this happne?')
                Lhist[:,i] = L_asympt * sfh[:,i]

                continue

            # If we made it here, it's time to integrate over star formation
            # at previous times. First, retrieve ages of stars formed in all
            # past star forming episodes.
            ages = tarr[i] - tarr[0:i+1]
            # Note: this will be in order of *descending* age, i.e., the
            # star formation episodes furthest in the past are first in the
            # array.

            # Recall also that `sfh` contains SFRs for all time, so any
            # z < zobs will contain zeroes, hence all the 0:i+1 slicing below.

            # Treat metallicity evolution? If so, need to grab luminosity as
            # function of age and Z.
            if self.pf['pop_enrichment']:

                assert batch_mode

                logA = np.log10(ages)
                logZ = np.log10(Z[:,0:i+1])
                L_per_msun = np.zeros_like(ages)
                logL_at_wave = self.L_of_Z_t(wave)

                L_per_msun = np.zeros_like(logZ)
                for j, _Z_ in enumerate(range(logZ.shape[0])):
                    L_per_msun[j,:] = 10**logL_at_wave(logA, logZ[j,:],
                        grid=False)

                # erg/s/Hz
                if batch_mode:
                    Lall = L_per_msun[:,0:i+1] * sfh[:,0:i+1]
                else:
                    Lall = L_per_msun[0:i+1] * sfh[0:i+1]

                if oversample:
                    raise NotImplemented('help!')
                else:
                    _dt = dt[0:i]

                _ages = ages
            else:

                ##
                # If time resolution is >= 2 Myr, over-sample final interval.
                if oversample and len(ages) > 1:

                    if batch_mode:
                        _ages, _SFR = self._oversample_sfh(ages, sfh[:,0:i+1], i)
                    else:
                        _ages, _SFR = self._oversample_sfh(ages, sfh[0:i+1], i)

                    _dt = np.abs(np.diff(_ages) * 1e6)

                    # `_ages` is in order of old to young.

                    # Now, compute luminosity at expanded ages.
                    L_per_msun = np.exp(_func(np.log(_ages)))

                    # Interpolate linearly at t < 1 Myr
                    L_per_msun[_ages < 1] = L_small_t(_ages[_ages < 1])
                    #L_per_msun[_ages < 10] = 0.

                    # erg/s/Hz/yr
                    if batch_mode:
                        Lall = L_per_msun * _SFR
                    else:
                        Lall = L_per_msun * _SFR

                else:
                    L_per_msun = np.exp(_func(np.log(ages)))
                    #L_per_msun = np.exp(np.interp(np.log(ages),
                    #    np.log(self.src.times), np.log(Loft),
                    #    left=np.log(Loft[0]), right=np.log(Loft[-1])))

                    _dt = dt[0:i]

                    # Fix early time behavior
                    L_per_msun[ages < 1] = L_small_t(ages[ages < 1])

                    _ages = ages

                    # erg/s/Hz/yr
                    if batch_mode:
                        Lall = L_per_msun * sfh[:,0:i+1]
                    else:
                        Lall = L_per_msun * sfh[0:i+1]

                # Correction for IMF sampling (can't use SPS).
                #if self.pf['pop_sample_imf'] and np.any(bursty):
                #    life = self._stars.tab_life
                #    on = np.array([life > age for age in ages])
                #
                #    il = np.argmin(np.abs(wave - self._stars.wavelengths))
                #
                #    if self._stars.aging:
                #        raise NotImplemented('help')
                #        lum = self._stars.tab_Ls[:,il] * self._stars.dldn[il]
                #    else:
                #        lum = self._stars.tab_Ls[:,il] * self._stars.dldn[il]
                #
                #    # Need luminosity in erg/s/Hz
                #    #print(lum)
                #
                #    # 'imf' is (z or age, mass)
                #
                #    integ = imf[bursty==1,:] * lum[None,:]
                #    Loft = np.sum(integ * on[bursty==1], axis=1)
                #
                #    Lall[bursty==1] = Loft


            # Apply local reddening
            #tau_bc = self.pf['pop_tau_bc']
            #if tau_bc > 0:
            #
            #    corr = np.ones_like(_ages) * np.exp(-tau_bc)
            #    corr[_ages > self.pf['pop_age_bc']] = 1
            #
            #    Lall *= corr

            ###
            ## Integrate over all times up to this tobs
            if batch_mode:
                # Should really just np.sum here...using trapz assumes that
                # the SFH is a smooth function and not a series of constant
                # SFRs. Doesn't really matter in practice, though.
                if not do_all_time:
                    Lhist = np.trapz(Lall, dx=_dt, axis=1)
                else:
                    Lhist[:,i] = np.trapz(Lall, dx=_dt, axis=1)
            else:
                if not do_all_time:
                    Lhist = np.trapz(Lall, dx=_dt)
                else:
                    Lhist[i] = np.trapz(Lall, dx=_dt)

            ##
            # In this case, we only need one iteration of this loop.
            ##
            if not do_all_time:
                break


        ##
        # Redden spectra
        ##
        tau = np.zeros_like(sfh)
        if 'Sd' in hist:

            # Redden away!
            if np.any(hist['Sd'] > 0) and (band is None):

                assert 'kappa' in extras

                #_kappa = self._cache_kappa(wave)

                #if _kappa is None:
                kappa = extras['kappa'](wave=wave, Mh=Mh, z=zobs)
                #self._cache_kappa_[wave] = kappa
                #else:
                #    kappa = _kappa

                kslc = idnum if idnum is not None else Ellipsis

                if idnum is not None:
                    Sd = hist['Sd'][kslc]
                    if type(hist['fcov']) in [int, float, np.float64]:
                        fcov = hist['fcov']
                    else:
                        fcov = hist['fcov'][kslc]

                    rand = hist['rand'][kslc]
                else:
                    Sd = hist['Sd']
                    fcov = hist['fcov']
                    rand = hist['rand']

                tau = kappa * Sd

                clear = rand > fcov
                block = ~clear

                if idnum is not None:
                    Lout = Lhist * np.exp(-tau[izobs])
                    #if self.pf['pop_dust_holes'] == 'big':
                    #    Lout = Lhist * clear[izobs] \
                    #         + Lhist * np.exp(-tau[izobs]) * block[izobs]
                    #else:
                    #    Lout = Lhist * (1. - fcov[izobs]) \
                    #         + Lhist * fcov[izobs] * np.exp(-tau[izobs])
                else:
                    Lout = Lhist * np.exp(-tau[:,izobs])
                    #if self.pf['pop_dust_holes'] == 'big':
                    #    print(Lhist.shape, clear.shape, tau.shape, block.shape)
                    #    Lout = Lhist * clear[:,izobs] \
                    #         + Lhist * np.exp(-tau[:,izobs]) * block[:,izobs]
                    #else:
                    #    Lout = Lhist * (1. - fcov[:,izobs]) \
                    #         + Lhist * fcov[:,izobs] * np.exp(-tau[:,izobs])
            else:
                Lout = Lhist.copy()
        else:
            Lout = Lhist.copy()

        #del Lhist, tau, Lall
        #gc.collect()

        ##
        # Sum luminosity of parent halos along merger tree
        ##

        # Don't change shape, just zero-out luminosities of
        # parent halos after they merge?
        if hist is not None:
            do_mergers = self.pf['pop_mergers'] and batch_mode

            do_mergers = do_mergers and 'children' in hist

            if do_mergers:
                do_mergers = do_mergers and hist['children'] is not None

            if do_mergers:
                flags = hist['flags']
                child_iz, child_iM, is_main = hist['children'].T
                is_central = is_main

                # Convert z indices to ARES order.
                child_iz = hist['z'].size - child_iz - 1

                if np.all(is_central == 1):
                    pass
                else:
                    print("Looping over {} halos...".format(sfh.shape[0]))
                    pb = ProgressBar(sfh.shape[0],
                        use=self.pf['progress_bar'],
                        name='L += L_progenitors')
                    pb.start()

                    # Loop over all 'branches'
                    for i in range(sfh.shape[0]):

                        pb.update(i)

                        # Be careful with disrupted halos.
                        # In the future, could deposit this luminosity
                        # onto a grid or look for missed descendants and
                        # perform some kind of branch grafting.
                        if np.any(flags[i,:] == 1):
                            j = np.argwhere(flags[i,:] == 1)
                            if zobs <= zarr[j]:
                                Lout[i] = 0.0

                            continue

                        # This means the i'th halo is alive and well at
                        # the final redshift, i.e., it's a central and
                        # we don't need to do anything here.
                        if is_central[i]:
                            continue

                        # Only increment luminosity of descendants
                        # after merger.
                        # Remember: ARES indices go from high-z to low-z
                        zmerge = zarr[child_iz[i]]
                        if zobs > zmerge:
                            continue

                        # At this point, need to figure out which child
                        # halos to dump mass and SFH into...
                        # Lout is just 1-D at this point, i.e., just
                        # luminosity *now*.
                        # Add luminosity to child halo. Zero out
                        # luminosity of parent to avoid double
                        # counting. Note that nh will
                        # also have been zeroed out but it's good to
                        # zero-out both.
                        # NOTE: should use dust reddening of
                        # descendant, hence use of Lhist again
                        T = np.exp(-tau[child_iM[i],izobs])
                        Lout[child_iM[i]] += Lhist[i] * T
                        Lout[i] = 0.0

                    pb.finish()

            #elif (hist['children'] is not None):
            #    # Not treating mergers. Just filter out all non-centrals?
            #    pass

        ##
        # Will be unhashable types so just save to a unique identifier
        ##
        self._cache_lum_[self._cache_lum_ctr] = kw, Lout
        self._cache_lum_ctr_ += 1

        # Get outta here.
        return Lout
