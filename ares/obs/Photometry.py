"""

Photometry.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Mon 27 Jan 2020 09:59:10 EST

Description:

"""

import numpy as np
from .Survey import Survey
from ..util import ParameterFile
from ..physics.Constants import flux_AB, c

all_cameras = ['wfc', 'wfc3', 'nircam', 'roman', 'irac', 'wise', '2mass',
    'panstarrs', 'euclid', 'spherex']

class Photometry(object):
    def __init__(self, **kwargs):
        self.pf = ParameterFile(**kwargs)

    @property
    def force_perfect(self):
        if not hasattr(self, '_force_perfect'):
            self._force_perfect = False
        return self._force_perfect

    @force_perfect.setter
    def force_perfect(self, value):
        self._force_perfect = value

    @property
    def cameras(self):
        if not hasattr(self, '_cameras'):
            self._cameras = {}
            for cam in all_cameras:
                self._cameras[cam] = Survey(cam=cam,
                    force_perfect=self.force_perfect,
                    cache=self.pf['pop_synth_cache_phot'])

        return self._cameras

    def get_filter_info(self, cam, filt):
        """
        Returns the central wavelength and width of user-supplied filter,
        both in microns.
        """
        return self.cameras[cam].get_filter_info(filt)

    def get_required_spectral_range(self, zobs, cam, filters=None,
        filter_set=None, picky=True, rest_wave=None, tol=0.2, dlam=20.):
        """
        Return a range of rest-wavelengths [Angstroms] needed to sample the
        full range of wavelengths probed by a given set of photometric filters.
        """

        # Might be stored for all redshifts so pick out zobs
        if type(filters) == dict:
            filters = filters[round(zobs)]

        # Get transmission curves
        if cam in self.cameras.keys():
            filter_data = self.cameras[cam].read_throughputs(
                filter_set=filter_set, filters=filters)

        # Figure out spectral range we need to model for these filters.
        # Find bluest and reddest filters, set wavelength range with some
        # padding above and below these limits.
        lmin = np.inf
        lmax = 0.0
        for filt in filter_data:
            x, y, cent, dx, Tavg = filter_data[filt]

            # If we're only doing this for the sake of measuring a slope, we
            # might restrict the range based on wavelengths of interest,
            # i.e., we may not use all the filters.

            # Right now, will include filters as long as their center is in
            # the requested band. This results in fluctuations in slope
            # measurements, so to be more stringent set picky=True.
            #if rest_wave is not None:

            #    if picky:
            #        l = (cent - dx[1]) * 1e4 / (1. + zobs)
            #        r = (cent + dx[0]) * 1e4 / (1. + zobs)

            #        if (l < rest_wave[0]) or (r > rest_wave[1]):
            #            continue

            #    cent_r = cent * 1e4 / (1. + zobs)
            #    if (cent_r < rest_wave[0]) or (cent_r > rest_wave[1]):
            #        continue

            lmin = min(lmin, cent - dx[1] * (1. + tol))
            lmax = max(lmax, cent + dx[0] * (1. + tol))

        # Convert from microns to Angstroms, undo redshift.
        lmin = lmin * 1e4 / (1. + zobs)
        lmax = lmax * 1e4 / (1. + zobs)

        #lmin = max(lmin, self.src.tab_waves_c.min())
        #lmax = min(lmax, self.src.tab_waves_c.max())

        # Force edges to be multiples of dlam
        l1 = lmin - dlam * 2
        l1 -= l1 % dlam
        l2 = lmax + dlam * 2

        waves = np.arange(l1, l2+dlam, dlam)

        return waves

    def get_photometry(self, flux, owaves, flux_units=None,
        cam='wfc3', filters='all', filter_set=None, presets=None,
        rest_wave=None,
        idnum=None, picky=False,
        load=True, use_pbar=True):
        """
        Take as input a spectrum (or set of spectra) and 'photometrize' them,
        i.e., return corresponding photometry in some set of filters.

        Parameters
        ----------
        flux : np.ndarray
            This should be an array of shape (num galaxies, num wavelengths)
            containing the observed spectra of interest.
        waves : np.ndarray
            Array of observed wavelengths in microns.


        Returns
        -------
        Tuple containing (in this order):
            - Names of all filters included
            - Midpoints of photometric filters [microns]
            - Width of filters [microns]
            - Apparent magnitudes corrected for filter transmission.

        """

        # Might be stored for all redshifts so pick out zobs
        if type(filters) == dict:
            filters = filters[round(zobs)]

        # Get transmission curves
        if cam in self.cameras.keys():
            filter_data = self.cameras[cam].read_throughputs(
                filter_set=filter_set, filters=filters)
        else:
            # Can supply spectral windows, e.g., Calzetti+ 1994, in which
            # case we assume perfect transmission but otherwise just treat
            # like photometric filters.
            assert type(filters) in [list, tuple, np.ndarray]

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

        # Might be running over lots of galaxies
        batch_mode = False
        if flux.ndim == 2:
            batch_mode = True

        # Convert microns to cm. micron * (m / 1e6) * (1e2 cm / m)
        freq_obs = c / (owaves * 1e-4)

        # Why do NaNs happen? Just nircam.
        #flux[np.isnan(flux)] = 0.0

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

            #if rest_wave is not None:
            #    cent_r = cent * 1e4 / (1. + zobs)
            #    if (cent_r < rest_wave[0]) or (cent_r > rest_wave[1]):
            #        continue

            # Re-grid transmission onto provided wavelength axis.
            T_regrid = np.interp(owaves, x, T, left=0, right=0)
            #func = interp1d(x, T, kind='cubic', fill_value=0.0,
            #    bounds_error=False)
            #T_regrid = func(wave_obs)

            #T_regrid = np.interp(np.log(wave_obs), np.log(x), T, left=0.,
            #    right=0)

            # Remember: observed flux is in erg/s/cm^2/Hz

            # Integrate over frequency to get integrated flux in band
            # defined by filter.
            if batch_mode:
                integrand = -1. * flux * T_regrid[None,:]
                _yphot = np.sum(integrand[:,0:-1] * np.diff(freq_obs)[None,:],
                    axis=1)
            else:
                integrand = -1. * flux * T_regrid
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

        if batch_mode:
            mphot = np.swapaxes(mphot, 0, 1)

        # We're done
        return fphot, xphot, wphot, mphot


def get_filters_from_waves(z, fset, wave_lo=1300., wave_hi=2600., picky=True):
    """
    Given a redshift and a full filter set, return the filters that probe
    a given wavelength range (rest-UV continuum by default).

    Parameters
    ----------
    z : int, float
        Redshift of interest.
    fset : dict
        A filter set, i.e., the kind of dictionary created by
        ares.obs.Survey.get_throughputs.
    wave_lo, wave_hi: int, float
        Rest wavelengths bounding range of interest [Angstrom].
    picky : bool
        If True, will only return filters that lie entirely in the specified
        wavelength range. If False, returned list of filters will contain those
        that straddle the boundaries (if such cases exist).

    Returns
    -------
    List of filters that cover the specified rest-wavelength range.

    """

    # Compute observed wavelengths in microns
    l1 = wave_lo * (1. + z) * 1e-4
    l2 = wave_hi * (1. + z) * 1e-4

    out = []
    for filt in fset.keys():
        # Hack out numbers
        _x, _y, mid, dx, Tbar = fset[filt]

        fhi = mid + dx[0]
        flo = mid - dx[1]

        entirely_in = (flo >= l1) and (fhi <= l2)
        partially_in = (flo <= l1 <= fhi) or (flo <= l2 <= fhi)

        if picky and (partially_in and not partially_in):
            continue
        elif picky and entirely_in:
            pass
        elif not (partially_in or entirely_in):
            continue
        elif (partially_in or entirely_in):
            pass

        out.append(filt)

    return out
