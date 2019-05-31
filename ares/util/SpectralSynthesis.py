"""

SpectralSynthesis.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Sat 25 May 2019 09:58:14 EDT

Description: 

"""

import numpy as np
from ..util import Survey
from ..util import ProgressBar
from ..util import ParameterFile
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from ..physics.Cosmology import Cosmology
from ..physics.Constants import s_per_myr, c, h_p, erg_per_ev

flux_AB = 3631. * 1e-23 # 3631 * 1e-23 erg / s / cm**2 / Hz
nanoJ = 1e-23 * 1e-9

all_cameras = ['wfc', 'wfc3', 'nircam']

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
    def cameras(self):
        if not hasattr(self, '_cameras'):
            self._cameras = {}
            for cam in all_cameras:
                self._cameras[cam] = Survey(cam=cam)

        return self._cameras

    def Slope(self, zobs, spec=None, waves=None, sfh=None, zarr=None, tarr=None,
        tobs=None, cam=None, rest_wave=(1600., 2300.), band=None, hist={},
        return_norm=False, filters=None, filter_set=None, dlam=10., idnum=None,
        method='fit', window=1, extras={}, picky=False):
        """
        Compute slope in some wavelength range or using photometry.
        """
        
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
            
            # Log-linear fit
            func = lambda x, p0, p1: p0 * (x / 1.)**p1
            
            if type(cam) not in [list, tuple]:
                cam = [cam]
            
            filt = []
            xphot = []
            dxphot = []
            ycorr = []
            for _cam in cam:
                _filters, _xphot, _dxphot, _yphot, _ycorr = \
                    self.Photometry(sfh=sfh, hist=hist, idnum=idnum,
                    cam=_cam, filters=filters, filter_set=filter_set, 
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
                else:
                    N = sfh.shape[0]
                    
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
           
            
            r = _x * 1e4 / (1. + zobs)
            ok = np.logical_and(r >= rest_wave[0], r <= rest_wave[1])
            x = _x[ok==1]

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
                guess = np.array([ma, -2.5])                
            
            if ok.sum() == 2:
                print("WARNING: Estimating slope from only two points: {}".format(filt[isort][ok==1]))

        ##
        # Fit a PL to points.
        if method == 'fit':
            
            if len(x) < 2:
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
                        popt[:,i], pcov[:,:,i] = curve_fit(func, x, y[:,i], 
                            p0=guess[i], maxfev=1000)
                    except RuntimeError:
                        popt[:,i], pcov[:,:,i] = -99999, -99999
                                
            else:                    
                try:
                    popt, pcov = curve_fit(func, x, y, p0=guess)
                except RuntimeError:
                    popt, pcov = -99999, -99999
                        
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
            return popt[1]
        
    def ObserveSpectrum(self, zobs, spec=None, sfh=None, waves=None,
        flux_units='Hz', tarr=None, tobs=None, zarr=None, hist={}, 
        idnum=None, window=1, extras={}):
        """
        Take an input spectrum and "observe" it at redshift z.
        
        Parameters
        ----------
        wave : np.ndarray
            Rest wavelengths in [Angstrom]
        spec : np.ndarray
            Specific luminosities in [erg/s/A]
        z : int, float
            Redshift.
        
        Returns
        -------
        Observed wavelengths in microns, observed fluxes in erg/s/cm^2/Hz.
        
        """
        
        if spec is None:
            spec = self.Spectrum(sfh, waves, tarr=tarr, zarr=zarr, 
                zobs=zobs, tobs=None, hist=hist, idnum=idnum,
                extras=extras, window=window)
    
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
        
        if flux_units == 'Hz':
            pass
        else:
            f /= dwdn

        return waves * (1. + zobs) / 1e4, f

    def Photometry(self, spec=None, sfh=None, cam='wfc3', filters='all', 
        filter_set=None, dlam=10., rest_wave=None, extras={}, window=1,
        tarr=None, zarr=None, zobs=None, tobs=None, band=None, hist={},
        idnum=None, flux_units=None, picky=False):
        """
        Just a wrapper around `Spectrum`.

        Returns
        -------
        Tuple containing (in this order):
            - Names of all filters included
            - Midpoints of photometric filters [microns]
            - Width of filters [microns]
            - Apparent magnitudes in each filter.
            - Apparent magnitudes corrected for filter transmission.

        """
        
        assert (tobs is not None) or (zobs is not None)
        
        if zobs is None:
            zobs = self.cosm.z_of_t(tobs * s_per_myr)
                    
        # Get transmission curves
        filter_data = self.cameras[cam]._read_throughputs(filter_set=filter_set, 
            filters=filters)
        all_filters = filter_data.keys()    

        # Figure out spectral range we need to model for these filters.
        # Find bluest and reddest filters, set wavelength range with some
        # padding above and below these limits.
        lmin = np.inf
        lmax = 0.0
        ct = 0
        for filt in filter_data:
            x, y, cent, dx, Tavg, norm = filter_data[filt]
                                    
            # If we're only doing this for the sake of measuring a slope, we
            # might restrict the range based on wavelengths of interest, i.e.,
            # we may not use all the filters.
            
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
                        
            lmin = min(lmin, cent - dx[1])
            lmax = max(lmax, cent + dx[0])
            ct += 1
            
        # No filters in range requested    
        if ct == 0:    
            return [], [], [], [], []
                            
        # Convert from microns to Angstroms, undo redshift.
        lmin = lmin * 1e4 / (1. + zobs)
        lmax = lmax * 1e4 / (1. + zobs)
                
        lmin = max(lmin, self.src.wavelengths.min())
        lmax = min(lmax, self.src.wavelengths.max())
                                                
        # Here's our array of REST wavelengths
        waves = np.arange(lmin, lmax+dlam, dlam)
                
        # Get spectrum first.
        if spec is None:
            spec = self.Spectrum(sfh, waves, tarr=tarr, tobs=tobs,
                zarr=zarr, zobs=zobs, band=band, hist=hist,
                idnum=idnum, extras=extras, window=window)
            
        # Might be running over lots of galaxies
        batch_mode = False 
        if spec.ndim == 2:
            batch_mode = True    
                        
        # Observed wavelengths in micron, flux in erg/s/cm^2/Hz
        wave_obs, flux_obs = self.ObserveSpectrum(zobs, spec=spec, 
            waves=waves, extras=extras, window=window)
            
        # Convert microns to cm. micron * (m / 1e6) * (1e2 cm / m)
        freq_obs = c / (wave_obs * 1e-4)
                    
        # Why do NaNs happen? Just nircam. 
        flux_obs[np.isnan(flux_obs)] = 0.0
        
        # Loop over filters and re-weight spectrum
        xphot = []      # Filter centroids
        wphot = []      # Filter width
        yphot_obs = []  # Observed magnitudess [apparent]
        yphot_corr = [] # Magnitudes corrected for filter transmissions.
    
        # Loop over filters, compute fluxes in band (accounting for 
        # transmission fraction) and convert to observed magnitudes.
        for filt in all_filters:

            x, T, cent, dx, Tavg, dHz = filter_data[filt]
            
            if rest_wave is not None:
                cent_r = cent * 1e4 / (1. + zobs)
                if (cent_r < rest_wave[0]) or (cent_r > rest_wave[1]):
                    continue
                        
            # Re-grid transmission onto provided wavelength axis.
            T_regrid = np.interp(wave_obs, x, T, left=0, right=0)
                 
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
            
            #corr = np.sum(T_regrid[0:-1] * -1. * np.diff(freq_obs), axis=-1)
                                                                               
            xphot.append(cent)
            yphot_obs.append(_yphot / dHz)
            yphot_corr.append(_yphot / Tavg / dHz)
            wphot.append(dx)
        
        xphot = np.array(xphot)
        wphot = np.array(wphot)
        yphot_obs = np.array(yphot_obs)
        yphot_corr = np.array(yphot_corr)
        
        # Convert to magnitudes and return
        return all_filters, xphot, wphot, -2.5 * np.log10(yphot_obs / flux_AB), \
            -2.5 * np.log10(yphot_corr / flux_AB)
        
    def Spectrum(self, sfh, waves, tarr=None, zarr=None, window=1,
        zobs=None, tobs=None, band=None, idnum=None, units='Hz', hist={},
        extras={}):
        """
        This is just a wrapper around `Luminosity`.
        """
        
        if sfh.ndim == 2 and idnum is not None:
            sfh = sfh[idnum,:]
        
        batch_mode = sfh.ndim == 2
        time_series = (zobs is None) and (tobs is None)
        
        # Shape of output array depends on some input parameters
        shape = []
        if batch_mode:
            shape.append(sfh.shape[0])
        if time_series:
            shape.append(tarr.size)
        shape.append(len(waves))
            
        spec = np.zeros(shape)
        for i, wave in enumerate(waves):
            slc = (Ellipsis, i) if (batch_mode or time_series) else i
            
            spec[slc] = self.Luminosity(sfh, wave=wave, tarr=tarr, zarr=zarr,
                zobs=zobs, tobs=tobs, band=band, hist=hist, idnum=idnum,
                extras=extras, window=window)
                            
                
        if units in ['A', 'Ang']:
            #freqs = c / (waves / 1e8)
            #tmp = np.abs(np.diff(waves) / np.diff(freqs))
            #dwdn = np.concatenate((tmp, [tmp[-1]]))
            dwdn = waves**2 / (c * 1e8)
            spec /= dwdn    
        
        return spec
        
    def Magnitude(self, sfh, wave=1600., tarr=None, zarr=None, window=1,
        zobs=None, tobs=None, band=None, idnum=None, hist={}, extras={}):
        
        L = self.Luminosity(sfh, wave=wave, tarr=tarr, zarr=zarr, 
            zobs=zobs, tobs=tobs, band=band, idnum=idnum, hist=hist, 
            extras=extras, window=window)
        
        MAB = self.magsys.L_to_MAB(L, z=zobs)
        
        return MAB    

    def Luminosity(self, sfh, wave=1600., tarr=None, zarr=None, window=1,
        zobs=None, tobs=None, band=None, idnum=None, hist={}, extras={}):
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
        window : int, float
            Average over interval about `wave`? [Angstrom]
        zobs : int, float   
            If supplied, luminosity will be return only for an observation 
            at this redshift.
        tobs : int, float   
            If supplied, luminosity will be return only for an observation 
            at this time.
        hist : dict
            Extra information we may need, e.g., metallicity, dust optical 
            depth, etc. to compute spectrum.
        
        Returns
        -------
        Luminosity at wavelength=`wave` in units of erg/s/Hz.
        
        
        """
        
        if sfh.ndim == 2 and idnum is not None:
            sfh = sfh[idnum,:]
                
        # If SFH is 2-D it means we're doing this for multiple galaxies at once.
        # The first dimension will be number of galaxies and second dimension
        # is time/redshift.
        batch_mode = sfh.ndim == 2
                
        # Parse time/redshift information
        if zarr is not None:
            assert tarr is None
            
            tarr = self.cosm.t_of_z(zarr) / s_per_myr
        else:
            zarr = self.cosm.z_of_t(tarr * s_per_myr)
            
        assert np.all(np.diff(tarr) > 0), \
            "Must supply SFH in time-ascending (i.e., redshift-descending) order!"
        
        # Convert tobs to redshift.        
        if tobs is not None:
            zobs = self.cosm.z_of_t(tobs * s_per_myr)
            assert tarr.min() <= tobs <= tarr.max(), \
                "Requested time of observation (`tobs`) not in supplied range!"
            
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
                
            assert zarr.min() <= zobs <= zarr.max(), \
                "Requested time of observation (`tobs`) not in supplied range!"
                                
        fill = np.zeros(1)
        tyr = tarr * 1e6
        dt = np.hstack((np.diff(tyr), fill))
        
        # Figure out if we need to over-sample the grid we've got to more
        # accurately solve for young stellar populations.
        oversample = (zobs is not None) \
            and self.pf['pop_ssp_oversample'] \
            and (dt[-2] >= 2e6)
        ##
        # Done parsing time/redshift
        
        # Is this luminosity in some bandpass or monochromatic?
        if band is not None:
            Loft = self.src.IntegratedEmission(band[0], band[1])
            #raise NotImplemented('help!')
            print("Note this now has different units.")
        else:
            Loft = self.src.L_per_SFR_of_t(wave, avg=window)
        #

        # Setup interpolant for luminosity as a function of SSP age.      
        _func = interp1d(np.log(self.src.times), np.log(Loft),
            kind='cubic', bounds_error=False, 
            fill_value=Loft[-1])
            
        # Extrapolate linearly at times < 1 Myr
        _m = (Loft[1] - Loft[0]) \
          / (self.src.times[1] - self.src.times[0])
        L_small_t = lambda age: _m * age + Loft[0]
        
        #L_small_t = Loft[0]
        
        # Extrapolate as PL at t < 1 Myr based on first two
        # grid points
        #m = np.log(Loft[1] / Loft[0]) \
        #  / np.log(self.src.times[1] / self.src.times[0])
        #func = lambda age: np.exp(m * np.log(age) + np.log(Loft[0]))
                
        #if zobs is None:
        Lhist = np.zeros_like(sfh)
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
            if (zobs is not None):
                if (zarr[i] > zobs):
                    continue

            # If we made it here, it's time to integrate over star formation
            # at previous times. First, retrieve ages of stars formed in all 
            # past star forming episodes.
            ages = tarr[i] - tarr[0:i+1]

            # Treat metallicity evolution? If so, need to grab luminosity as 
            # function of age and Z.
            if self.pf['pop_enrichment']:

                if batch_mode:
                    Z = hist['Z'][slc][:,0:i+1]
                else:
                    Z = hist['Z'][slc][0:i+1]

                logA = np.log10(ages)
                logZ = np.log10(Z)
                L_per_msun = self.L_of_Z_t(wave)(logA, logZ, grid=False)
                                        
                # erg/s/Hz
                if batch_mode:
                    Lall = L_per_msun * sfh[:,0:i+1]
                else:
                    Lall = L_per_msun * sfh[0:i+1]
                    
                if oversample:
                    raise NotImplemented('help!')    
                else:
                    _dt = dt
                    
                _ages = ages    
            else:    
                
                # If time resolution is >= 2 Myr, over-sample final interval.
                if oversample:
                                        
                    # Use 1 Myr time resolution for final stretch.
                    # Treat 'final stretch' as free parameter? 10 Myr? 20?
                    
                    ct = 0
                    while (ages[-2-ct] - ages[-1]) < self.pf['pop_ssp_oversample_age']:
                        ct += 1
                    
                    ifin = -1 - ct
                                        
                    extra = np.arange(ages[-1], ages[ifin], 1.)[-1::-1]
                    
                    # Must augment ages and dt accordingly
                    _ages = np.hstack((ages[0:ifin], extra))
                    _dt = np.abs(np.diff(_ages) * 1e6)
                    
                    # 
                    _dt = np.hstack((_dt, [0]))
                    
                    # Now, compute luminosity at expanded ages.
                    L_per_msun = np.exp(_func(np.log(_ages)))    
                    
                    #L_per_msun = np.exp(np.interp(np.log(_ages), 
                    #    np.log(self.src.times), np.log(Loft), 
                    #    left=-np.inf, right=np.log(Loft[-1])))
                        
                    # Interpolate linearly at t < 1 Myr    
                    func = lambda age: age * Loft[0]
                    L_per_msun[_ages < 1] = func(_ages[_ages < 1])    
                    
                    # Must reshape SFR to match. Assume constant SFR within
                    # over-sampled integral.                    
                    #xSFR = SFR[:,ifin:] * np.ones((SFR.shape[0], extra.size))
                    
                    # Must allow non-constant SFR within over-sampled region
                    # as it may be tens of Myr
                    xSFR = np.ones((sfh.shape[0], extra.size))
                    
                    # Walk back from the end and fill in SFRs
                    N = extra.size / (ct + 1)
                    for _i in range(0, ct+1):                     
                        _fill = sfh[:,ifin+_i]
                        fill = np.reshape(np.repeat(_fill, N), (sfh.shape[0], N))
                        
                        #print(_i, xSFR[:,_i*N:(_i+1)*N].shape, fill.shape)
                        xSFR[:,_i*N:(_i+1)*N] = fill.copy()
                            
                        #print(_i, fill[0], fill.shape)
                        #raw_input('<enter>')

                    if batch_mode:
                        _SFR = np.hstack((sfh[:,0:ifin], xSFR))
                    else:
                        print('help')
                        _SFR = np.hstack((sfh[0:i], SFR[i+1] * np.ones_like(extra)))
                    
                    # erg/s/Hz
                    if batch_mode:
                        Lall = L_per_msun * _SFR
                    else:    
                        Lall = L_per_msun * _SFR
                                                                 
                else:    
                    L_per_msun = np.exp(np.interp(np.log(ages), 
                        np.log(self.src.times), np.log(Loft), 
                        left=np.log(Loft[0]), right=np.log(Loft[-1])))
                    
                    _dt = dt
                                                
                    # Fix early time behavior
                    L_per_msun[ages < 1] = L_small_t(ages[ages < 1])
        
                    _ages = ages
        
                    # erg/s/Hz
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

            # Integrate over all times up to this tobs            
            if batch_mode:
                if (zobs is not None):
                    Lhist = np.trapz(Lall[:,0:i+1], dx=_dt[0:i], axis=1)
                else:
                    Lhist[:,i] = np.trapz(Lall, dx=_dt[0:i], axis=1)
            else:
                if (zobs is not None):
                    Lhist = np.trapz(Lall, x=tyr[0:i+1])                
                else:    
                    Lhist[i] = np.trapz(Lall, x=tyr[0:i+1])                

            ##
            # In this case, we only need one iteration of this loop.
            ##
            if zobs is not None:
                break
                                   
        ##
        # Redden spectra
        ##
        if 'Sd' in hist:
            
            # Redden away!        
            if np.any(hist['Sd'] > 0) and (band is None):
                # Reddening is binary and probabilistic
                                
                assert 'kappa' in extras
                
                kappa = extras['kappa'](wave=wave)
                                                                
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
                    Lout = Lhist * clear[-1] \
                         + Lhist * np.exp(-tau[-1]) * block[-1]
                else:    
                    Lout = Lhist * clear[:,-1] \
                         + Lhist * np.exp(-tau[:,-1]) * block[:,-1]
                                 
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
            
            if 'children' in hist:
                if (hist['children'] is not None) and do_mergers:
                    
                    child_iz, child_iM = children.T
                
                    is_central = child_iM == -1
                                             
                    if np.all(is_central == 1):
                        pass
                    else:    
                    
                        print("Looping over {} halos...".format(sfh.shape[0]))
                                    
                        pb = ProgressBar(sfh.shape[0])
                        pb.start()
                
                        # Loop over all 'branches'
                        for i in range(SFR.shape[0]):
                                            
                            # This means the i'th halo is alive and well at the
                            # final redshift, i.e., it's a central
                            if is_central[i]:
                                continue
                             
                            pb.update(i)
                                                                    
                            # At this point, need to figure out which child halos
                            # to dump mass and SFH into...    
                            
                            # Be careful with redshift array. 
                            # We're now working in ascending time, reverse redshift,
                            # so we need to correct the child iz values. We've also
                            # chopped off elements at z < zobs.
                            #iz = Nz0 - child_iz[i]
                            
                            # This `iz` should not be negative despite us having
                            # chopped up the redshift array since getting to this
                            # point in the loop is predicated on being a parent of
                            # another halo, i.e., not surviving beyond this redshift. 
                            
                            # Lout is just 1-D at this point, i.e., just luminosity
                            # *now*. 
                                            
                            # Add luminosity to child halo. Zero out luminosity of 
                            # parent to avoid double counting. Note that nh will
                            # also have been zeroed out but we're just being careful.
                            Lout[child_iM[i]] += 1 * Lout[i]
                            Lout[i] = 0.0
                        
                        pb.finish()
                                    
        # Get outta here.
        return Lout
        
    def get_beta(self, z, data, filter_set='W'):
        filt, xphot, wphot, mag, magcorr = \
            self.photometrize_spectrum(data, z, flux_units=True, 
            filter_set=filter_set)
    
        # Need to sort first.
        iphot = np.argsort(xphot)
    
        xphot = xphot[iphot]
        wphot = wphot[iphot]
        mag = mag[iphot]
        magcorr = magcorr[iphot]
        all_filt = np.array(filt)[iphot]

        flux = 10**(magcorr * -0.4)

        slopes = {}
        for i, filt in enumerate(xphot):
        
            if i == (len(xphot) - 1):
                break
            
            beta = np.log10(flux[i+1] / flux[i]) \
                 / np.log10(xphot[i+1] / xphot[i])
        
            midpt = np.mean([xphot[i], xphot[i+1]])
        
            # In Angstrom
            rest_wave = midpt * (1. / 1e6) * 1e10 / (1. + z)
                
            slopes[(all_filt[i], all_filt[i+1])] = midpt, beta
        
        
        return slopes    