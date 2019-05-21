"""

Survey.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Sat 16 Feb 2019 10:40:18 EST

Description: 

"""

import os
import copy
import numpy as np
import matplotlib.pyplot as pl
from ..physics.Constants import c
from ..physics.Cosmology import Cosmology

try:
    import ares
except ImportError:
    pass

flux_AB = 3631. * 1e-23 # 3631 * 1e-23 erg / s / cm**2 / Hz
nanoJ = 1e-23 * 1e-9

_path = os.environ.get('ARES') + '/input'

class Survey(object):
    def __init__(self, cam='nircam', mod='modA', chip=1):
        self.camera = cam
        self.chip = chip
        
        if cam == 'nircam':
            self.path = '{}/nircam/nircam_throughputs/{}/filters_only'.format(_path, mod)
        elif cam == 'wfc3':
            self.path = '{}/wfc3'.format(_path)
        elif cam == 'wfc':
            self.path = '{}/wfc'.format(_path)
        else:
            raise NotImplemented('Unrecognized camera \'{}\''.format(cam))
    @property
    def cosm(self):
        if not hasattr(self, '_cosm'):
            self._cosm = Cosmology()
        return self._cosm
    
    @property
    def src(self):
        if not hasattr(self, '_src'):
            from ares.sources import SynthesisModel
            self._src = SynthesisModel(source_sed='eldridge2009')
            print("Defaulting to BPASS v1 source model.")
        return self._src
            
    @src.setter
    def src(self, value):
        self._src = value
    
    @property
    def wavelengths(self):
        """
        Wavelength array [Angstrom] in REST frame of sources.
        """
        if not hasattr(self, '_wavelengths'):
            self._wavelengths = self.src.wavelengths
        return self._wavelengths
    
    @wavelengths.setter
    def wavelengths(self, value):
        self._wavelengths = value
        
    @property
    def frequencies(self):
        if not hasattr(self, '_frequencies'):
            self._frequencies = c / (self.wavelengths / 1e8)
        return self._frequencies    
        
    @property
    def dwdn(self):
        if not hasattr(self, '_dwdn'):
            tmp = np.abs(np.diff(self.wavelengths) / np.diff(self.frequencies))
            self._dwdn = np.concatenate((tmp, [tmp[-1]]))
        return self._dwdn
    
    def PlotFilters(self, ax=None, fig=1, filter_set='W', annotate=True):
        """
        Plot transmission curves for NIRCAM filters.
        """
        
        if ax is None:
            fig = pl.figure(fig, figsize=(6, 6))
            ax = fig.add_subplot(111)
            gotax = False
        else:
            gotax = True
        
        data = self._read_throughputs(filter_set)
        
        colors = ['k', 'b', 'c', 'm', 'y', 'r', 'orange', 'g'] * 10
        for i, filt in enumerate(data.keys()):
            
            ax.plot(data[filt][0], data[filt][1], label=filt, color=colors[i])
            
            if annotate:
                if filt.endswith('IR'):
                    _filt = filt[0:-3]
                else:    
                    _filt = filt
                    
                ax.annotate(_filt, (data[filt][2], 0.8), ha='center', va='top',
                    color=colors[i], rotation=90)
            
        ax.set_xlabel(r'Observed Wavelength $[\mu \mathrm{m}]$')    
        ax.set_ylabel('Transmission')
        ax.set_ylim(-0.05, 1.05)
        #ax.legend(loc='best', frameon=False, fontsize=10, ncol=2)
            
        return ax
    
    def _read_throughputs(self, filter_set='W'):
        if self.camera == 'nircam':
            return self._read_nircam(filter_set)
        elif self.camera == 'wfc3':
            return self._read_wfc3(filter_set)        
        elif self.camera == 'wfc':
            return self._read_wfc(filter_set)    
        else:
            raise NotImplemented('help')
            
    def _read_nircam(self, filter_set='W'):

        if not hasattr(self, '_filter_cache'):
            self._filter_cache = {}
            
        if type(filter_set) != list:
            filter_set = [filter_set]

        data = {}
        for fn in os.listdir(self.path):
                    
            pre = fn.split('_')[0]
            
            for _filters in filter_set:
                
                if _filters in self._filter_cache:
                    data[pre] = self._filter_cache[_filters]
                    continue
                    
                if _filters not in pre:
                    continue        
        
                # Need to distinguish W from W2
                if (_filters == 'W') and ('W2' in pre):
                    continue
                                                                        
                # Determine the center wavelength of the filter based its string
                # identifier.    
                k = pre.rfind(_filters)    
                cent = float('{}.{}'.format(pre[1], pre[2:k]))
                
                # Wavelength [micron], transmission
                x, y = np.loadtxt('{}/{}'.format(self.path, fn), unpack=True, 
                    skiprows=1)
                
                data[pre] = self._get_filter_prop(x, y, cent)
            
                self._filter_cache[pre] = copy.deepcopy(data[pre])
        
        return data   
        
    def _read_wfc(self, filter_set='W'):
        if not hasattr(self, '_filter_cache'):
            self._filter_cache = {}
            
        if type(filter_set) != list:
            filter_set = [filter_set]    
            
        data = {}
        for fn in os.listdir(self.path):
        
            pre = fn.split('wfc_')[1].split('.dat')[0]
            
            for _filters in filter_set:
            
                if _filters in self._filter_cache:
                    data[pre] = self._filter_cache[_filters]
                    continue
                    
                if _filters not in pre:
                    continue
                
                # Determine the center wavelength of the filter based on its string
                # identifier.    
                k = pre.rfind(_filters)
                cent = float('0.{}'.format(pre[1:k]))
                
                x, y = np.loadtxt('{}/{}'.format(self.path, fn), 
                    unpack=True, skiprows=1)
                                            
                # Convert wavelengths from nanometers to microns
                data[pre] = self._get_filter_prop(x / 1e4, y, cent)

                self._filter_cache[pre] = copy.deepcopy(data[pre])
        
        return data    
                 
    def _read_wfc3(self, filter_set='W'):
        if not hasattr(self, '_filter_cache'):
            self._filter_cache = {}
            
        if type(filter_set) != list:
            filter_set = [filter_set]    
            
        data = {}
        for fn in os.listdir(self.path+'/IR'):
        
            pre = fn.split('_throughput')[0]
            
            for _filters in filter_set:
            
                if _filters in self._filter_cache:
                    data[pre] = self._filter_cache[_filters]
                    continue
                    
                if _filters not in pre:
                    continue
        
                # Determine the center wavelength of the filter based on its string
                # identifier.    
                k = pre.rfind(_filters)
                cent = float('{}.{}'.format(pre[1], pre[2:k]))
                
                _i, x, y = np.loadtxt('{}/IR/{}'.format(self.path, fn), 
                    unpack=True, skiprows=1, delimiter=',')
                            
                # Convert wavelengths from Angstroms to microns
                data[pre] = self._get_filter_prop(x / 1e4, y, cent)

                self._filter_cache[pre] = copy.deepcopy(data[pre])
        
        return data
        
    def _get_filter_prop(self, x, y, cent):    
        Tmax = max(y)
        hmax = y > 0.5 * Tmax
                
        # Compute width of filter
        hi = max(x[hmax == True])
        lo = min(x[hmax == True])
        mi = np.mean(x[hmax == True])
        dx = np.array([hi - mi, mi - lo])
        
        ok = np.logical_and(x >= lo, x <= hi)
        Tavg = np.mean(y[ok==1])
        
        # Get the Hz^-1 units back
        norm = c / (lo * 1e-4) - c / (hi * 1e-4)
        
        return x, y, cent, dx, Tavg, norm
        
        
    def observe_spectrum(self, spec, z):
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
    
        dL = self.cosm.LuminosityDistance(z)
    
        # Flux at Earth in erg/s/cm^2/Hz
        f = spec * self.dwdn / (4. * np.pi * dL**2)
    
        return self.wavelengths * (1. + z) / 1e4, f

    def photometrize_spectrum(self, spec, z, filter_set='W', flux_units=None):
        """
        Take a spectrum and determine what it would look like to NIRCAM.
    
        Returns
        -------
        Tuple containing (in this order):
            - Midpoints of photometric filters [microns]
            - Width of filters [microns]
            - Apparent magnitudes in each filter.
            - Apparent magnitudes correctedf or filter transmission.
    
        """
    
        data = self._read_throughputs(filter_set)
    
        # Observed wavelengths in micron, flux in erg/s/cm^2/Hz
        wave_obs, flux_obs = self.observe_spectrum(spec, z)
            
        # Convert microns to cm. micron * (m / 1e6) * (1e2 cm / m)
        freq_obs = c / (wave_obs * 1e-4)
        
        if flux_units is not None:
            _diff = np.diff(freq_obs) / np.diff(wave_obs)
            dndw = np.hstack((_diff, [0]))
        
            flux_obs *= dndw
        
        # Convert microns to Ang. micron * (m / 1e6) * (1e10 A / m)
        tmp = np.abs(np.diff(freq_obs) / np.diff(wave_obs * 1e4))
        dndw = np.concatenate((tmp, [tmp[-1]]))

        # Loop over filters and re-weight spectrum
        xphot = []      # Filter centroids
        wphot = []      # Filter width
        yphot_obs = []  # Observed magnitudess [apparent]
        yphot_corr = [] # Magnitudes corrected for filter transmissions.
    
        # Loop over filters, compute fluxes in band (accounting for 
        # transmission fraction) and convert to observed magnitudes.
        all_filters = data.keys()
        
        for filt in all_filters:

            x, y, cent, dx, Tavg, norm = data[filt]

            # Re-grid transmission onto provided wavelength axis.
            filt_regrid = np.interp(wave_obs, x, y, left=0, right=0)

            # Observed flux is in erg/s/cm^2/Hz
            _yphot = np.abs(np.trapz(filt_regrid * flux_obs, x=freq_obs))            
        
            xphot.append(cent)    
            yphot_obs.append(_yphot / norm)
            yphot_corr.append(_yphot / norm / Tavg)
            wphot.append(dx)
                            
        
        xphot = np.array(xphot)
        wphot = np.array(wphot)
        yphot_obs = np.array(yphot_obs)
        yphot_corr = np.array(yphot_corr)
        
        return all_filters, xphot, wphot, -2.5 * np.log10(yphot_obs / flux_AB), \
            -2.5 * np.log10(yphot_corr / flux_AB)
        
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