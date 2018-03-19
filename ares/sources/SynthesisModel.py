"""

SynthesisModel.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Mon Apr 11 11:27:45 PDT 2016

Description: 

"""

import numpy as np
from .Source import Source
from ares.physics import Cosmology
from scipy.optimize import minimize
from ..util.ReadData import read_lit
from ..util.Math import interp1d
from ..util.ParameterFile import ParameterFile
from ares.physics.Constants import h_p, c, erg_per_ev, g_per_msun, s_per_yr, \
    s_per_myr, m_H, ev_per_hz

class DummyClass(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.Nt = 11
    
    @property 
    def times(self):
        return np.linspace(0, 500, self.Nt)
    
    @property
    def weights(self):
        return np.ones_like(self.times)
    
    def _load(self, **kwargs):
        # Energies must be in *descending* order
        if np.all(np.diff(kwargs['pop_E']) > 0):
            E = kwargs['pop_E'][-1::-1]
            L = kwargs['pop_L'][-1::-1]
        else:
            E = kwargs['pop_E']
            L = kwargs['pop_L'] 

        data = np.array([L] * self.Nt).T
        wave = 1e8 * h_p * c / (E * erg_per_ev)
        
        assert len(wave) == data.shape[0], "len(pop_L) must == len(pop_E)."
                
        return wave, data
        
class SynthesisModel(Source):
    def __init__(self, **kwargs):
        self.pf = ParameterFile(**kwargs)
                
    @property
    def litinst(self):
        if not hasattr(self, '_litinst'):
            if self.pf['source_sed'] == 'user':
                self._litinst = DummyClass()
            else:    
                self._litinst = read_lit(self.pf['source_sed'])
                
        return self._litinst
    
    @property
    def cosm(self):
        if not hasattr(self, '_cosm'):
            self._cosm = Cosmology(**self.pf)
        return self._cosm
        
    def AveragePhotonEnergy(self, Emin, Emax):
        """
        Return average photon energy in supplied band.
        """
    
        j1 = np.argmin(np.abs(Emin - self.energies))
        j2 = np.argmin(np.abs(Emax - self.energies))
        
        E = self.energies[j2:j1][-1::-1]
        
        # Units: erg / s / Hz
        to_int = self.Spectrum(E)
         
        # Units: erg / s
        return np.trapz(to_int * E, x=E) / np.trapz(to_int, x=E)

    def Spectrum(self, E):
        """
        Return a normalized version of the spectrum at photon energy E / eV.
        """
        # reverse energies so they are in ascending order
        nrg = self.energies[-1::-1]
        return np.interp(E, nrg, self.sed_at_tsf[-1::-1]) / self.norm

    @property
    def sed_at_tsf(self):
        if not hasattr(self, '_sed_at_tsf'):
            # erg / s / Hz -> erg / s / eV
            if self.pf['source_rad_yield'] == 'from_sed':
                self._sed_at_tsf = \
                    self.data[:,self.i_tsf] * self.dwdn / ev_per_hz
            else:
                self._sed_at_tsf = self.data[:,self.i_tsf]
                    
        return self._sed_at_tsf

    @property
    def dE(self):
        if not hasattr(self, '_dE'):
            tmp = np.abs(np.diff(self.energies))
            self._dE = np.concatenate((tmp, [tmp[-1]]))
        return self._dE

    @property
    def dndE(self):
        if not hasattr(self, '_dndE'):
            tmp = np.abs(np.diff(self.frequencies) / np.diff(self.energies))
            self._dndE = np.concatenate((tmp, [tmp[-1]]))
        return self._dndE
    
    @property
    def dwdn(self):
        if not hasattr(self, '_dwdn'):
            tmp = np.abs(np.diff(self.wavelengths) / np.diff(self.frequencies))
            self._dwdn = np.concatenate((tmp, [tmp[-1]]))
        return self._dwdn

    @property
    def norm(self):
        """
        Normalization constant that forces self.Spectrum to have unity
        integral in the (Emin, Emax) band.
        """
        if not hasattr(self, '_norm'):
            # Note that we're not using (EminNorm, EmaxNorm) band because
            # for SynthesisModels we don't specify luminosities by hand. By
            # using (EminNorm, EmaxNorm), we run the risk of specifying a 
            # range not spanned by the model.
            j1 = np.argmin(np.abs(self.Emin - self.energies))
            j2 = np.argmin(np.abs(self.Emax - self.energies))
            
            # Remember: energy axis in descending order
            self._norm = np.trapz(self.sed_at_tsf[j2:j1][-1::-1], 
                x=self.energies[j2:j1][-1::-1])
                
        return self._norm
        
    @property
    def i_tsf(self):
        if not hasattr(self, '_i_tsf'):
            self._i_tsf = np.argmin(np.abs(self.pf['source_tsf'] - self.times))
        return self._i_tsf
    
    @property
    def data(self):
        """
        Units = erg / s / A / [depends]
        
        Where, if instantaneous burst, [depends] = 1e6 Msun
        and if continuous SF, [depends] = Msun / yr
        
        """
        if not hasattr(self, '_data'):
            
            Zall = np.sort(list(self.metallicities.values()))
                        
            # Check to see dimensions of tmp. Depending on if we're 
            # interpolating in Z, it might be multiple arrays.
            if (self.pf['source_Z'] in Zall):
                if self.pf['source_sed_by_Z'] is not None:
                    _tmp = self.pf['source_sed_by_Z'][1]
                    self._data = _tmp[np.argmin(np.abs(Zall - self.pf['source_Z']))]
                else:
                    self._wavelengths, self._data = \
                        self.litinst._load(**self.pf)
            else:
                if self.pf['source_sed_by_Z'] is not None:
                    _tmp = self.pf['source_sed_by_Z'][1]
                    assert len(_tmp) == len(Zall)
                else:
                    self._wavelengths, _tmp = \
                        self.litinst._load(**self.pf)

                # Shape is (Z, wavelength, time)?
                to_interp = np.array(_tmp)
                self._data_all_Z = to_interp
                
                # If outside table's range, just use endpoints
                if self.pf['source_Z'] > max(Zall):
                    _raw_data = np.log10(to_interp[-1])
                elif self.pf['source_Z'] < min(Zall):
                    _raw_data = np.log10(to_interp[0])
                else:
                    # If within range, interpolate
                    _raw_data = np.zeros_like(to_interp[0])
                    for i, t in enumerate(self.litinst.times):
                        inter = interp1d(np.log10(Zall), 
                            np.log10(to_interp[:,:,i]), axis=0, 
                            kind=self.pf['interp_Z'])
                        _raw_data[:,i] = inter(np.log10(self.pf['source_Z']))
                                                                
                self._data = 10**_raw_data
                
                # By doing the interpolation in log-space we sometimes
                # get ourselves into trouble in bins with zero flux. 
                # Gotta correct for that!
                self._data[np.argwhere(np.isnan(self._data))] = 0.0
                
            # Normalize by SFR or cluster mass.    
            if self.pf['source_ssp']:
                self._data *= (self.pf['source_mass'] / 1e6)
            else:    
                self._data *= self.pf['source_sfr']
                
        return self._data
    
    @property
    def wavelengths(self):
        if not hasattr(self, '_wavelengths'):
            if self.pf['source_sed_by_Z'] is not None:
                self._wavelengths, junk = self.pf['source_sed_by_Z']
            else:
                data = self.data
            
        return self._wavelengths

    @property
    def Nfreq(self):
        return len(self.energies)
    
    @property
    def E(self):
        if not hasattr(self, '_E'):
            self._E = np.sort(self.energies)
        return self._E

    @property
    def LE(self):
        """
        Should be dimensionless?
        """
        if not hasattr(self, '_LE'):
            if self.pf['source_ssp']:
                raise NotImplemented('No support for SSPs yet (due to t-dep)!')

            _LE = self.sed_at_tsf * self.dE / self.Lbol_at_tsf
            
            s = np.argsort(self.energies)
            self._LE = _LE[s]
            
        return self._LE

    @property
    def energies(self):
        if not hasattr(self, '_energies'):
            self._energies = h_p * c / (self.wavelengths / 1e8) / erg_per_ev
        return self._energies
        
    @property
    def Emin(self):
        return np.min(self.energies)
    
    @property
    def Emax(self):
        return np.max(self.energies)    
        
    @property
    def frequencies(self):
        if not hasattr(self, '_frequencies'):
            self._frequencies = c / (self.wavelengths / 1e8)
        return self._frequencies

    @property
    def weights(self):
        return self.litinst.weights  
        
    @property
    def times(self):
        return self.litinst.times
    
    @property
    def metallicities(self):
        return self.litinst.metallicities
        
    @property
    def time_averaged_sed(self):
        if not hasattr(self, '_tavg_sed'):
            self._tavg_sed = np.dot(self.data, self.weights) / self.times.max()
        
        return self._tavg_sed

    @property
    def emissivity_per_sfr(self):
        """
        Photon emissivity.
        """
        if not hasattr(self, '_E_per_M'):
            self._E_per_M = np.zeros_like(self.data)
            for i in xrange(self.times.size):
                self._E_per_M[:,i] = self.data[:,i] / (self.energies * erg_per_ev)    

        return self._E_per_M

    @property
    def uvslope(self):
        if not hasattr(self, '_uvslope'):
            self._uvslope = np.zeros_like(self.data)
            for i in xrange(self.times.size):
                self._uvslope[1:,i] = np.diff(np.log(self.data[:,i])) \
                    / np.diff(np.log(self.wavelengths))

        return self._uvslope
        
    def fit_uvslope(self, lam=1600, dlam=200):
        
        slc = np.logical_and(lam-dlam <= self.wavelengths, 
                            self.wavelengths <= lam+dlam)
        
        _uvslope_fit = np.zeros_like(self.times)        
        for i in range(self.times.size):
            
            logL = np.log(self.data[slc,i])
            logw = np.log(self.wavelengths[slc])

            model = lambda pars: pars[0] + pars[1] * logw

            to_min = lambda x, *args: np.sum((model(x) - logL)**2)

            res = minimize(to_min, np.array([42., -2.]))

            a, b = res.x

            _uvslope_fit[i] = b
            
        return _uvslope_fit    
    
    def LUV_of_t(self):
        return self.L_per_SFR_of_t()
    
    def L_per_SFR_of_t(self, wave=1600., avg=1):
        """
        UV luminosity per unit SFR.
        """
                
        j = np.argmin(np.abs(wave - self.wavelengths))
        
        dwavednu = np.diff(self.wavelengths) / np.diff(self.frequencies)
        
        if avg == 1:
            yield_UV = self.data[j,:] * np.abs(dwavednu[j])
        else:
            assert avg % 2 != 0, "avg must be odd"
            s = (avg - 1) / 2
            yield_UV = np.mean(self.data[j-s:j+s,:] * np.abs(dwavednu[j-s:j+s]))
        
        return yield_UV

    def LUV(self):
        return self.L_per_SFR_of_t()[-1]

    @property
    def L1600_per_sfr(self):
        return self.L_per_sfr()
        
    def L_per_sfr(self, wave=1600., avg=1):
        """
        Specific emissivity at provided wavelength.
        
        Parameters
        ----------
        wave : int, float
            Wavelength at which to determine emissivity.
        avg : int
            Number of wavelength bins over which to average
        
        Units are 
            erg / s / Hz / (Msun / yr)
        or 
            erg / s / Hz / Msun

        """
        
        yield_UV = self.L_per_SFR_of_t(wave)
            
        # Interpolate in time to obtain final LUV
        if self.pf['source_tsf'] in self.times:
            return yield_UV[np.argmin(np.abs(self.times - self.pf['source_tsf']))]
            
        k = np.argmin(np.abs(self.pf['source_tsf'] - self.times))
        if self.times[k] > self.pf['source_tsf']:
            k -= 1
            
        if not hasattr(self, '_LUV_interp'):
            self._LUV_interp = interp1d(self.times, yield_UV, kind='linear')
            
        return self._LUV_interp(self.pf['source_tsf'])
        
    def kappa_UV_of_t(self):        
        return 1. / self.LUV_of_t()
        
    def kappa_UV(self):    
        """
        Number of photons emitted per stellar baryon of star formation.
        
        If star formation is continuous, this will have units of:
            (Msun / yr) / (erg / s / Hz)
        If star formation is in a burst, this will have units of:
            Msun / (erg / s / Hz)
        Returns
        -------
        Two-dimensional array containing photon yield per unit stellar baryon per
        second per angstrom. First axis corresponds to photon wavelength (or energy), 
        and second axis to time.
        
        """
        
        return 1. / self.LUV()

    def integrated_emissivity(self, l0, l1, unit='A'):
        # Find band of interest -- should be more precise and interpolate
        
        if unit == 'A':
            x = self.wavelengths
            i0 = np.argmin(np.abs(x - l0))
            i1 = np.argmin(np.abs(x - l1))
        elif unit == 'Hz':
            x = self.frequencies
            i1 = np.argmin(np.abs(x - l0))
            i0 = np.argmin(np.abs(x - l1))
        
        # Current units: photons / sec / baryon / Angstrom      
        
        # Count up the photons in each spectral bin for all times
        photons_per_b_t = np.zeros_like(self.times)
        for i in xrange(self.times.size):
            photons_per_b_t[i] = np.trapz(self.emissivity_per_sfr[i1:i0,i], 
                x=x[i1:i0])
                
        t = self.times * s_per_myr 
      
    def erg_per_phot(self, Emin, Emax):
        return self.eV_per_phot(Emin, Emax) * erg_per_ev  

    def eV_per_phot(self, Emin, Emax):
        """
        Compute the average energy per photon (in eV) in some band.
        """
        
        if self.pf['source_ssp']:
            # Assume last time-bin below.
            raise NotImplemented('help!')
        
        i0 = np.argmin(np.abs(self.energies - Emin))
        i1 = np.argmin(np.abs(self.energies - Emax))

        it = -1  # time index
        
        # [self.data] = erg / s / A / [depends]

        # Must convert units
        E_tot = np.trapz(self.data[i1:i0,it] * self.wavelengths[i1:i0], 
            x=np.log(self.wavelengths[i1:i0]))
        N_tot = np.trapz(self.data[i1:i0,it] * self.wavelengths[i1:i0] \
            / self.energies[i1:i0] / erg_per_ev, 
            x=np.log(self.wavelengths[i1:i0]))

        return E_tot / N_tot / erg_per_ev

    def rad_yield(self, Emin, Emax):
        """
        Must be in the internal units of erg / g.
        """
        
        erg_per_msun_yr = \
           self.IntegratedEmission(Emin, Emax, energy_units=True)[-1]
        erg_per_g = erg_per_msun_yr * s_per_yr / g_per_msun
        
        return erg_per_g
        
    @property
    def Lbol_at_tsf(self):
        if not hasattr(self, '_Lbol_at_tsf'):
            self._Lbol_at_tsf = self.Lbol(self.pf['source_tsf'])
        return self._Lbol_at_tsf

    def Lbol(self, t):
        """
        Return bolometric luminosity at time `t`.
        
        Assume 1 Msun / yr SFR.
        """
        
        L = self.IntegratedEmission(energy_units=True)
        
        return np.interp(t, self.times, L)

    def IntegratedEmission(self, Emin=None, Emax=None, energy_units=False):
        """
        Compute photons emitted integrated in some band for all times.

        Returns
        -------
        Integrated flux between (Emin, Emax) for all times in units of 
        photons / sec / (Msun [/ yr]), unless energy_units=True, in which
        case its erg instead of photons.
        """

        # Find band of interest -- should be more precise and interpolate
        if Emin is None:
            Emin = np.min(self.energies)
        if Emax is None:
            Emax = np.max(self.energies)
            
        i0 = np.argmin(np.abs(self.energies - Emin))
        i1 = np.argmin(np.abs(self.energies - Emax))
        
        if i0 == i1:
            raise ValueError('Are EminNorm and EmaxNorm set properly?')

        # Count up the photons in each spectral bin for all times
        flux = np.zeros_like(self.times)
        for i in xrange(self.times.size):
            if energy_units:
                integrand = self.data[i1:i0,i] * self.wavelengths[i1:i0]
            else:
                integrand = self.data[i1:i0,i] * self.wavelengths[i1:i0] \
                    / (self.energies[i1:i0] * erg_per_ev)
                        
            flux[i] = np.trapz(integrand, x=np.log(self.wavelengths[i1:i0]))
                
        # Current units: 
        # if pop_ssp: photons / sec / (Msun / 1e6)
        # else: photons / sec / (Msun / yr)
        
        return flux
                
    @property
    def Nion(self):
        if not hasattr(self, '_Nion'):
            self._Nion = self.PhotonsPerBaryon(13.6, 24.6)
        return self._Nion
        
    @property
    def Nlw(self):
        if not hasattr(self, '_Nlw'):
            self._Nlw = self.PhotonsPerBaryon(10.2, 13.6)
        return self._Nlw    
        
    def PhotonsPerBaryon(self, Emin, Emax):    
        """
        Compute the number of photons emitted per unit stellar baryon.

        ..note:: This integrand over the provided band, and cumulatively over time.

        Parameters
        ----------
        Emin : int, float
            Minimum rest-frame photon energy to consider [eV].
        Emax : int, float
            Maximum rest-frame photon energy to consider [eV].

        Returns
        -------
        An array with the same dimensions as ``self.times``, representing the 
        cumulative number of photons emitted per stellar baryon of star formation
        as a function of time.

        """

        #assert self.pf['pop_ssp'], "Probably shouldn't do this for continuous SF."
        photons_per_s_per_msun = self.IntegratedEmission(Emin, Emax)    

        # Current units: 
        # if pop_ssp: 
        #     photons / sec / (Msun / 1e6)
        # else: 
        #     photons / sec / (Msun / yr)

        # Integrate (cumulatively) over time
        if self.pf['source_ssp']:
            photons_per_b_t = photons_per_s_per_msun / self.cosm.b_per_msun
            return np.trapz(photons_per_b_t, x=self.times*s_per_myr) / 1e6
        # Take steady-state result
        else:
            photons_per_b_t = photons_per_s_per_msun * s_per_yr \
                / self.cosm.b_per_msun
            
            # Return last element: steady state result
            return photons_per_b_t[-1]
                            
