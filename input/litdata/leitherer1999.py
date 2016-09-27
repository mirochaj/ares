"""
Leitherer, C., Schaerer, D., Goldader, J. D., Delgado, R. M. G., Robert, C.,
Kune, D. F., de Mello, D. F., Devost, D., & Heckman, T. M. 1999, ApJS,
123, 3

Notes
-----

"""

import re, os
import numpy as np
from ares.physics import Cosmology
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d, RectBivariateSpline
from ares.physics.Constants import h_p, c, erg_per_ev, g_per_msun, s_per_yr, \
    s_per_myr, m_H

_input = os.getenv('ARES') + '/input/starburst99/data'

metallicities = \
{
 'a': 0.040,
 'b': 0.020,
 'c': 0.008,
 'd': 0.004,
 'e': 0.001,
}

sf_laws = \
{
 'continuous': 1.0,       # solar masses per year
 'instantaneous': 1e6,    # solar masses
}

imf_options = [2.35, 3.3, 30.]

info = \
{
 'flux_units': r'$\mathrm{erg} \ \mathrm{s}^{-1} \ \AA^{-1}$',
}

pars = \
{
 'pop_Z': 0.04,
 'pop_imf': 2.35,
 'nebular': False,
 'pop_ssp': True,
 'pop_tsf': 500.,
}

_weights = np.array([1.] * 20 + [10.] * 8 + [100.] * 8)
times = np.cumsum(_weights)

def _reader(fn, skip=3, dtype=float):
    """
    Read output from starburst99.

    Parameters
    ----------
    fn : str
        Name of file within $ARES/input/starburst99 to open.
    skip : int
        Number of lines to skip at beginning of file. I don't think this should
        ever change.
        
    """

    f = open('%s/%s' % (_input, fn), 'r')

    data = []
    for i, line in enumerate(f):
        if i < skip:
            continue

        data.append(map(dtype, line.split()))

    return np.array(data)

def _fignum_to_figname():
    num, names = _reader('README', skip=18, dtype=str).T
    
    num = map(int, num)
    
    prefixes = []
    for name in names:
        if '*' in name:
            prefix = name.partition('*')[0]
        else:
            prefix = name.partition('.')[0]
            
        prefixes.append(prefix)
    
    return num, prefixes
    
fig_num, fig_prefix = _fignum_to_figname()
    
def _figure_name(pop_Z=0.04, pop_imf=2.35, pop_nebular=False, pop_ssp=True, 
    **kwargs):
    """
    Only built for figures 1-12 right now.
    
    Parameters
    ----------
    imf : float
        2.35
        3.3
        30
    """
    
    options = np.arange(1, 13)
    mask = np.ones_like(options)
    
    # Can't be odd
    if pop_ssp:
        mask[options % 2 == 0] = 0
    else:
        mask[options % 2 == 1] = 0
    
    # Can't be > 6
    if pop_nebular:
        mask[options > 6] = 0
    else:
        mask[options <= 6] = 0
    
    if pop_imf == 2.35:
        for i in options:
            if i not in [1,2,7,8]:
                mask[i-1] *= 0
    elif pop_imf == 3.3:
        for i in options:
            if i not in [3,4,9,10]:
                mask[i-1] *= 0  
    elif pop_imf == 30:
        for i in options:
            if i not in [5,6,11,12]:
                mask[i-1] *= 0
                      
    Zvals = metallicities.values()
    
    if pop_Z not in Zvals:
        raise ValueError('Unrecognized metallicity.')
        
    Z_suffix = metallicities.keys()[Zvals.index(pop_Z)]
    
    if mask.sum() > 1:
        raise ValueError('Ambiguous SED.')
    
    j = options[mask == 1]
    k = fig_num.index(j)
    
    return '%s%s.dat' % (fig_prefix[k], Z_suffix)
    
def _load(**kwargs):
    """
    Return wavelengths, fluxes, for given set of parameters (at all times).
    """
    Zvals = np.sort(metallicities.values())
            
    if kwargs['pop_Z'] not in Zvals:
        
        data = []
        for Z in Zvals:
            tmp = kwargs.copy()
            tmp['pop_Z'] = Z
            fn = _figure_name(**tmp)
            _data = _reader(fn)
            data.append(_data[:,1:])
        
        # Has dimensions (metallicity, wavelengths, times)
        data_3d = np.array(data)
        
        # Same for all metallicities
        wavelengths = _data[:,0]
        
        _raw_data = np.zeros_like(_data)
        for i, t in enumerate(times):
            # Data in this case is already in log10
            interp = RectBivariateSpline(np.log10(Zvals), np.log10(wavelengths), 
                data_3d[:,:,i])
            _raw_data[:,i] = \
                interp(np.log10(kwargs['pop_Z']), np.log10(wavelengths))
                 
        data = 10**_raw_data
                    
    else:        
        fn = _figure_name(**kwargs)
        _raw_data = _reader(fn)
        wavelengths = _raw_data[:,0]
        data = 10**_raw_data[:,1:]
        
    return wavelengths, data    
        
# Wavelengths in Angstroms (ascending)
#wave = s99_data[:,0]
#
## For time-integral
#weights = np.array([1] * 20 + [10] * 8 + [100] * 8)

#class StellarPopulation(object):
#    def __init__(self, **kwargs):
#        self.pf = pars.copy()
#        self.pf.update(kwargs)
#    
#        self._load()
#    
#    def _load(self):
#        
#        Zvals = np.sort(metallicities.values())
#                
#        if self.pf['pop_Z'] not in Zvals:
#            
#            pf = self.pf.copy()
#            data = []
#            for Z in Zvals:
#                pf['pop_Z'] = Z
#                fn = _figure_name(**pf)
#                _raw_data = _reader(fn)
#                data.append(_raw_data[:,1:])
#            
#            data = np.array(data)
#            self._wavelengths = _raw_data[:,0]
#            
#            self._raw_data = np.zeros_like(_raw_data)
#            for i, t in enumerate(self.times):
#                interp = RectBivariateSpline(np.log10(Zvals), np.log10(self.wavelengths), 
#                    data[:,:,i])
#                self._raw_data[:,i] = \
#                    interp(np.log10(self.pf['pop_Z']), np.log10(self.wavelengths))
#                     
#            self._data = 10**self._raw_data         
#                        
#        else:        
#            self.fn = _figure_name(**self.pf)
#            self._raw_data = _reader(self.fn)
#    
#    @property
#    def data(self):
#        """
#        Units = erg / s / A / [depends]
#        
#        Where, if instantaneous burst, [depends] = Msun
#        and if continuous SF, [depends] = Msun / yr
#        
#        """
#        if not hasattr(self, '_data'):
#            self._data = 10**self._raw_data[:,1:]
#        return self._data
#    
#    @property
#    def cosm(self):
#        if not hasattr(self, '_cosm'):
#            self._cosm = Cosmology()
#            
#        return self._cosm
#    
#    @property
#    def wavelengths(self):
#        if not hasattr(self, '_wavelengths'):
#            self._wavelengths = self._raw_data[:,0]
#            
#        return self._wavelengths
#        
#    @property
#    def energies(self):
#        if not hasattr(self, '_energies'):
#            self._energies = h_p * c / (self.wavelengths / 1e8) / erg_per_ev
#    
#        return self._energies    
#        
#    @property
#    def frequencies(self):
#        if not hasattr(self, '_frequencies'):
#            self._frequencies = c / (self.wavelengths / 1e8)
#    
#        return self._frequencies
#
#    @property
#    def weights(self):
#        if not hasattr(self, '_weights'):
#            self._weights = np.array([1.] * 20 + [10.] * 8 + [100.] * 8)
#    
#        return self._weights    
#        
#    @property
#    def times(self):
#        if not hasattr(self, '_times'):
#            self._times = np.cumsum(self.weights)
#        
#        return self._times
#        
#    @property
#    def time_averaged_sed(self):
#        if not hasattr(self, '_tavg_sed'):
#            self._tavg_sed = np.dot(self.data, self.weights) / self.times.max()
#        
#        return self._tavg_sed
#
#    @property
#    def emissivity_per_sfr(self):
#        """
#        Photon emissivity.
#        """
#        if not hasattr(self, '_E_per_M'):
#            self._E_per_M = np.zeros_like(self.data)
#            for i in range(self.times.size):
#                self._E_per_M[:,i] = self.data[:,i] / (self.energies * erg_per_ev)    
#
#            if self.pf['pop_ssp']:
#                self._E_per_M /= 1e6
#            else:
#                pass
#
#        return self._E_per_M
#
#    @property
#    def uvslope(self):
#        if not hasattr(self, '_uvslope'):
#            self._uvslope = np.zeros_like(self.data)
#            for i in range(self.times.size):
#                self._uvslope[1:,i] = np.diff(np.log(self.data[:,i])) \
#                    / np.diff(np.log(self.wavelengths))
#
#        return self._uvslope
#    
#    def LUV_of_t(self):
#        return self.L_per_SFR_of_t()
#    
#    def L_per_SFR_of_t(self, wave=1500.):
#        """
#        UV luminosity per unit SFR.
#        """
#                
#        j = np.argmin(np.abs(wave - self.wavelengths))
#        
#        dwavednu = np.diff(self.wavelengths) / np.diff(self.frequencies)
#        
#        yield_UV = self.data[j,:] * np.abs(dwavednu[j])
#        
#        # Current units: 
#        # if pop_ssp: erg / sec / Hz / (Msun / 1e6)
#        # else: erg / sec / Hz / (Msun / yr)
#                    
#        # to erg / s / A / Msun
#        if self.pf['pop_ssp']:
#            yield_UV /= 1e6
#        # or erg / s / A / (Msun / yr)
#        else:
#            pass
#            
#        return yield_UV
#        
#    def LUV(self):
#        return self.L_per_SFR_of_t()[-1]
#        
#    @property
#    def L1500(self):
#        return self.L_per_sfr()   
#        
#    def L_per_sfr(self, wave=1500.):   
#        """
#        Specific emissivity at provided wavelength.
#        
#        Units are 
#            erg / s / Hz / (Msun / yr)
#        or 
#            erg / s / Hz / Msun
#        """
#        
#        yield_UV = self.L_per_SFR_of_t(wave)
#            
#        # Interpolate in time to obtain final LUV
#        if self.pf['pop_tsf'] in self.times:
#            return yield_UV[np.argmin(np.abs(self.times - self.pf['pop_tsf']))]
#            
#        k = np.argmin(np.abs(self.pf['pop_tsf'] - self.times))    
#        if self.times[k] > self.pf['pop_tsf']:
#            k -= 1
#            
#        if not hasattr(self, '_LUV_interp'):
#            self._LUV_interp = interp1d(self.times, yield_UV, kind='cubic')
#            
#        return self._LUV_interp(self.pf['pop_tsf'])
#        
#    def kappa_UV_of_t(self):        
#        return 1. / self.LUV_of_t()
#        
#    def kappa_UV(self):    
#        """
#        Number of photons emitted per stellar baryon of star formation.
#        
#        If star formation is continuous, this will have units of:
#            (Msun / yr) / (erg / s / Hz)
#        If star formation is in a burst, this will have units of:
#            Msun / (erg / s / Hz)
#        Returns
#        -------
#        Two-dimensional array containing photon yield per unit stellar baryon per
#        second per angstrom. First axis corresponds to photon wavelength (or energy), 
#        and second axis to time.
#        
#        """
#        
#        return 1. / self.LUV()
#
#    def integrated_emissivity(self, l0, l1, unit='A'):
#        # Find band of interest -- should be more precise and interpolate
#        
#        if unit == 'A':
#            x = self.wavelengths
#            i0 = np.argmin(np.abs(x - l0))
#            i1 = np.argmin(np.abs(x - l1))
#        elif unit == 'Hz':
#            x = self.frequencies
#            i1 = np.argmin(np.abs(x - l0))
#            i0 = np.argmin(np.abs(x - l1))
#        
#        # Current units: photons / sec / baryon / Angstrom      
#        
#        # Count up the photons in each spectral bin for all times
#        photons_per_b_t = np.zeros_like(self.times)
#        for i in range(self.times.size):
#            photons_per_b_t[i] = np.trapz(self.emissivity_per_sfr[i1:i0,i], 
#                x=x[i1:i0])
#                
#        t = self.times * s_per_myr 
#      
#    def erg_per_phot(self, Emin, Emax):
#        return self.eV_per_phot(Emin, Emax) * erg_per_ev  
#        
#    def eV_per_phot(self, Emin, Emax):
#        i0 = np.argmin(np.abs(self.energies - Emin))
#        i1 = np.argmin(np.abs(self.energies - Emax))
#        
#        it = -1
#                
#        # Must convert units
#        E_avg = np.trapz(self.data[i1:i0,it] * self.energies[i1:i0], 
#            x=self.wavelengths[i1:i0]) \
#            / np.trapz(self.data[i1:i0,it], x=self.wavelengths[i1:i0])    
#        
#        return E_avg
#        
#    def yield_per_sfr(self, Emin, Emax):
#        """
#        Must be in the internal units of erg / g.
#        """
#        
#        # Units self-explanatory
#        N = self.PhotonsPerBaryon(Emin, Emax)
#
#        # Convert to erg / g        
#        return N * self.erg_per_phot(Emin, Emax) * self.cosm.b_per_g
# 
#    def PhotonsPerBaryon_of_t(self, Emin, Emax):    
#        """
#        Compute photons emitted per baryon for all times.
#        
#        Returns
#        -------
#        Integrated flux between (Emin, Emax) for all times in units of 
#        photons / sec / (Msun [/ yr])
#        """
#        # Find band of interest -- should be more precise and interpolate
#        i0 = np.argmin(np.abs(self.energies - Emin))
#        i1 = np.argmin(np.abs(self.energies - Emax))
#                                     
#        # Count up the photons in each spectral bin for all times
#        photons_per_b_t = np.zeros_like(self.times)
#        for i in range(self.times.size):
#            integrand = self.data[i1:i0,i] * self.wavelengths[i1:i0]\
#                / (self.energies[i1:i0] * erg_per_ev)
#            
#            # Current units of integrand: photons / sec / Angstrom / [depends on ssp]
#            
#            photons_per_b_t[i] = \
#                np.trapz(integrand, x=np.log(self.wavelengths[i1:i0]))
#            
#        # Current units: 
#        # if pop_ssp: photons / sec / (Msun / 1e6)
#        # else: photons / sec / (Msun / yr)
#        
#        return photons_per_b_t
#        
#    @property
#    def Nion(self):
#        if not hasattr(self, '_Nion'):
#            self._Nion = self.PhotonsPerBaryon(13.6, 24.6)
#        return self._Nion
#        
#    @property
#    def Nlw(self):
#        if not hasattr(self, '_Nlw'):
#            self._Nlw = self.PhotonsPerBaryon(10.2, 13.6)
#        return self._Nlw    
#        
#    def PhotonsPerBaryon(self, Emin, Emax):    
#        """
#        Compute the number of photons emitted per unit stellar baryon.
#        
#        ..note:: This integrand over the provided band, and cumulatively over time.
#        
#        Parameters
#        ----------
#        Emin : int, float
#            Minimum rest-frame photon energy to consider [eV].
#        Emax : int, float
#            Maximum rest-frame photon energy to consider [eV].
#        
#        Returns
#        -------
#        An array with the same dimensions as ``self.times``, representing the 
#        cumulative number of photons emitted per stellar baryon of star formation
#        as a function of time.
#
#        """
#
#        #assert self.pf['pop_ssp'], "Probably shouldn't do this for continuous SF."
#
#        photons_per_b_t = self.PhotonsPerBaryon_of_t(Emin, Emax)    
#
#        # Current units: 
#        # if pop_ssp: photons / sec / (Msun / 1e6)
#        # else: photons / sec / (Msun / yr)
#
#        g_per_b = self.cosm.g_per_baryon
#
#        # Integrate (cumulatively) over time
#        if self.pf['pop_ssp']:
#            photons_per_b_t *= g_per_b / g_per_msun
#            return np.trapz(photons_per_b_t, x=self.times * s_per_myr) / 1e6
#        # Take steady-state result
#        else:
#            photons_per_b_t *= s_per_yr
#            photons_per_b_t *= g_per_b / g_per_msun
#            return photons_per_b_t[-1]
#
##class Spectrum(StellarPopulation):
##    def __init__(self, **kwargs):
##        StellarPopulation.__init__(self, **kwargs)
##    
##    @property
##    def Lbol(self):
##        if not hasattr(self, '_Lbol'):
##            to_int = self.intens
##               
##            self._Lbol = np.trapz(to_int, x=self.energies[-1::-1])
##            
##        return self._Lbol
##               
##    @property
##    def intens(self):
##        if not hasattr(self, '_intens'):
##            self._intens = self.data[-1::-1,-1] * self.dlde
##    
##        return self._intens
##        
##    @property
##    def nrg(self):
##        if not hasattr(self, '_nrg'):
##            self._nrg = self.energies[-1::-1]
##
##        return self._nrg
##        
#    @property
#    def dlde(self):
#        if not hasattr(self, '_dlde'):
#            diff = np.diff(self.wavelengths) / np.diff(self.energies)
#            self._dlde = np.concatenate((diff, [diff[-1]]))
#                    
#        return self._dlde
#        
#    def __call__(self, E, t=0.0):        
#        return np.interp(E, self.nrg, self.data[-1::-1,0]) #/ self.Lbol
#        
#        
        