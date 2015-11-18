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
from ares.physics.Constants import h_p, c, erg_per_ev, g_per_msun, s_per_yr, \
    s_per_myr

_input = os.getenv('ARES') + '/input/starburst99'

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
 'Z': 0.04,
 'imf': 2.35,
 'nebular': True,
 'continuous_sf': False,
}

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
    
def _figure_name(Z=0.04, imf=2.35, nebular=True, continuous_sf=False):
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
    if continuous_sf:
        mask[options % 2 == 1] = 0
    else:
        mask[options % 2 == 0] = 0
    
    # Can't be > 6
    if nebular:
        mask[options > 6] = 0
    else:
        mask[options <= 6] = 0
    
    if imf == 2.35:
        for i in options:
            if i not in [1,2,7,8]:
                mask[i-1] *= 0
    elif imf == 3.3:
        for i in options:
            if i not in [3,4,9,10]:
                mask[i-1] *= 0  
    elif imf == 30:
        for i in options:
            if i not in [5,6,11,12]:
                mask[i-1] *= 0
                      
    Zvals = metallicities.values()
    
    if Z not in Zvals:
        raise ValueError('Unrecognized metallicity.')
        
    Z_suffix = metallicities.keys()[Zvals.index(Z)]
    
    if mask.sum() > 1:
        raise ValueError('Ambiguous SED.')
    
    j = options[mask == 1]
    k = fig_num.index(j)
    
    return '%s%s.dat' % (fig_prefix[k], Z_suffix)
    
# Wavelengths in Angstroms (ascending)
#wave = s99_data[:,0]
#
## For time-integral
#weights = np.array([1] * 20 + [10] * 8 + [100] * 8)

class StellarPopulation:
    def __init__(self, **kwargs):
        self.pf = pars.copy()
        self.pf.update(kwargs)
    
        self._load()
    
    def _load(self):
        self.fn = _figure_name(**self.pf)
        self._raw_data = _reader(self.fn)
    
    @property
    def data(self):
        """
        Units = erg / s / A / [depends]
        
        Where, if instantaneous burst, [depends] = Msun
        and if continuous SF, [depends] = Msun / yr
        
        """
        if not hasattr(self, '_data'):
            self._data = 10**self._raw_data[:,1:]
        return self._data
    
    @property
    def cosm(self):
        if not hasattr(self, '_cosm'):
            self._cosm = Cosmology()
            
        return self._cosm
    
    @property
    def wavelengths(self):
        if not hasattr(self, '_wavelengths'):
            self._wavelengths = self._raw_data[:,0]
            
        return self._wavelengths
        
    @property
    def energies(self):
        if not hasattr(self, '_energies'):
            self._energies = h_p * c / (self.wavelengths / 1e8) / erg_per_ev
    
        return self._energies    
        
    @property
    def frequencies(self):
        if not hasattr(self, '_frequencies'):
            self._frequencies = c / (self.wavelengths / 1e8)
    
        return self._frequencies

    @property
    def weights(self):
        if not hasattr(self, '_weights'):
            self._weights = np.array([1.] * 20 + [10.] * 8 + [100.] * 8)
    
        return self._weights    
        
    @property
    def times(self):
        if not hasattr(self, '_times'):
            self._times = np.cumsum(self.weights)
        
        return self._times
        
    @property
    def time_averaged_sed(self):
        if not hasattr(self, '_tavg_sed'):
            self._tavg_sed = np.dot(self.data, self.weights) / self.times.max()
        
        return self._tavg_sed

    @property
    def emissivity_per_sfr(self):
        """
        Photon emissivity?
        """
        if not hasattr(self, '_E_per_M'):
            self._E_per_M = np.zeros_like(self.data)
            for i in range(self.times.size):
                self._E_per_M[:,i] = self.data[:,i] / (self.energies * erg_per_ev)    

            if self.pf['continuous_sf']:
                pass
            else:
                self._E_per_M /= 1e6

        return self._E_per_M

    @property
    def uvslope(self):
        if not hasattr(self, '_uvslope'):
            self._uvslope = np.zeros_like(self.data)
            for i in range(self.times.size):
                self._uvslope[1:,i] = np.diff(np.log(self.data[:,i])) \
                    / np.diff(np.log(self.wavelengths))

        return self._uvslope

    @property
    def NperB(self):
        """
        Number of photons emitted per stellar baryon of star formation.
        
        Returns
        -------
        Two-dimensional array containing photon yield per unit stellar baryon per
        second per angstrom. First axis corresponds to photon wavelength (or energy), 
        and second axis to time.
        
        """
        if not hasattr(self, '_NperB'):
            self._NperB = np.zeros_like(self.data)
            for i in range(self.times.size):
                self._NperB[:,i] = self.data[:,i] / (self.energies * erg_per_ev) 
                
            # Introduce "per baryon" units assuming 1 Msun in stars
            self._NperB /= (g_per_msun / self.cosm.g_per_baryon)
            
            if self.pf['continuous_sf']:
                pass
            else:
                self._NperB /= 1e6

        return self._NperB
        
    @property
    def kappa_UV(self):    
        """
        Number of photons emitted per stellar baryon of star formation.
        
        Returns
        -------
        Two-dimensional array containing photon yield per unit stellar baryon per
        second per angstrom. First axis corresponds to photon wavelength (or energy), 
        and second axis to time.
        
        """
        
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
        for i in range(self.times.size):
            photons_per_b_t[i] = np.trapz(self.emissivity_per_sfr[i1:i0,i], 
                x=x[i1:i0])
                
        t = self.times * s_per_myr    
                
    def PhotonsPerBaryon(self, Emin, Emax):    
        """
        Compute the cumulative number of photons emitted per unit stellar baryon.
        
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
        
        # Find band of interest -- should be more precise and interpolate
        i0 = np.argmin(np.abs(self.energies - Emin))
        i1 = np.argmin(np.abs(self.energies - Emax))
              
        # Current units: photons / sec / baryon / Angstrom      
                                     
        # Count up the photons in each spectral bin for all times
        photons_per_b_t = np.zeros_like(self.times)
        for i in range(self.times.size):
            photons_per_b_t[i] = np.trapz(self.NperB[i1:i0,i], 
                x=self.wavelengths[i1:i0])
            
        # Current units: photons / sec / baryon
        t = self.times * s_per_myr
        
        # Integrate (cumulatively) over time
        return cumtrapz(photons_per_b_t, x=t, initial=0.0)
                
class Spectrum(StellarPopulation):
    def __init__(self, **kwargs):
        StellarPopulation.__init__(self, **kwargs)
    
    @property
    def Lbol(self):
        if not hasattr(self, '_Lbol'):
            to_int = self.intens
               
            self._Lbol = np.trapz(to_int, x=self.energies[-1::-1])
            
        return self._Lbol
               
    @property
    def intens(self):
        if not hasattr(self, '_intens'):
            self._intens = self.data[-1::-1,-1] * self.dlde
    
        return self._intens
        
    @property
    def nrg(self):
        if not hasattr(self, '_nrg'):
            self._nrg = self.energies[-1::-1]

        return self._nrg
        
    @property
    def dlde(self):
        if not hasattr(self, '_dlde'):
            diff = np.diff(self.wavelengths) / np.diff(self.energies)
            self._dlde = np.concatenate((diff, [diff[-1]]))
                    
        return self._dlde
        
    def __call__(self, E, t=0.0):        
        return np.interp(E, self.nrg, self.data[-1::-1,0]) #/ self.Lbol
        
        
        