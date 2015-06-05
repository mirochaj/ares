"""
Leitherer, C., Schaerer, D., Goldader, J. D., Delgado, R. M. G., Robert, C.,
Kune, D. F., de Mello, D. F., Devost, D., & Heckman, T. M. 1999, ApJS,
123, 3

Notes
-----

"""

import re, os
import numpy as np
from ares.physics.Constants import h_p, c, erg_per_ev

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

info = \
{
 'flux_units': r'$\mathrm{erg} \ \mathrm{s}^{-1} \ \AA^{-1}$',
}

pars = \
{
 'Z': 0.04,
 'imf': 2.35,
 'nebular': True,
 'continuous_sf': True,
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
    
def _figure_name(Z=0.04, imf=2.35, nebular=True, continuous_sf=True):
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
    
        self.data = 10**self._raw_data[:,1:]
    
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
    def weights(self):
        if not hasattr(self, '_weights'):
            self._weights = np.array([1] * 20 + [10] * 8 + [100] * 8)
    
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

class Spectrum:
    def __init__(self):
        pass
    
    def __call__(self, E, t=0.0):
        pass
        