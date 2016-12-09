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
                 
        data = 10**data_3d
                    
    else:        
        fn = _figure_name(**kwargs)
        _raw_data = _reader(fn)
        wavelengths = _raw_data[:,0]
        data = 10**_raw_data[:,1:]
        
    return wavelengths, data    
        
        