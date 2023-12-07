"""
Weaver et al. (2022) COSMOS2020 number counts (Table 2).
"""

import numpy as np

# These are bin centers
magbins = np.arange(19.25, 27.75, 0.5)

# These are in log10(number / mag / deg^2)
# Not clear if there's a factor of 2 lurking here due to common 0.5 mag bin width
data_farmer = \
{
 'i': np.array([3.01, 3.23, 3.44, 3.64, 3.85, 4.03, 4.21, 4.38, 4.54, 4.71, 4.86,
    4.97, 5.08, 5.20, 5.29, 5.35, 5.22]),
 'K': np.array([3.64, 3.85, 4.03, 4.18, 4.29, 4.42, 4.56, 4.68, 4.79, 4.90, 5.00,
    5.11, 5.22, 5.21, 5.03, -np.inf, -np.inf]),
 'K_ultradeep': np.array([3.65, 3.86, 4.02, 4.16, 4.29, 4.42, 4.54, 4.66, 4.78,
    4.88, 4.97, 5.07, 5.18, 5.24, 5.13, -np.inf, -np.inf]),
}

def get_cts(band='K', classic=False, ultradeep=True):
    """
    Return the number counts as a function of magnitude from COSMOS2020.

    .. note :: In Weaver et al. galaxies are selected via izYJHK composite.

    Parameters
    ----------
    band : str
        Has to be either K or i
    classic : bool
        Weaver et al. provide two different versions of the catalog, one with
        'classic' approach, another using newer "Farmer".
    ultradeep : bool
        Two different versions of the catalog based on depth as well.

    """

    data = data_classic if classic else data_farmer
    key = f'{band}_ultradeep' if (ultradeep and band == 'K') else band

    return magbins, 10**data[key]
