"""
Table 2 in Driver et al. 2016
"""

import numpy as np

ebl = {}
ebl['FUV'] = 0.153, 1.45, 1.45, 1.36, 0.07, 0.00, 0.04, 0.16
ebl['NUV'] = 0.225,	3.15, 3.14,	2.86, 0.15, 0.02, 0.05, 0.45
ebl['u']   = 0.356, 4.03, 4.01, 3.41, 0.19, 0.04, 0.09, 0.46
ebl['g']   = 0.470, 5.36, 5.34, 5.05, 0.25, 0.04, 0.05, 0.59
ebl['r']   = 0.618, 7.47, 7.45, 7.29, 0.34, 0.05, 0.04, 0.69
ebl['i']   = 0.749, 9.55, 9.52, 9.35, 0.44, 0.00, 0.05, 0.92
ebl['z']   = 0.895, 10.15, 10.13, 9.98, 0.47, 0.03, 0.05, 0.96
ebl['Y']   = 1.021, 10.44, 10.41, 10.23, 0.48, 0.00, 0.07, 1.05
ebl['J']   = 1.252, 10.38, 10.35, 10.22, 0.48, 0.00, 0.05, 0.99
ebl['H']   = 1.643, 10.12, 10.10, 9.99, 0.47, 0.01, 0.06, 1.01
ebl['K']   = 2.150, 8.72, 8.71, 8.57, 0.40, 0.02, 0.04, 0.76
ebl['IRAC1'] = 3.544, 5.17, 5.15, 5.03, 0.24, 0.03, 0.06, 0.43
ebl['IRAC2'] = 4.487, 3.60, 3.59, 3.47, 0.17, 0.02, 0.05, 0.28
ebl['IRAC4'] = 7.841, 2.45, 2.45, 1.49, 0.11, 0.77, 0.15, 0.08

bands = ebl.keys()
waves = [ebl[key][0] for key in ebl.keys()]

cols = 'Wavelength', 'Best Fit', 'Median',	'Lower Limit', \
    'Zero-point Error', 'Fitting Error', 'Poisson Error', 'CV Error'

def plot_ebl(ax, **kwargs):
    """
    Plot the mean EBL [nW/m^2/sr] as a function of observed wavelength [microns].
    """
    waves_pl = []
    lo = []; hi = []
    for i, band in enumerate(bands):
        if band not in ebl:
            continue

        waves_pl.append(waves[i])
        err = np.sqrt(np.sum(np.array(ebl[band][4:])**2))
        lo.append(ebl[band][1]-err)
        hi.append(ebl[band][1]+err)

    ax.fill_between(waves_pl, lo, hi, **kwargs)

    return ax
