"""
Helgason et al. (2012).
"""

import numpy as np

# Table 2
helgason_cols = ['l_eff', 'N', 'zmax', 'Mstar', 'q', 'pstar', 'p', 'alpha', 'r']
helgason_fits = \
{
 'UV': [0.15, 24, 8.0, -19.62, 1.1, 2.43, 0.2, -1.00, 0.086],
 'U':  [0.36, 27, 4.5, -20.20, 1.0, 5.46, 0.5, -1.00, 0.076],
 'B':  [0.45, 44, 4.5, -21.35, 0.6, 3.41, 0.4, -1.00, 0.055],
 'V':  [0.55, 18, 3.6, -22.13, 0.5, 2.42, 0.5, -1.00, 0.060],
 'R':  [0.65, 25, 3.0, -22.40, 0.5, 2.25, 0.5, -1.00, 0.070],
 'I':  [0.79, 17, 3.0, -22.80, 0.4, 2.05, 0.4, -1.00, 0.070],
 'z':  [0.91, 7,  2.9, -22.86, 0.4, 2.55, 0.4, -1.00, 0.060],
 'J':  [1.27, 15, 3.2, -23.04, 0.4, 2.21, 0.6, -1.00, 0.035],
 'H':  [1.63, 6,  3.2, -23.41, 0.5, 1.91, 0.8, -1.00, 0.035],
 'K':  [2.20, 38, 3.8, -22.97, 0.4, 2.74, 0.8, -1.00, 0.035],
 'L':  [3.60, 6,  0.7, -22.40, 0.2, 3.29, 0.8, -1.00, 0.035],
 'M':  [4.50, 6,  0.7, -21.84, 0.3, 3.29, 0.8, -1.00, 0.035],
}

bands = helgason_fits.keys()
waves = [helgason_fits[key][0] for key in helgason_fits.keys()]

# Table 3
mlim = [22, 24, 26, 28, None] # cols

helgason_ebl = {}
helgason_ebl['B'] = (3.33, 1.72, -0.82), (2.26, 1.56, -0.71), (1.17, 1.24, -0.50),\
    (0.52, 0.88, -0.29), (4.92, 1.81, -0.88)
helgason_ebl['V'] = (2.95, 1.54, -0.73), (1.90, 1.36, -0.61), (0.96, 1.05, -0.41),\
    (0.42, 0.73, -0.23), (5.65, 1.73, -0.85)
helgason_ebl['R'] = (2.86, 1.54, -0.73), (1.75, 1.31, -0.58), (0.85, 0.98, -0.38),\
    (0.37, 0.67, -0.21), (6.56, 1.82, -0.92)
helgason_ebl['I'] = (2.81, 1.58, -0.76), (1.58, 1.27, -0.55), (0.72, 0.92, -0.34),\
    (0.30, 0.61, -0.17), (7.97, 2.01, -1.06)
helgason_ebl['J'] = (2.59, 1.56, -0.77), (1.20, 1.10, -0.47), (0.48, 0.72, -0.25),\
    (0.18, 0.45, -0.12), (9.60, 2.40, -1.28)
helgason_ebl['H'] = (2.25, 1.50, -0.71), (0.96, 0.96, -0.40), (0.36, 0.57, -0.19),\
    (0.13, 0.34, -0.09), (9.34, 2.59, -1.29)
helgason_ebl['K'] = (1.74, 1.41, -0.60), (0.69, 0.82, -0.30), (0.24, 0.44, -0.13),\
    (0.08, 0.23, -0.06), (8.09, 2.52, -1.14)
helgason_ebl['L'] = (0.98, 1.05, -0.40), (0.34, 0.57, -0.17), (0.11, 0.27, -0.06),\
    (0.03, 0.12, -0.02), (4.87, 1.72, -0.71)
helgason_ebl['M'] = (0.75, 0.83, -0.31), (0.24, 0.45, -0.13), (0.07, 0.20, -0.04),\
    (0.02, 0.09, -0.02), (3.28, 1.21, -0.49)


def plot_ebl(ax, as_contours=True, **kwargs):
    """
    Plot the mean EBL [nW/m^2/sr] as a function of observed wavelength [microns].
    """
    waves_pl = []
    lo = []; hi = []
    for i, band in enumerate(bands):
        if band not in helgason_ebl:
            continue

        if not as_contours:
            ax.scatter(waves[i], helgason_ebl[band][-1][0], **kwargs)

            kw = kwargs.copy()
            if 'marker' in kwargs:
                del kw['marker']

            ax.plot([waves[i]]*2,
                helgason_ebl[band][-1][0] + np.array(helgason_ebl[band][-1][1:]),
                **kwargs)

        else:
            waves_pl.append(waves[i])
            lo.append(helgason_ebl[band][-1][0] + np.array(helgason_ebl[band][-1][1]))
            hi.append(helgason_ebl[band][-1][0] + np.array(helgason_ebl[band][-1][2]))

    ax.fill_between(waves_pl, lo, hi, **kwargs)

    return ax
