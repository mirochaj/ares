"""

blue_tides.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Tue  4 Jan 2022 16:42:52 EST

Description: A smattering of data from Waters et al. (2016) and Wilkins et
al. (2016).

"""

import numpy as np

inf = np.inf

smhm = \
{
 'bin': 0.25,
 'bin_edges': np.arange(10.5, 12.2, 0.25),
 'z': np.arange(8., 14, 1),
 'info': 'log10(stellar mass / DM halo mass)',
 'reference': 'Table A1 in Wilkins et al. (2016)',
 'smhm': np.array(
    [[-2.52, -2.34, -2.18, -2.05, -1.89, -1.75, -1.87],
     [-2.57, -2.40, -2.26, -2.11, -1.99, -2.05, -inf],
     [-2.60, -2.45, -2.30, -2.21, -inf, -inf, -inf],
     [-2.66, -2.52, -2.38, -2.28, -inf, -inf, -inf],
     [-2.67, -2.54, -2.46, -inf, -inf, -inf, -inf],
     [-2.75, -2.56, -inf, -inf, -inf, -inf, -inf]])
}

ssfr = \
{
 'bin': 0.25,
 'bin_edges': np.arange(8., 10.2, 0.25),
 'z': np.arange(8., 14, 1),
 'info': 'log10(SFR / stellar mass)',
 'reference': 'Table A4 in Wilkins et al. (2016)',
 'ssfr': 'TBD',
}

mass_to_light = \
{
 'bin': 0.25,
 'bin_edges': np.arange(8., 10.2, 0.25),
 'z': np.arange(8., 14, 1),
 'info': 'log10(Mstell / LUV), where [LUV]=erg/s/Hz at 1500A',
 'reference': 'Table A6 in Wilkins et al. (2016)',
 'observed': np.array(
    [
     [-20.03, -19.98, -19.91, -19.80, -19.67, -19.51, -19.38, -19.25, -19.21],
     [-20.10, -20.05, -19.97, -19.86, -19.72, -19.56, -19.44, -19.32, -inf],
     [-20.20, -20.15, -20.07, -19.93, -19.77, -19.62, -19.5, -inf, -inf],
     [-20.25, -20.19, -20.10, -19.94, -19.81, -inf, -inf, -inf, -inf],
     [-20.32, -20.23, -20.15, -20.00, -inf, -inf, -inf, -inf, -inf],
     [-20.37, -20.33, -inf, -inf, -inf, -inf, -inf, -inf, -inf]]),
 'intrinsic': 'TBD',
}
