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
 'reference': 'Table A1 in Wilkins et al. (2017)',
 'data': np.array(
    [[-2.52, -2.34, -2.18, -2.05, -1.89, -1.75, -1.87],
     [-2.57, -2.40, -2.26, -2.11, -1.99, -2.05, -inf],
     [-2.60, -2.45, -2.30, -2.21, -inf, -inf, -inf],
     [-2.66, -2.52, -2.38, -2.28, -inf, -inf, -inf],
     [-2.67, -2.54, -2.46, -inf, -inf, -inf, -inf],
     [-2.75, -2.56, -inf, -inf, -inf, -inf, -inf]])
}

smf = \
{
 'bin': 0.2,
 'bin_edges': np.arange(8, 10.25, 0.2),
 'z': np.arange(8., 14, 1),
 'info': 'log10(number density / dex in Mstell / cMpc^3)',
 'reference': 'Table A2 in Wilkins et al. (2017)',
 'data': np.array(
    [[-2.76, -2.97, -3.19, -3.42, -3.66, -3.92, -4.20, -4.50, -4.87, -5.17,
        -5.59, -5.96],
     [-3.22, -3.45, -3.71, -3.98, -4.27, -4.58, -4.93, -5.28, -5.65, -6.07,
        -inf, -inf],
     [-3.72, -4.01, -4.30, -4.61, -4.96, -5.34, -5.65, -6.11, -inf, -inf,
        -inf, -inf],
     [-4.31, -4.64, -4.97, -5.35, -5.70, -6.21, -inf, -inf, -inf, -inf,
        -inf, -inf],
     [-4.94, -5.30, -5.70, -6.14, -6.53, -inf, -inf, -inf, -inf, -inf,
        -inf, -inf],
     [-5.61, -6.02, -6.40, -inf, -inf, -inf, -inf, -inf, -inf, -inf,
        -inf, -inf]])
}

sfrdf = \
{
 'bin': 0.2,
 'bin_edges': np.arange(-0.6, 2.45, 0.2),
 'z': np.arange(8., 14, 1),
 'info': 'log10(number density / dex in SFR / cMpc^3)',
 'reference': 'Table A3 in Wilkins et al. (2017)',
 'data': np.array(
    [[-2.22, -2.41, -2.61, -2.80, -3.01, -3.22, -3.46, -3.7, -3.97, -4.25,
        -4.57, -4.91, -5.28, -5.65, -6.12, -6.57],
     [-2.51, -2.71, -2.93, -3.16, -3.39, -3.64, -3.90, -4.14, -4.44, -4.79,
        -5.12, -5.44, -5.91, -6.34, -inf, -inf],
     [-2.80, -3.02, -3.26, -3.52, -3.78, -4.06, -4.39, -4.69, -5.05, -5.41,
        -5.77, -6.21, -inf, -inf, -inf, -inf],
     [-3.11, -3.36, -3.62, -3.93, -4.22, -4.54, -4.89, -5.21, -5.63, -5.98,
        -6.32, -inf, -inf, -inf, -inf, -inf],
     [-3.50, -3.77, -4.07, -4.42, -4.72, -5.13, -5.49, -5.82, -6.23, -inf,
        -inf, -inf, -inf, -inf, -inf, -inf],
     [-3.93, -4.21, -4.54, -4.92, -5.28, -5.67, -6.04, -6.4, -inf, -inf,
        -inf, -inf, -inf, -inf, -inf, -inf]])
}

ssfr = \
{
 'bin': 0.25,
 'bin_edges': np.arange(8., 10.2, 0.25),
 'z': np.arange(8., 14, 1),
 'info': 'log10(SFR / stellar mass)',
 'reference': 'Table A4 in Wilkins et al. (2017)',
 'data': np.array(
    [[-8.10, -8.09, -8.09, -8.10, -8.10, -8.10, -8.11, -8.11, -8.10],
     [-8.01, -8.00, -7.99, -7.97, -7.98, -7.96, -7.97, -7.96, -inf],
     [-7.91, -7.91, -7.91, -7.90, -7.89, -7.89, -7.96, -inf, -inf],
     [-7.81, -7.81, -7.79, -7.76, -7.74, -inf, -inf, -inf, -inf],
     [-7.76, -7.77, -7.74, -7.74, -inf, -inf, -inf, -inf, -inf],
     [-7.70, -7.65, -inf, -inf, -inf, -inf, -inf, -inf, -inf]])
}


mass_to_light = \
{
 'bin': 0.25,
 'bin_edges': np.arange(8., 10.2, 0.25),
 'z': np.arange(8., 14, 1),
 'info': 'log10(Mstell / LUV), where [LUV]=erg/s/Hz at 1500A',
 'reference': 'Table A6 in Wilkins et al. (2017)',
 'data': np.array(
    [
     [-20.03, -19.98, -19.91, -19.80, -19.67, -19.51, -19.38, -19.25, -19.21],
     [-20.10, -20.05, -19.97, -19.86, -19.72, -19.56, -19.44, -19.32, -inf],
     [-20.20, -20.15, -20.07, -19.93, -19.77, -19.62, -19.5, -inf, -inf],
     [-20.25, -20.19, -20.10, -19.94, -19.81, -inf, -inf, -inf, -inf],
     [-20.32, -20.23, -20.15, -20.00, -inf, -inf, -inf, -inf, -inf],
     [-20.37, -20.33, -inf, -inf, -inf, -inf, -inf, -inf, -inf]]),
 'data_int': 'TBD',
}

uvlf = \
{
 8: {'alpha': -1.84, 'log_phi_star': -8.9, 'M_star': -20.95},
 9: {'alpha': -1.94, 'log_phi_star': -9.74, 'M_star': -20.77},
 10: {'alpha': -2.01, 'log_phi_star': -10.27, 'M_star': -20.39},
 11: {'alpha': -2.07, 'log_phi_star': -10.77, 'M_star': -20.00},
 12: {'alpha': -2.12, 'log_phi_star': -11.4, 'M_star': -19.65},
 13: {'alpha': -2.13, 'log_phi_star': -11.71, 'M_star': -19.11},
}

def schechter_jaacks(M, log_phi_star=-9, M_star=-20, alpha=-2):
    """
    .. note :: There's a typo in Eq. 5 of Y. Feng et al. (2016). They had
    a factor of (1 - alpha), it should be (1 + alpha).
    """
    A = 0.4 * np.log(10.)
    log_phi = log_phi_star + np.log(A) + A * (M_star - M) * (1. + alpha) \
        - 10**(0.4 * (M_star - M))

    return np.exp(log_phi)
