"""

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2009-09-01.

Description: Contains various constants that may be of use.

Note:  All units are cgs unless stated otherwise.

"""

from math import pi

# lengths
km_per_pc = 3.08568e13
km_per_mpc = km_per_pc*1e6
km_per_gpc = km_per_mpc*1e3
cm_per_pc = km_per_pc*1e5
cm_per_kpc = cm_per_pc*1e3
cm_per_mpc = cm_per_pc*1e6
cm_per_gpc = cm_per_mpc*1e3
cm_per_km = 1e5
cm_per_m = 1e2
cm_per_rsun = 695500. * cm_per_km
cm_per_rEarth = 637100000.
cm_per_au = 1.49597871e13
cm_per_ang = 1e-8
m_per_mpc = cm_per_mpc * 1e-2

# masses
g_per_amu = 1.660538921e-24
g_per_msun = 1.98892e33
mH_amu = 1.00794
mHe_amu = 4.002602

# times
s_per_yr = 365.25*24*3600
s_per_kyr = s_per_yr*1e3
s_per_myr = s_per_kyr*1e3
s_per_gyr = s_per_myr*1e3

# angles
sqdeg_per_std = 180.0**2 / pi**2

# General constants
h = h_p = h_P = 6.626068e-27    # Planck's constant - [h] = erg*s
h_bar = h / 2 / pi   			# H-bar - [h_bar] = erg*s
c = 29979245800.0 				# Speed of light - [c] = cm/s
k_B = 1.3806503e-16			    # Boltzmann's constant - [k_B] = erg/K
G = 6.673e-8     				# Gravitational constant - [G] = cm^3/g/s^2
e = 1.60217646e-19   			# Electron charge - [e] = C
e_cgs = 4.803204e-10            # Electron charge - [e] = statC
m_e = 9.10938188e-28     		# Electron mass - [m_e] = g
m_p = 1.67262158e-24    		# Proton mass - [m_p] = g
m_n = 1.67492729e-24            # Neutron mass - [m_n] = g
sigma_T = 6.65e-25			    # Cross section for Thomson scattering - [sigma_T] = cm^2
alpha_FS = 1 / 137.035999070    # Fine structure constant - unitless
Ryd = 2.1798719e-11             # Rydberg in erg

# energies / wavelengths / frequencies
erg_per_j = 1e-7
erg_per_ev = e / erg_per_j
erg_per_kev = 1e3 * erg_per_ev

# 10^7 erg / J, so 1 W = 1 J / s = 10^7 erg/s
erg_per_s_per_W = 1e7
erg_per_s_per_nW = 1e7 / 1e9

# Convert specific intensities from eV^-1 to Hz^-1
ev_per_hz = h / erg_per_ev

# Convert mass density from CGS to Msun / Mpc^3
rho_cgs = cm_per_mpc**3 / g_per_msun

# Convert accretion rate density from cgs to Msun / yr / Mpc^3
rhodot_cgs = s_per_yr * cm_per_mpc**3 / g_per_msun

# Stefan-Boltzmann constant - [sigma_SB] = erg / cm^2 / deg^4 / s
sigma_SB = 2.0 * pi**5 * k_B**4 / 15.0 / c**2 / h**3

# Hydrogen
A10 = 2.85e-15 				    # HI 21cm spontaneous emission coefficient - [A10] = Hz
A_LyA = 6.25e8
E10 = 5.9e-6 				    # Energy difference between hyperfine states - [E10] = eV
m_H = m_p + m_e 			    # Mass of a hydrogen atom - [m_H] = g
nu_0 = 1420.4057e6 			    # Rest frequency of HI 21cm line - [nu_0] = Hz
nu_alpha = 2.47e15              # Rest frequency of Lyman-alpha - [nu_alpha] = Hz
T_star = 0.068 				    # Corresponding temperature difference between HI hyperfine states - [T_star] = K
a_0 = 5.292e-9 				    # Bohr radius - [a_0] = cm
f12 = 0.4162                    # Lyman-alpha oscillator strength

E_LL = Ryd / erg_per_ev
E_LyA = E_LL * (1. - 1. / 2**2)
E_LyB = E_LL * (1. - 1. / 3**2)
nu_alpha = E_LyA * erg_per_ev / h
nu_beta = E_LyB * erg_per_ev / h
nu_LL = E_LL * erg_per_ev / h
lam_LL = h_p * c * 1e8 / (E_LL * erg_per_ev)
lam_LyA = h_p * c * 1e8 / (E_LyA * erg_per_ev)
lam_LyB = h_p * c * 1e8 / (E_LyB * erg_per_ev)
dnu = nu_LL - nu_alpha
J21_num = 1e-21 / E_LyA / erg_per_ev

nu_0_mhz = nu_0 / 1e6

# solar luminosity
Lsun = 3.828e33
Tsun = 5778.
Rsun = 695508. * cm_per_km

# Helium
m_He = m_HeI = 2.0 * (m_p + m_n + m_e)
m_HeII = 2.0 * (m_p + m_n) + m_e
Y = 0.2477                      # Primordial helium abundance by mass
y = Y / 4. / (1. - Y)           # Primordial helium abundance by number

lsun = 3.839e33                     # Solar luminosity - erg / s
cm_per_rsun = 695500.0 * 1e5        # Radius of the sun - [cm_per_rsun] = cm
t_edd = sigma_T * c / 4. / pi / G / m_p # Eddington timescale (~0.45 Gyr)

xcorr = (3. / 7.)**0.5 * (6. / 7.)**3  # Correction factor for peak temperature location in disk

flux_AB = 3631. * 1e-23 # 3631 * 1e-23 erg / s / cm**2 / Hz
