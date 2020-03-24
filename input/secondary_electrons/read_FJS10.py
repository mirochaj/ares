"""
read_FJS10.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-09-20.

Description: Read in the Furlanetto & Stoever 2010 results, and make new tables 
that will be more suitable for ARES.

Notes: Run this script inside of whatever directory you download the Furlanetto 
& Stoever 2010 results in.  It will produce files called secondary_electron*.dat, 
and one file called secondary_electron_data.hdf5.  This last one is the most 
important, and will be used when secondary_ionization = 3.
     
"""

import os
import sys
import h5py
import numpy as np
    
E_th = [13.6, 24.6, 54.4]

# Ionized fraction points and corresponding files
x = np.array([1.0e-4, 2.318e-4, 4.677e-4, 1.0e-3, 2.318e-3, 
              4.677e-3, 1.0e-2, 2.318e-2, 4.677e-2, 1.0e-1, 
              0.5, 0.9, 0.99, 0.999])

xi_files = ['xi_0.999.dat', 'xi_0.990.dat', 'xi_0.900.dat', 'xi_0.500.dat', 
         'log_xi_-1.0.dat', 'log_xi_-1.3.dat', 'log_xi_-1.6.dat',
         'log_xi_-2.0.dat', 'log_xi_-2.3.dat', 'log_xi_-2.6.dat',
         'log_xi_-3.0.dat', 'log_xi_-3.3.dat', 'log_xi_-3.6.dat',
         'log_xi_-4.0.dat']

xi_files.reverse()
         
# Make some blank arrays         
energies = np.zeros(258)
heat = np.zeros([len(xi_files), 258])
fion = np.zeros_like(heat)
fexc = np.zeros_like(heat)
fLya = np.zeros_like(heat)
fHI = np.zeros_like(heat)
fHeI = np.zeros_like(heat)
fHeII = np.zeros_like(heat)

# Read in energy and fractional heat deposition for each ionized fraction.
for i, fn in enumerate(xi_files):
          
    # Read data 
    nrg, f_ion, f_heat, f_exc, n_Lya, n_ionHI, n_ionHeI, n_ionHeII, \
        shull_heat = np.loadtxt('x_int_tables/{!s}'.format(fn), skiprows=3,
        unpack=True)
       
    if i == 0:          
        for j, energy in enumerate(nrg):
            energies[j] = energy
    
    for j, h in enumerate(f_heat):
        heat[i][j] = h
        fion[i][j] = f_ion[j]
        fexc[i][j] = f_exc[j]
        fLya[i][j] = (n_Lya[j] * 10.2) / energies[j]
        fHI[i][j] = (n_ionHI[j] * E_th[0]) / energies[j]
        fHeI[i][j] = (n_ionHeI[j] * E_th[1]) / energies[j]
        fHeII[i][j] = (n_ionHeII[j] * E_th[2]) / energies[j]      
          
# We also want the heating as a function of ionized fraction for each photon energy.        
heat_xi = np.array(list(zip(*heat)))
fion_xi = np.array(list(zip(*fion)))
fexc_xi = np.array(list(zip(*fexc)))
fLya_xi = np.array(list(zip(*fLya)))
fHI_xi = np.array(list(zip(*fHI)))
fHeI_xi = np.array(list(zip(*fHeI)))
fHeII_xi = np.array(list(zip(*fHeII)))

# Write to hfd5          
f = h5py.File('secondary_electron_data.hdf5', 'w')
f.create_dataset('electron_energy', data=energies)
f.create_dataset('ionized_fraction', data=np.array(x))
f.create_dataset('f_heat', data=heat_xi)
f.create_dataset('fion_HI', data=fHI_xi)
f.create_dataset('fion_HeI', data=fHeI_xi)
f.create_dataset('fion_HeII', data=fHeII_xi)
f.create_dataset('f_Lya', data=fLya_xi)
f.create_dataset('fion', data=fion_xi)
f.create_dataset('fexc', data=fexc_xi)
f.close()

    

