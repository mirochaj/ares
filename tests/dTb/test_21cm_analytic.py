"""

test_21cm_analytic.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Nov 17 14:21:53 MST 2014

Description: Constant ionization, no heating, IGM evolution for single zone.

"""

import ares
import numpy as np
import matplotlib.pyplot as pl
from ares.physics.Constants import k_B

# 
## INPUT
zarr = np.linspace(8, 40)
Gamma = 5e-18    # constant ionized fraction
heat = 0         # no heat
xi0 = 1e-8       # initial ionized fraction in IGM
Ja = 2e-11
##
#

zi = zarr.max()

hydr = ares.physics.Hydrogen()
        
z_to_t = lambda z, zi=zi: hydr.cosm.LookbackTime(z, zi)        
        
C = 1. - xi0
def xi(z, Gamma=Gamma):
    t = z_to_t(z)
    
    return 1. - C * np.exp(-Gamma * t)
    
def Tk(z, eheat=heat):
    """
    Analytic approximation to thermal history.

    Assumes constant heating rate. 
    """
    
    norm = 2. / 3. / hydr.cosm.nH0 / (1 + hydr.cosm.y) / k_B
    
    t = z_to_t(z)
    
    # Heating term + adiabatic cooling term
    return norm * eheat * t + hydr.cosm.Tgas(z)

# Brightness temperature (arguments: z, xHII, Ts)
dTb = lambda z: hydr.DifferentialBrightnessTemperature(z, xi(z), Tk(z))

# Parameterized evolution (const. in time)
def Gamma_cgm_func(z, species=0, **kwargs):
    # Rate coefficient!
    return Gamma

def eheat_func(z, species=0, **kwargs):
    return heat

def Ja_func(z, **kwargs):
    return Ja

# Simulate it numerically! (parameterized everything)
sim = ares.simulations.Global21cm(initial_ionization=[xi0],
    final_redshift=zarr.min(), load_ics=False, initial_redshift=zarr.max(),
    Gamma_cgm=Gamma_cgm_func, recombination=0, secondary_ionization=0, 
    heat_igm=eheat_func, Ja=Ja_func, is_ion_src_igm=False, approx_He=0)
sim.run()

anl = ares.analysis.Global21cm(sim)

ax1 = anl.TemperatureHistory(fig=1, label='numerical')
ax1.scatter(zarr, map(Tk, zarr), color='b', label='analytic', s=100,
    facecolors='none')
ax1.legend(loc='lower right', fontsize=14)
ax1.plot(zarr, 2.725 * (1. + zarr), color='k', ls=':')

pl.draw()

ax2 = anl.IonizationHistory(fig=2, color='k', show_xe=False, 
    show_xibar=False, label='numerical')
ax2.scatter(zarr, map(xi, zarr), color='b', label='analytic',
    facecolors='none', s=100)
ax2.legend(loc='lower left', fontsize=14)
ax2.set_ylim(1e-9, 1.5)

pl.draw()

#ax3 = anl.GlobalSignature(fig=3, color='k', label='numerical', xaxis='z',
#    z_ax=False)
#ax3.scatter(zarr, map(dTb, zarr), color='b', label='analytic',
#    facecolors='none', s=100 )
#
#ax3.legend(loc='upper right', fontsize=14)
#
#pl.draw()
#
