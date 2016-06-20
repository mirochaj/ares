"""

test_analytic.py

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
zarr = np.linspace(10., 40)
Gamma = 5e-18    # constant ionizing background
heat = 0.0       # no heat
Ja = 2e-11       # constant Ly-a background
##
#

zi = zarr.max()
hydr = ares.physics.Hydrogen()
inits = ares.util.ReadData._load_inits()        

xi_CR = xi0 = np.interp(zi, inits['z'], inits['xe'])
Ti_CR = np.interp(zi, inits['z'], inits['Tk'])
        
z_to_t = lambda z, zi=zi: hydr.cosm.LookbackTime(z, zi)        
        
def xi(z, Gamma=Gamma):
    """
    Analytic approximation to the ionization history.

    Assumes constant ionization rate, no recombinations.
    """
    
    C = 1. - xi0 # constant of integration
    
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

# Parameterized evolution (const. in time)
def Gamma_igm_func(z, species=0, **kwargs):
    # Rate coefficient!
    return Gamma

def eheat_func(z, species=0, **kwargs):
    return heat
    
pars = \
{
 'load_ics': True, 
 'final_redshift': zarr.min(), 
 'initial_redshift': zarr.max(),
 'pop_k_ion_igm': Gamma_igm_func, 
 'pop_k_heat_igm': eheat_func, 
 'igm_recombination': 0, 
 'secondary_ionization': 0,
 'include_cgm': False, 
 'approx_He': 0,
 'igm_compton_scattering': False, # ionization is unrealistically high, so
                                  # artifically turn this off to prevent heat
}

# Simulate it numerically! (parameterized everything)
sim = ares.simulations.MultiPhaseMedium(**pars)
sim.run()

anl = ares.analysis.MultiPhaseMedium(sim)

ax1 = anl.TemperatureHistory(fig=1, label='numerical')
ax1.scatter(zarr, map(Tk, zarr), color='b', label='analytic', s=100,
    facecolors='none')
ax1.plot(zarr, 2.725 * (1. + zarr), color='k', ls=':', label=r'$T_{\gamma}$')

# Show against CosmoRec solution

ax1.plot(inits['z'], inits['Tk'], color='r', ls='--', label='CosmoRec')
ax1.set_xlim(zarr.min(), zarr.max())
ax1.legend(loc='lower right', fontsize=14)
ax1.set_yscale('linear')
ax1.set_ylim(0, 40)
pl.draw()

ax2 = anl.IonizationHistory(fig=2, color='k', label='numerical', zone='igm')
ax2.scatter(zarr, map(xi, zarr), color='b', label='analytic',
    facecolors='none', s=100)
ax2.legend(loc='lower left', fontsize=14)
ax2.set_ylim(1e-4, 1.5)

pl.draw()

