"""

test_21cm_parameterized_rates.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Nov 17 14:21:53 MST 2014

Description: 

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

sim = ares.simulations.Global21cm(is_ion_src_igm=False)
sim.run()

anl = ares.analysis.Global21cm(sim)
ax1 = anl.GlobalSignature()

# Parameterized evolution (const. in time)
def Gamma_cgm_func(z, species=0, **kwargs):
    return np.interp(z, sim.history['z'][-1::-1], 
        sim.history['cgm_Gamma_h_1'][-1::-1])
    
def eheat_func(z, species=0, **kwargs):
    return np.interp(z, sim.history['z'][-1::-1], 
        sim.history['igm_heat_h_1'][-1::-1])

def Ja_func(z, **kwargs):
    return np.interp(z, sim.history['z'][-1::-1], 
        sim.history['Ja'][-1::-1])

# Simulate it numerically! (parameterized everything)
sim2 = ares.simulations.Global21cm(Gamma_cgm=Gamma_cgm_func, 
    heat_igm=eheat_func, Ja=Ja_func, is_ion_src_igm=False)
sim2.run()

anl2 = ares.analysis.Global21cm(sim2)
anl2.GlobalSignature(ax=ax1, color='b')

ax2 = anl.TemperatureHistory(fig=2)
anl2.TemperatureHistory(ax=ax2, color='b')
pl.draw()

ax3 = anl.IonizationHistory(fig=3, color='k', show_xe=False, 
    show_xibar=False)
anl2.IonizationHistory(ax=ax3, color='b', show_xe=False, 
    show_xibar=False)

pl.draw()

