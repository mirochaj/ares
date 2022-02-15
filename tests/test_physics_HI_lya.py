"""

test_physics_HI_lya.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Mon 14 Feb 2022 15:24:50 EST

Description:

"""

import sys
import numpy as np
import matplotlib.pyplot as pl
from ares.physics import Hydrogen
from ares.simulations import Global21cm

Tarr = np.logspace(-1, 2)

hydr = Hydrogen(approx_Salpha=0)

z = 22.
Tk = 10.#hydr.cosm.get_Tgas(z)
x = np.arange(-100, 100)

fig, axes = pl.subplots(1, 2, figsize=(8, 4))

Jc = np.array([hydr.get_lya_profile(z, Tk, xx, continuum=1) for xx in x])
Ji = np.array([hydr.get_lya_profile(z, Tk, xx, continuum=0) for xx in x])

Ji_norm = Ji * Jc[x==0]
Ji_norm[x < 0] = Jc[x < 0]

axes[0].plot(x, Jc, color='k')
axes[1].plot(x, Ji_norm, color='k')

for ax in axes:
    ax.axhline(1, color='k', ls=':')
    ax.axvline(0, color='k', ls=':')
    ax.set_xlim(-105, 105)
    ax.set_ylim(-0.05, 1.05)


Ic = hydr.get_lya_int(z, Tk, Ja=1, continuum=1)
Ii = hydr.get_lya_int(z, Tk, Ja=1, continuum=0)

print(Ic, Ii)

print(hydr.get_lya_heating(z, Tk, Jc=1e-21, Ji=0.0))

sys.exit(0)

sim1 = Global21cm(fX=0.1, lya_heating=0)
sim2 = Global21cm(fX=0.1, lya_heating=1)

ax2 = None
colors = 'k', 'b'
ls = '-', '--'
for i, sim in enumerate([sim1]):
    sim.run()
    sim.TemperatureHistory(ax=ax2, fig=2, color=colors[i], ls=ls[i])
