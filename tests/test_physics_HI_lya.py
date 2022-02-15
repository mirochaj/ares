"""

test_physics_HI_lya.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Mon 14 Feb 2022 15:24:50 EST

Description:

"""

import numpy as np
import matplotlib.pyplot as pl
from ares.physics import Hydrogen
from ares.simulations import Global21cm

Tarr = np.logspace(-1, 2)

hydr = Hydrogen()

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
