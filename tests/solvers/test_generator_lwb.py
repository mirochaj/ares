"""

test_generator_lwb.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Aug 16 12:59:40 MDT 2013

Description: This is very similar to Haiman, Abel, & Rees (1997) Fig. 1.

"""

import ares, time
import numpy as np
import matplotlib.pyplot as pl
from ares.physics.Constants import E_LL, E_LyA, J21_num

pars = \
{
 'source_type': 'star',
 'source_temperature': 1e5,
 'Tmin': 1e3,
 'fstar': 1e-2,
 'source_sed': 'bb',
 'source_Emin': E_LyA,
 'source_Emax': E_LL,
 'source_EminNorm': 0.01,
 'source_EmaxNorm': 5e2,
 'approx_lwb': False,
 'lya_nmax': 8,
 'Nlw': 9690.,
 'norm_by': 'lw',
 'initial_redshift': 50,
 'final_redshift': 10,
 'redshifts_lwb': 1e4,
 'include_H_Lya': True,
}
    
# Solve using brute-force integration
rad = ares.solvers.UniformBackground(discrete_lwb=False, **pars)
E = np.linspace(5.0, E_LL-0.01, 500)

fig1 = pl.figure(1)
ax1 = fig1.add_subplot(111)

color = 'k'
for zf in [15, 30]:
    t1 = time.time()
    F = map(lambda EE: rad.AngleAveragedFlux(zf, EE, xavg=lambda z: 0.0, 
        energy_units=False), E)
    t2 = time.time()
    
    ax1.semilogy(E, np.array(F) / J21_num, color=color, label=r'$z=%i$' % zf)
    
    # Compute optically thin solution for comparison
    Fthin = map(lambda EE: rad.AngleAveragedFlux(zf, EE, tau=0.0,
        energy_units=False), E)
    
    
    ax1.semilogy(E, np.array(Fthin) / J21_num, color=color, ls='--')
    
    color = 'b'

ax1.set_xlabel(r'$h\nu \ (\mathrm{eV})$')
ax1.set_ylabel(r'$J_{\nu} / J_{21}$')
ax1.legend(loc='lower left', frameon=False, fontsize=16)
ax1.set_ylim(1e-3, 1e2)
pl.draw()

# Solve using generator
rad2 = ares.simulations.MetaGalacticBackground(discrete_lwb=True, **pars)

t3 = time.time()
rad2.run()
t4 = time.time()    

# Grab info about radiation background
z, E, flux = rad2.get_history()
Eflat = np.concatenate(E)

# Get line emission
z, E, flux_l = rad2.get_history(continuum=False)

ax1.scatter(Eflat, flux[np.argmin(np.abs(z-15.)),:] / J21_num,
    facecolors='none', color='k', marker='|', s=100, alpha=0.05)

ax1.scatter(Eflat, flux[np.argmin(np.abs(z-30.)),:] / J21_num,
    facecolors='none', color='b', marker='|', s=100, alpha=0.05)

# Emphasize Ja
totz15 = flux[np.argmin(np.abs(z-15.)),:] + flux_l[np.argmin(np.abs(z-15.)),:]
totz30 = flux[np.argmin(np.abs(z-30.)),:] + flux_l[np.argmin(np.abs(z-30.)),:]
ax1.scatter(Eflat[0], totz15[0] / J21_num,
    facecolors='none', color='k', marker='o', s=100)
ax1.scatter(Eflat[0], totz30[0] / J21_num,
    facecolors='none', color='b', marker='o', s=100)


pl.draw()
print "Generator provides %.2gx speed-up (per redshift point)." \
    % ((t2 - t1) / ((t4 - t3) / z.size))

# Compute Lyman-alpha flux
zarr = np.arange(10, 40)
Ja_1 = np.array(map(rad.LymanAlphaFlux, zarr))

# Read it in from discrete LWB calculation
Ja_2 = flux[:,0] + flux_l[:,0]
 
fig2 = pl.figure(2)
ax2 = fig2.add_subplot(111)

ax2.scatter(zarr, Ja_1 / J21_num, color='k', facecolors='none', s=50,
    label='numerical')
ax2.semilogy(z, Ja_2 / J21_num, color='b', label='generator')
ax2.set_xlabel(r'$z$')
ax2.set_ylabel(r'$J_{\alpha} / J_{21}$')
ax2.set_ylim(1e-4, 1e2)
ax2.legend(loc='lower left')
pl.draw()


