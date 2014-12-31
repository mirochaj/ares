"""

test_sawtooth.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Aug 16 12:59:40 MDT 2013

Description: This is very similar to Haiman, Abel, & Rees (1997) Fig. 1.

"""

import ares, time
import numpy as np
import matplotlib.pyplot as pl
from ares.physics.Constants import E_LL, J21_num

pars = \
{
 'source_type': 'star',
 'source_temperature': 1e5,
 'Tmin': 1e3,
 'fstar': 1e-2,
 'spectrum_type': 'bb',
 'spectrum_Emin': 5.,
 'spectrum_Emax': E_LL,
 'spectrum_EminNorm': 0.01,
 'spectrum_EmaxNorm': 5e2,
 'approx_He': False,
 'approx_xrb': False,
 'approx_lwb': False,
 'lya_nmax': 8, 
 'Nlw': 9690.,
 'norm_by': 'lw',
 'is_ion_src_cgm': False,
 'is_ion_src_igm': False,
 'is_heat_src_igm': False,
 'initial_redshift': 40,
 'final_redshift': 10,
}
    
rad = ares.solvers.UniformBackground(discrete_lwb=False, **pars)
E = np.linspace(5.0, E_LL-0.01, 500)

color = 'k'
for zf in [15, 30]:
    t1 = time.time()
    F = map(lambda EE: rad.AngleAveragedFlux(zf, EE, xavg=lambda z: 0.0, 
        energy_units=False), E)
    t2 = time.time()
    
    Fthin = map(lambda EE: rad.AngleAveragedFlux(zf, EE, tau=0.0,
        energy_units=False), E)

    pl.semilogy(E, np.array(F) / J21_num, color=color, label=r'$z=%i$' % zf)
    pl.semilogy(E, np.array(Fthin) / J21_num, color=color, ls='--')
    
    color = 'b'

pl.xlabel(r'$h\nu \ (\mathrm{eV})$')
pl.ylabel(r'$J_{\nu} / J_{21}$')
pl.legend(loc='center left', frameon=False)
pl.ylim(1e-3, 1e2)

# Solve using generator
rad2 = ares.solvers.UniformBackground(discrete_lwb=True, **pars)
z, E_n, flux_n = rad2.LWBackground()

for i, nrg in enumerate(E_n):
    t3 = time.time()
    pl.scatter(nrg, flux_n[i][np.argmin(np.abs(z-15.)),:] / J21_num,
        facecolors='none', color='k', marker='|', s=100, alpha=0.05)
    t4 = time.time()
    
    pl.scatter(nrg, flux_n[i][np.argmin(np.abs(z-30.)),:] / J21_num,
        facecolors='none', color='b', marker='|', s=100, alpha=0.05)

print "Generator provides %ix speed-up." % ((t2 - t1) / (t4 - t3))

# Compute Lyman-alpha flux
zarr = np.arange(10, 40)
Ja_1 = np.array(map(rad.LymanAlphaFlux, zarr))
Ja_2 = rad2.LymanAlphaFlux(z=None, fluxes=flux_n)
    
fig = pl.figure(2)
ax = fig.add_subplot(111)

ax.scatter(zarr, Ja_1 / J21_num, color='k', facecolors='none', s=50,
    label='numerical')
ax.semilogy(z, Ja_2 / J21_num, color='b', label='generator')
ax.set_xlabel(r'$z$')
ax.set_ylabel(r'$J_{\alpha} / J_{21}$')
ax.set_ylim(1e-4, 1e2)
ax.legend(loc='lower left')
pl.draw()


