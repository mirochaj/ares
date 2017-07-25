"""

test_physics_HI_beta.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Fri Sep 23 15:58:34 PDT 2016

Description: Make a plot like in Furlanetto, Oh, & Briggs (2006) 
Figure 9. Just use a tanh model because it's fast.

"""

import ares

hydr = ares.physics.Hydrogen()

# Just to get reasonable histories
sim = ares.simulations.Global21cm(tanh_model=True, 
    output_redshifts=np.linspace(4, 100), interp_cc='linear')

z = sim.history['z']
Tk = sim.history['igm_Tk']
xHI = 1. - sim.history['cgm_h_2']
Ja = sim.history['Ja']
ne = (1. - sim.history['igm_h_1']) * hydr.cosm.nH(z)

pl.semilogx(1+z, hydr.beta_d(z, Tk, 1.-xHI, ne, Ja), color='k', ls='-',
    label=r'$\beta$')
pl.semilogx(1+z, hydr.beta_x(z, Tk, 1.-xHI, ne, Ja), color='b', ls=':',
    label=r'$\beta_x$')
pl.semilogx(1+z, hydr.beta_a(z, Tk, 1.-xHI, ne, Ja), color='g', ls='-.',
    label=r'$\beta_{\alpha}$')
pl.semilogx(1+z, hydr.beta_T(z, Tk, 1.-xHI, ne, Ja), color='r', ls='--',
    label=r'$\beta_T$')

pl.legend(loc='lower left', frameon=True, fontsize=14)
pl.xlabel(r'$1+z$')
pl.ylabel(r'$\beta_i$')
pl.xlim(5, 100)
pl.ylim(-2, 2)

