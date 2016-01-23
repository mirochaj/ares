"""

test_mz.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Wed Jan 13 09:58:02 PST 2016

Description: 

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

pars = \
{
 'pop_Tmin': 1e4,
 'pop_Macc': 'mcbride2009',

 'pop_kappa_UV': 1.15e-28,
 #'pop_sed': 'leitherer1999',
 
 # Dust
 'pop_lf_dustcorr': True,
 'pop_lf_beta': -2.,
 
 'sfe_Mfun': 'lognormal',
 
 'pop_lf_Mmax': 1e15,
  
 'sfe_fpeak': 'pl',
 'sfe_Mpeak': 'pl',
 'sfe_sigma': 'constant',
  
 'sfe_Mpeak_par0': 5e11,
 'sfe_Mpeak_par1': -1.,
 'sfe_Mpeak_par2': 0.2,
 'sfe_fpeak_par0': 0.3,
 'sfe_fpeak_par1': -1.,
 'sfe_fpeak_par2': 0.2,
 'sfe_sigma_par0': 0.8,
 'sfe_sigma_par1': None,

 'sfe_Mfun_lo': 'pl',
 'sfe_Mfun_lo_par0': 1e10,
 'sfe_Mfun_lo_par1': 0.33,
 
 'sfe_Mfun_hi': 'exp',
 'sfe_Mfun_hi_par0': 2e13,
 'sfe_Mfun_hi_par1': 0.1, #could force less than Mpeak

}

b15 = ares.util.read_lit('bouwens2015')

# Plot models compared to observational data
M = np.linspace(-24, -10, 40)
L = np.logspace(25, 32, 40)
Mh = np.logspace(7, 13)

fig0 = pl.figure(0); ax0 = fig0.add_subplot(111)
fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)
fig2 = pl.figure(2); ax2 = fig2.add_subplot(111)

extrap = [{'sfe_Mfun_lo': None, 'sfe_Mfun_hi': None},
          {'sfe_Mfun_lo': 'pl', 'sfe_Mfun_hi': None},
          {'sfe_Mfun_lo': 'pl', 'sfe_Mfun_hi': 'exp'},
          {'sfe_Mfun_lo': None, 'sfe_Mfun_hi': 'exp'}]

extrap = [{'sfe_Mfun_lo': 'pl', 'sfe_Mfun_hi': None},
          {'sfe_Mfun_lo': 'pl', 'sfe_Mfun_hi': 'exp'}]

marker = ['o', '^']
ls = ['-', '--', ':', '-.']
for k, ext in enumerate(extrap):

    pars.update(ext)
    pop = ares.populations.GalaxyPopulation(**pars)

    colors = ['k', 'b', 'g', 'r', 'c', 'y', 'm']
    for i, z in enumerate(b15.redshifts[:1]):
        if k == 0:
            ax0.errorbar(b15.data['lf'][z]['M'], b15.data['lf'][z]['phi'],
                yerr=b15.data['lf'][z]['err'], color=colors[i+1], fmt='o')
        
        phi_I_1 = pop.LuminosityFunction(z, M, mags=True)
        phi_I_2 = pop.LuminosityFunction(z, L, mags=False)
        
        mag, phi1 = pop.phi_of_M(z)
        lum, phi2 = pop.phi_of_L(z)
        
        ax0.semilogy(mag, phi1, color=colors[i], ls=ls[k])
        ax2.loglog(lum, phi2, color=colors[i], ls=ls[k])

        ax0.scatter(M, phi_I_1, color=colors[i], marker=marker[k], facecolors='none')
        ax2.scatter(L, phi_I_2, color=colors[i], marker=marker[k], facecolors='none')

        ax1.loglog(Mh, pop.fstar(z, Mh), color=colors[i], ls=ls[k])

ax0.set_xlabel(r'$M_{\mathrm{UV}}$')
ax0.set_ylabel(r'$\phi(M)$')
ax0.set_ylim(1e-7, 1e2)
ax0.set_xlim(-25, 0)
ax1.set_ylim(1e-3, 1)
ax2.set_xlim(1e20, 1e32)
ax2.set_ylim(1e-35, 1e-18)
pl.show()


