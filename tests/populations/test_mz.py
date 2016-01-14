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
 #'pop_model': 'ham',
 'pop_Macc': 'mcbride2009',

 #'pop_kappa_UV': 1.15e-28,
 'pop_sed': 'leitherer1999',
 
 'pop_sfe_Mfun': 'lognormal',
 
 'pop_sfe_fpeak': 'constant',
 'pop_sfe_Mpeak': 'constant',
 'pop_sfe_sigma': 'constant',
 
 'pop_sfe_Mpeak_par0': 5e11,
 'pop_sfe_Mpeak_par1': None,
 'pop_sfe_fpeak_par0': 0.3,
 'pop_sfe_fpeak_par1': None,
 'pop_sfe_sigma_par0': 0.8,
 'pop_sfe_sigma_par1': None,
 'pop_lf_Mmax': 1e13,

 'pop_sfe_Mfun_lo': 'pl',
 'pop_sfe_Mfun_lo_par0': 1e10,
 'pop_sfe_Mfun_lo_par1': 0.33,
 
 'pop_sfe_Mfun_hi': 'exp',
 'pop_sfe_Mfun_hi_par0': 0.1,
 'pop_sfe_Mfun_hi_par1': 1e10, #could force less than Mpeak

 # Dust
 'pop_lf_dustcorr': True,
 'pop_lf_beta': -2.,
}

b15 = ares.util.read_lit('bouwens2015')

pop = ares.populations.GalaxyMZ(**pars)

# Plot models compared to observational data
M = np.arange(-24, -10, 0.05)
Mh = np.logspace(7, 13)

fig0 = pl.figure(0); ax0 = fig0.add_subplot(111)
fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)

ls = ['-', '--', ':']
for k, norm in enumerate([0.1, 0.3]):

    pars['pop_sfe_fpeak_par0'] = norm

    colors = ['k', 'b', 'g', 'r', 'c', 'y', 'm']
    for i, z in enumerate(b15.redshifts[::2]):
        if k == 0:
            ax0.errorbar(b15.data['lf'][z]['M'], b15.data['lf'][z]['phi'],
                yerr=b15.data['lf'][z]['err'], color=colors[i], fmt='o')
        phi = pop.LuminosityFunction(z, M, mags=True)
        
        ax0.semilogy(M + pop.A1600(z, M), phi, color=colors[i], ls=ls[k])

        ax1.loglog(Mh, pop.fstar(z, Mh), color=colors[i], ls=ls[k])

ax0.set_xlabel(r'$M_{\mathrm{UV}}$')
ax0.set_ylabel(r'$\phi(M)$')
ax0.set_ylim(1e-7, 1e-1)
ax0.set_xlim(-25, -10)

pl.show()


