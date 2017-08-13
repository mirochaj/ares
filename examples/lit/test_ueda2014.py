"""
test_Lx(z)

Author: Jacob Jost
Affiliation: University of Colorado at Boulder (Undergraduate)
Created on: Thu June 4 09:00:00 MDT 2015

NOTE: plots may differ than Ueda et al. (2014) because their plots show the De-absorbed
XLF and the redshift ranges arebinned, also my be cutting the plots off at a certain value***
"""

import ares
import numpy as np
from scipy import integrate
import matplotlib.pyplot as pl

u03 = ares.util.read_lit('ueda2003')
u14 = ares.util.read_lit('ueda2014')
z = np.linspace(2.4, 3.0, 100)
L = np.logspace(42.5, 47, 100)

"""
Compare to Ueda et al. (2003) LDDE model.
"""

fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)

"""
Plot the QSO-LF at a few redshifts, show PLE, PDE, LDDE models.
"""
fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)

colors = 'k', 'b', 'g'
for i, redshift in enumerate(z):
    lf_ple = u03.LuminosityFunction(L, redshift, evolution='ldde')
    lf_pde = u14.LuminosityFunction(L, redshift, evolution='ldde')
    
    ax1.loglog(L, lf_ple, color=colors[i], ls='--')
    ax1.loglog(L, lf_pde, color=colors[i], ls='-', label=r'$z=%i$' % redshift)
    ax1.loglog(L, lf_ldde, color=colors[i], ls=':')
    
ax1.set_xlabel(r'$L_X$')    
ax1.set_ylabel(r'$\phi(L_X)$')
ax1.legend(loc='best')


#---------------------------------------------------

#models = []
#r = []
#for i, redshift in enumerate(z):
#    model1 = []
#    models.append(model1)
#    r.append(redshift)
#    for p, Lx in enumerate(L):
#        model = u14.LuminosityFunction_LDDE(Lx, redshift, **u14.qsolf_LDDE2_hardpars)
#        model1.append(model)
#models = np.array(models) 
#r = np.array(r)
#
#for i, j in enumerate(models):
#    pl.loglog(L, models[i], alpha = 0.25)
#    pl.title(r'The 2-10 KeV Band at $z$ ~ $%.1f - %.1f$' % (r[0], r[-1]))
#    pl.ylim(10**-12, 10**-2)
#    pl.xlim(L[0], L[-23])
#
#ax1.set_xlabel(r'$L_X$')    
#ax1.set_ylabel(r'$\phi(L_X)$')
#ax1.legend(loc='best')
#pl.show()
#
##---------------------------------------------------
#
#"""
#test_Lx(z)
#
#Author: Jacob Jost
#Affiliation: University of Colorado at Boulder (Undergraduate)
#Created on: Thu June 4 09:00:00 MDT 2015
#
#NOTE: 
#    plots may differ than Ueda et al. (2014) because their plots show the De-absorbed
#    XLF and the redshift ranges arebinned, also my be cutting the plots off at a certain value***
#
#    HEADS UP: this takes a while to run, use caution. 
#"""
#
##---------------------------------------------------
#
#u14 = ares.util.read_lit('ueda2014')
#z = np.linspace(0, 5, 100)
#models = 100
#
#
#hardpars = u14.qsolf_LDDE2_hardpars #parameters dictionary
#harderr = u14.qsolf_LDDE2_harderr #parameters error dictionary
#hardall = hardpars.copy() #copying parameters dictionary
#hardall.update(harderr) #combining parameters dictionary & parameters error dictionary
#hardsamples = u14.randomsamples(100, **hardall)
#
#fig2 = pl.figure(2); ax2 = fig2.add_subplot(111)
#
#integrand = []
#for i in range(models):
#    integrand1 = []
#    r = []
#    integrand.append(integrand1)
#    for j in range(len(z)):
#        x = lambda Lx: u14.LuminosityFunction_LDDE(Lx, z[j],\
#        **hardsamples[i])
#        p, err = integrate.quad(x, 10**41, 10**46)
#        r.append(z[j])
#        integrand1.append(p)
#          
#for i, j in enumerate(integrand):
#    pl.semilogy(z, integrand[i], color = 'k', alpha = 0.25)
#
#pl.title(r'The 2-10 KeV Band at $z$ ~ $%.1f - %.1f$' % (r[0], r[-1]))
#
#ax2.set_xlabel(r'$z$')    
#ax2.set_ylabel(r'$\phi(L_X)$')
#ax2.legend(loc='best')
#pl.show()
#
##---------------------------------------------------
#
#"""
#test_Lx(z)
#
#Author: Jacob Jost
#Affiliation: University of Colorado at Boulder (Undergraduate)
#Created on: Thu June 4 09:00:00 MDT 2015
#
#NOTE: plots may differ than Ueda et al. (2014) because their plots show the De-absorbed
#XLF and the redshift ranges arebinned, also my be cutting the plots off at a certain value***
#"""
#
#z = np.linspace(2.4, 3.0, 100)
#L = np.logspace(42.5, 47, 100)
#fig3 = pl.figure(3); ax3 = fig3.add_subplot(111)
#
#hardpars = u14.qsolf_LDDE2_hardpars #parameters dictionary
#harderr = u14.qsolf_LDDE2_harderr #parameters error dictionary
#hardall = hardpars.copy() #copying parameters dictionary
#hardall.update(harderr) #combining parameters dictionary & parameters error dictionary
#hardsamples = u14.randomsamples(100, **hardall)
#
#results = []
#r = []
#for i, redshift in enumerate(z):
#    model1 = []
#    results.append(model1)
#    r.append(redshift)
#    for p, Lx in enumerate(L):
#        model = u14.LuminosityFunction_LDDE(Lx, redshift,\
#        **hardsamples[i])
#        model1.append(model)
#
#          
#for i, j in enumerate(results):
#    pl.loglog(L, results[i], color = 'r', alpha = 0.25)
#
#pl.title(r'The 2-10 KeV Band at $z$ ~ $%.1f - %.1f$' % (r[0], r[-1]))
#pl.ylim(10**-12, 10**-4)
#pl.xlim(L[0], L[-23])
#ax1.set_xlabel(r'$L_X$')    
#ax1.set_ylabel(r'$\phi(L_X)$')
#ax1.legend(loc='best')
#pl.show()