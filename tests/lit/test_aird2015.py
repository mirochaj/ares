"""
test_Lx(z)

Author: Jacob Jost
Affiliation: University of Colorado at Boulder (Undergraduate)
Created on: Thu June 4 09:00:00 MDT 2015

--- = spacing between different sections of code

The redshift can be set so the same redshift is used for each plot or can be set 
for each individual plot. You either need to keep the top z for the overall or 
hash it out and unhash the other z's to set individually.
"""

import ares
import numpy as np
import matplotlib.pyplot as pl

a15 = ares.util.read_lit('aird2015') 

z = 1.0
#------------------------------------------------------------

#z = 5.0
L = np.logspace(41, 47, 100)

fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)

models = []
for p, Lx in enumerate(L):
    model = a15.LuminosityFunction_LDDE1_hardband(Lx, z)
    models.append(model)
models = np.array(models)


pl.loglog(L, models, color = 'k', label = r'LDDE1-Hard Band')
#pl.title('2-7 KeV LDDE1 at z ~ %.1f' % (z))
pl.ylim(10**-9.1, 10**-2)

ax1.set_xlabel(r'$L_X$')    
ax1.set_ylabel(r'$\phi(L_X)$')
ax1.legend(loc='best')

#------------------------------------------------------------

#z = 5.0
L = np.logspace(41, 47, 100)

fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)

models = []
for p, Lx in enumerate(L):
    model = a15.LuminosityFunction_LDDE2_hardband(Lx, z)
    models.append(model)
models = np.array(models)

pl.loglog(L, models, color = 'g', label = r'LDDE2-Hard Band')
#pl.title('2-7 KeV LDDE2 at z ~ %.1f' % (z))
pl.ylim(10**-9.1, 10**-2)

ax1.set_xlabel(r'$L_X$')    
ax1.set_ylabel(r'$\phi(L_X)$')
ax1.legend(loc='best')
pl.show()

#------------------------------------------------------------

#z = 5.0
L = np.logspace(41, 47, 100)

fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)

models = []
for p, Lx in enumerate(L):
    model = a15.LuminosityFunction_LDDE1_softband(Lx, z)
    models.append(model)
models = np.array(models)


pl.loglog(L, models, color = 'r', label = r'LDDE1-Soft Band')
#pl.title('0.5-2 KeV LDDE1 at z ~ %.1f' % (z))
pl.ylim(10**-9.1, 10**-2)
ax1.set_xlabel(r'$L_X$')    
ax1.set_ylabel(r'$\phi(L_X)$')
ax1.legend(loc='best')

#------------------------------------------------------------

#z = 5.0
L = np.logspace(41, 47, 100)

fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)

models = []
for p, Lx in enumerate(L):
    model = a15.LuminosityFunction_LDDE2_softband(Lx, z)
    models.append(model)
models = np.array(models)


pl.loglog(L, models, color = 'b', label = r'LDDE2-Soft Band')
pl.title(r'Different models for $L_X$ for soft and hard bands at $z$ ~ $%.1f$' % (z))
pl.ylim(10**-9.1, 10**-2) 

ax1.set_xlabel(r'$L_X$')    
ax1.set_ylabel(r'$\phi(L_X)$')
ax1.legend(loc='best')
