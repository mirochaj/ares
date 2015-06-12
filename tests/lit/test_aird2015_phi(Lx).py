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

a15 = ares.util.read_lit('aird2015_phi(Lx)') 
Legend = ['Green = LDDE1 softband', 'Red = LDDE1 hardband', 'Blue =  LDDE2 softband', 'Black = LDDE2 hardband']
#z = 1.0
#------------------------------------------------------------

z = 5.0
L = np.logspace(41, 47, 100)
m = 1000

fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)

models = []
for t in range(m):
    model = a15.LuminosityFunction_LDDE1_hardband(L, z, 10000, 1)
    models.append(model)

for i, j in enumerate(models):
    pl.loglog(L, models[i], color = 'r', alpha = 0.1)
    
#pl.title('2-7 KeV LDDE1 at z ~ %.1f' % (z))
pl.ylim(10**-9.1, 10**-2)

ax1.set_xlabel(r'$L_X$')    
ax1.set_ylabel(r'$\phi(L_X)$')
#ax1.legend((Legend[1]), loc='best')
pl.show()

#------------------------------------------------------------

z = 5.0
L = np.logspace(41, 47, 100)
m = 1000

fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)

models = []
for t in range(m):
    model = a15.LuminosityFunction_LDDE1_softband(L, z, 10000, 1)
    models.append(model)

for i, j in enumerate(models):
    pl.loglog(L, models[i], color = 'g', alpha = 0.1)
    
#pl.title('2-7 KeV LDDE1 at z ~ %.1f' % (z))
pl.ylim(10**-9.1, 10**-2)

ax1.set_xlabel(r'$L_X$')    
ax1.set_ylabel(r'$\phi(L_X)$')
#ax1.legend((Legend[0]), loc='best')
pl.show()

#------------------------------------------------------------good

z = 5.0
L = np.logspace(41, 47, 1000)
m = 1000

fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)

models = []
for t in range(m):
    model = a15.LuminosityFunction_LDDE2_hardband(L, z, 10000, 1)
    models.append(model) 
    #print len(model)
#models = np.array(models)


for i, j in enumerate(models):
    
    pl.loglog(L, models[i], color = 'k', alpha = 0.1)
#pl.title('0.5-2 KeV LDDE1 at z ~ %.1f' % (z))
pl.ylim(10**-9.1, 10**-2)
ax1.set_xlabel(r'$L_X$')    
ax1.set_ylabel(r'$\phi(L_X)$')
#ax1.legend((Legend[3]), loc='best')
pl.show()

#------------------------------------------------------------good

z = 5.0
L = np.logspace(41, 47, 100)
m = 1000

fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)

models = []
for t in range(m):
    model = a15.LuminosityFunction_LDDE2_softband(L, z, 10000, 1)
    models.append(model) 

for i, j in enumerate(models):
    
    pl.loglog(L, models[i], color = 'b', alpha = 0.1)
#pl.title(r'Different models for $L_X$ for soft and hard bands at $z$ ~ $%.1f$' % (z))
pl.ylim(10**-9.1, 10**-2) 
ax1.set_xlabel(r'$L_X$')    
ax1.set_ylabel(r'$\phi(L_X)$')
ax1.legend((Legend), loc='best')
pl.show()

#------------------------------------------------------------"""