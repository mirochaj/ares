# -*- coding: utf-8 -*-
"""
test_Lx(z)

Author: Jacob Jost
Affiliation: University of Colorado at Boulder (Undergraduate)
Created on: Thu June 12 11:00:00 MDT 2015


The redshift can be set so the same redshift is used for each plot or can be set 
for each individual plot. You either need to keep the top z for the overall or 
hash it out and unhash the other z's to set individually.

Models = # of models you want to run

NOTE: 
    
    This has not been vectorized so any more than 50 models will take quite some
    time to run. 

    If you want to look at a particular model, just use triple quotes to take the 
    section you dont need out. 
    
    To converte from the 2-10 KeV band to the 0.5-8 Kev Band divide integrand1 by 1.33.
    
    --- = spacing between different sections of code
"""

import ares
import numpy as np
import matplotlib.pyplot as pl
from scipy import integrate

a15 = ares.util.read_lit('aird2015_Lx')

Legend = ['Green = LDDE1 softband', 'Red = LDDE1 hardband', \
'Blue =  LDDE2 softband', 'Black = LDDE2 hardband']

z = np.linspace(0, 5, 100)

#------------------------------------------------------------

#z = np.linspace(0, 5, 100)
models = 50
Legend1 = ['Red = LDDE1 hardband']
                
integrand1=[]
for i in range(models):
    
    integrand = [] 
    for j in range(len(z)):
       
        x = lambda Lx: a15._LuminosityFunction_LDDE1_integrate(Lx, z[j],\
        a15.qsolf_LDDE1_hardpars_integration.values()[6], \
        a15.qsolf_LDDE1_hardpars_integration.values()[7][i], \
        a15.qsolf_LDDE1_hardpars_integration.values()[3][i],\
        a15.qsolf_LDDE1_hardpars_integration.values()[2][i], \
        a15.qsolf_LDDE1_hardpars_integration.values()[1][i], \
        a15.qsolf_LDDE1_hardpars_integration.values()[0][i], \
        a15.qsolf_LDDE1_hardpars_integration.values()[8][i], \
        a15.qsolf_LDDE1_hardpars_integration.values()[4][i], \
        a15.qsolf_LDDE1_hardpars_integration.values()[5][i])
        p, err = integrate.quad(x, 10**41, 10**46)
        integrand.append(p)
    integrand1.append(integrand)

#HEADS UP: this takes a while to run, use caution. 

for i in range(len(integrand1)):
    pl.semilogy(z, integrand1[i], alpha = 0.25, color = 'r')
    
#fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)
#ax1.set_ylabel(r'$L_X(z)$')    
#ax1.set_xlabel(r'$z$')
#pl.legend((Legend1), loc='best')
#pl.show()

#------------------------------------------------------------"""

#z = np.linspace(0, 5, 100)
models = 50
Legend2 = ['Green = LDDE1 Softband']
                
integrand1=[]
for i in range(models):
    
    integrand = [] 
    for j in range(len(z)):
        x = lambda Lx: a15._LuminosityFunction_LDDE1_integrate(Lx, z[j],\
        a15.qsolf_LDDE1_softpars_integration.values()[6], \
        a15.qsolf_LDDE1_softpars_integration.values()[7][i], \
        a15.qsolf_LDDE1_softpars_integration.values()[3][i],\
        a15.qsolf_LDDE1_softpars_integration.values()[2][i], \
        a15.qsolf_LDDE1_softpars_integration.values()[1][i], \
        a15.qsolf_LDDE1_softpars_integration.values()[0][i], \
        a15.qsolf_LDDE1_softpars_integration.values()[8][i], \
        a15.qsolf_LDDE1_softpars_integration.values()[4][i], \
        a15.qsolf_LDDE1_softpars_integration.values()[5][i])
        p, err = integrate.quad(x, 10**41, 10**46)
        integrand.append(p)
    integrand1.append(integrand)

#HEADS UP: this takes a while to run, use caution. 

for i in range(len(integrand1)):
    pl.semilogy(z, integrand1[i], alpha = 0.25, color = 'g')
    
#fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)
#ax1.set_ylabel(r'$L_X(z)$')    
#ax1.set_xlabel(r'$z$')
#pl.legend((Legend2, Legend1), loc='best')
#pl.show()

#------------------------------------------------------------

#z = np.linspace(0, 5, 100)
models = 50
Legend3 = ['Black = LDDE2 hardband']
               
integrand1=[]
for i in range(models):
    
    integrand = [] 
    for j in range(len(z)):
        x = lambda Lx: a15._LuminosityFunction_LDDE2_integrate(Lx, z[j],\
        a15.qsolf_LDDE2_hardpars_integration.values()[3], \
        a15.qsolf_LDDE2_hardpars_integration.values()[-2][i], \
        a15.qsolf_LDDE2_hardpars_integration.values()[2][i],\
        a15.qsolf_LDDE2_hardpars_integration.values()[1][i], \
        a15.qsolf_LDDE2_hardpars_integration.values()[9][i], \
        a15.qsolf_LDDE2_hardpars_integration.values()[6][i], \
        a15.qsolf_LDDE2_hardpars_integration.values()[7][i], \
        a15.qsolf_LDDE2_hardpars_integration.values()[-1][i], \
        a15.qsolf_LDDE2_hardpars_integration.values()[-3][i],\
        a15.qsolf_LDDE2_hardpars_integration.values()[-6][i],\
        a15.qsolf_LDDE2_hardpars_integration.values()[5][i],\
        a15.qsolf_LDDE2_hardpars_integration.values()[-4][i],\
        a15.qsolf_LDDE2_hardpars_integration.values()[4][i],\
        a15.qsolf_LDDE2_hardpars_integration.values()[0][i])
        p, err = integrate.quad(x, 10**41, 10**46)
        integrand.append(p)
    integrand1.append(integrand)

#HEADS UP: this takes a while to run, use caution. 

for i in range(len(integrand1)):
    pl.semilogy(z, integrand1[i], alpha = 0.25, color = 'k')
    
#fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)
#ax1.set_ylabel(r'$L_X(z)$')    
#ax1.set_xlabel(r'$z$')
#pl.legend((Legend3), loc='best')
#pl.show()

#------------------------------------------------------------

#z = np.linspace(0, 5, 100)
models = 50
Legend4 = ['Blue = LDDE2 Softband']
               
integrand1=[]
for i in range(models):
    
    integrand = [] 
    for j in range(len(z)):
        x = lambda Lx: a15._LuminosityFunction_LDDE2_integrate(Lx, z[j],\
        a15.qsolf_LDDE2_softpars_integration.values()[3], \
        a15.qsolf_LDDE2_softpars_integration.values()[-2][i], \
        a15.qsolf_LDDE2_softpars_integration.values()[2][i],\
        a15.qsolf_LDDE2_softpars_integration.values()[1][i], \
        a15.qsolf_LDDE2_softpars_integration.values()[9][i], \
        a15.qsolf_LDDE2_softpars_integration.values()[6][i], \
        a15.qsolf_LDDE2_softpars_integration.values()[7][i], \
        a15.qsolf_LDDE2_softpars_integration.values()[-1][i], \
        a15.qsolf_LDDE2_softpars_integration.values()[-3][i],\
        a15.qsolf_LDDE2_softpars_integration.values()[-6][i],\
        a15.qsolf_LDDE2_softpars_integration.values()[5][i],\
        a15.qsolf_LDDE2_softpars_integration.values()[-4][i],\
        a15.qsolf_LDDE2_softpars_integration.values()[4][i],\
        a15.qsolf_LDDE2_softpars_integration.values()[0][i])
        p, err = integrate.quad(x, 10**41, 10**46)
        integrand.append(p)
    integrand1.append(integrand)

#HEADS UP: this takes a while to run, use caution. 

for i in range(len(integrand1)):
    pl.semilogy(z, integrand1[i], alpha = 0.25, color = 'b')
    
#fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)
#ax1.set_ylabel(r'$L_X(z)$')    
#ax1.set_xlabel(r'$z$')
#pl.legend((Legend), loc='best')
#pl.show()

#---------------------------------------------------------

#z = np.linspace(0, 5, 50)
models = 50
Legend5 = ['Black = 0.5-8 KeV Band (LDDE2)']
               
integrand1=[]
for i in range(models):
    
    integrand = [] 
    for j in range(len(z)):
        x = lambda Lx: a15._LuminosityFunction_LDDE2_integrate(Lx, z[j],\
        a15.qsolf_LDDE2_hardpars_integration.values()[3], \
        a15.qsolf_LDDE2_hardpars_integration.values()[-2][i], \
        a15.qsolf_LDDE2_hardpars_integration.values()[2][i],\
        a15.qsolf_LDDE2_hardpars_integration.values()[1][i], \
        a15.qsolf_LDDE2_hardpars_integration.values()[9][i], \
        a15.qsolf_LDDE2_hardpars_integration.values()[6][i], \
        a15.qsolf_LDDE2_hardpars_integration.values()[7][i], \
        a15.qsolf_LDDE2_hardpars_integration.values()[-1][i], \
        a15.qsolf_LDDE2_hardpars_integration.values()[-3][i],\
        a15.qsolf_LDDE2_hardpars_integration.values()[-6][i],\
        a15.qsolf_LDDE2_hardpars_integration.values()[5][i],\
        a15.qsolf_LDDE2_hardpars_integration.values()[-4][i],\
        a15.qsolf_LDDE2_hardpars_integration.values()[4][i],\
        a15.qsolf_LDDE2_hardpars_integration.values()[0][i])
        p, err = integrate.quad(x, 10**41, 10**46)
        integrand.append(p)
    integrand1.append(integrand)
integrand1 = np.array(integrand1)

#HEADS UP: this takes a while to run, use caution. 

for i in range(len(integrand1)):
    pl.semilogy(z, integrand1[i], alpha = 0.25, color = 'k')
    
fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)
ax1.set_ylabel(r'$L_X(z)$')    
ax1.set_xlabel(r'$z$')
pl.legend((Legend), loc='best')
pl.show()

#---------------------------------------------------------