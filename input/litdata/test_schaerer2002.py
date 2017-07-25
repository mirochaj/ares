"""

test_schaerer.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Wed May 31 20:56:50 PDT 2017

Description: 

"""

import ares
import numpy as np
import schaerer2003 as s03
import matplotlib.pyplot as pl
from scipy.integrate import quad
from ares.physics.Constants import cm_per_rsun, g_per_msun, sigma_SB, \
    erg_per_ev, s_per_myr, Lsun


corr = []
colors = ['k', 'b', 'g', 'r', 'c', 'y', 'm', 'gray']*2
for j, mass in enumerate(s03.masses):
    x, y, T = s03._load(source_mass=mass)
    
    src = ares.sources.Star(source_temperature=T, source_Emin=10.2, 
        source_Emax=5e2)

    E = np.linspace(1, 1e2, 100)
    F = np.array(map(src.Spectrum, E)) / E
    
    Emin = []
    Earr = []
    hist = []
    for i, pt in enumerate(x):
        Eavg = src.AveragePhotonEnergy(*pt)
        Earr.append(Eavg)
        Emin.append(pt[0])
        band = np.array([[Eavg-pt[0], pt[1]-Eavg]]).T
        
        h = y[i] / Eavg#/ (pt[1] - pt[0])
        hist.append(h)
        
        # Normalizing by pt[1] - pt[0] leads to errors in LW
        #pl.errorbar(Eavg, h, xerr=band,
        #    fmt='o', color=colors[j], mec='none')
            
    EE = np.argsort(Emin)
    pl.plot(np.array(Earr)[EE], np.array(hist)[EE], 
        drawstyle='steps-mid', color=colors[j])

    # Normalize to Q(H)
    #norm = Earr[1] * (y[1] / (x[1][1] - x[1][0])) / src.Spectrum(Earr[1])
    norm = Earr[1] * (y[1] / Earr[1]) / src.Spectrum(Earr[1])

    pl.plot(E, F * norm, color=colors[j], ls='--')
    
    correction = hist[-1] / (np.interp(Earr[-1], E, F * norm))
    corr.append(correction)
    

pl.xscale('linear')
pl.yscale('log')
pl.ylim(1e43, 1e51)

# Add points for the Bromm models

#for j, mass in enumerate(s03.masses):
#    T = 1.1e5 * (mass / 100.)**0.025
#    tau = 3 * s_per_myr
#    R = (mass / 370.)**(1. / 2.2) * 10. * cm_per_rsun
#    
#    Lbol = 4. * np.pi * R**2 * sigma_SB * T**4
#    
#    src = ares.sources.Star(source_temperature=T, 
#        source_Emin=10.2, source_Emax=5e2)
#    
#    for band in [(11.2, 13.6), (13.6, 24.6), (24.6, 54.4), (54.4, 1e2)]:
#    
#        fband = quad(src.Spectrum, *band)[0] 
#    
#        Eavg = src.AveragePhotonEnergy(*band)
#        Q = (Lbol * fband / Eavg / erg_per_ev)
#    
#        pl.scatter(Eavg, Q, color=colors[j])
#    
    
    



