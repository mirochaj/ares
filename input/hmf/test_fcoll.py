"""

test_hmf.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Thu Jul  7 15:29:10 PDT 2016

Description: 

"""

import ares
import numpy as np
import matplotlib.pyplot as pl
from scipy.integrate import simps

pop = ares.populations.HaloPopulation(pop_sfr_model='fcoll', pop_Mmin=1e8,
    hmf_interp='linear')

zarr = np.arange(10, 50, 0.1)

pl.semilogy(zarr, pop.fcoll(zarr), color='k', lw=3, alpha=0.5)

new_fcoll = []

j = np.argmin(np.abs(pop.halos.M - 1e8))

for z in zarr:
    i = np.argmin(np.abs(pop.halos.z - z))
    fcoll_mgtm1 = pop.halos.mgtm[i,j] / pop.halos.MF.mean_density0
    
    dndm = pop.halos.dndm[i,j]
    M = pop.halos.M
    
    ok = M >= 1e8
    
    dndlnm = dndm * M
    
    #fcoll_mgtm2 = np.trapz(dndlnm, x=np.log(M)) / pop.halos.MF.mean_density0
    fcoll_mgtm2 = simps(dndlnm[ok], x=np.log(M[ok])) / pop.halos.MF.mean_density0
        
    print z, fcoll_mgtm1, fcoll_mgtm2#, fcoll
    
    new_fcoll.append(fcoll_mgtm2)
    
pl.semilogy(zarr, new_fcoll, color='b', lw=1)    
    


