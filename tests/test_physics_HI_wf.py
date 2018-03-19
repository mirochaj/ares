"""

test_physics_lya_coupling.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Sep 25 10:01:27 MDT 2013

Description: 

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

def test():
    zarr = np.linspace(5, 40)
    
    hydr1 = ares.physics.Hydrogen(approx_Salpha=1)
    hydr2 = ares.physics.Hydrogen(approx_Salpha=2)
    hydr3 = ares.physics.Hydrogen(approx_Salpha=3)
    
    pl.plot(zarr, list(map(lambda z: 1. - hydr1.Sa(z, hydr1.cosm.Tgas(z)), zarr)), 
        color='k', ls='-')
    pl.plot(zarr, list(map(lambda z: 1. - hydr2.Sa(z, hydr2.cosm.Tgas(z)), zarr)), 
        color='k', ls='--', label='Chuzhoy, Alvarez, & Shapiro \'05')
    pl.plot(zarr, list(map(lambda z: 1. - hydr3.Sa(z, hydr3.cosm.Tgas(z)), zarr)), 
        color='k', ls=':', label='Furlanetto & Pritchard \'06')   
         
    pl.ylim(0, 1.1)
    pl.xlabel(r'$z$')
    pl.ylabel(r'$1-S_{\alpha}$')
    pl.legend(loc='upper right', frameon=False, fontsize=14)
    pl.annotate(r'assuming $T_k \propto (1+z)^2$', (0.05, 0.05), 
        xycoords='axes fraction')
        
    pl.savefig('{!s}.png'.format(__file__[0:__file__.rfind('.')]))
    pl.close()    

    assert True

if __name__ == '__main__':
    test()
    
