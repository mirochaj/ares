"""

test_physics_collisional_coupling.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Nov  9 16:01:47 2012

Description: 

"""

import numpy as np   
import matplotlib.pyplot as pl
from ares.physics import Hydrogen

def test():

    hydr = Hydrogen()
    hydr2 = Hydrogen(interp_cc='linear')
    
    # Relevant temperature range
    T = np.logspace(0, 4, 500)
    
    fig1 = pl.figure(1)#; fig2 = pl.figure(2)
    ax1 = fig1.add_subplot(111)#; ax2 = fig2.add_subplot(111)
    
    # Plot coupling coefficients vs. temperature (tabulated)
    ax1.scatter(hydr.tabulated_coeff['T_H'], 
        hydr.tabulated_coeff['kappa_H'], color = 'k')
    ax1.scatter(hydr.tabulated_coeff['T_e'], 
        hydr.tabulated_coeff['kappa_e'], color = 'k')
        
    # Interpolated values
    ax1.loglog(T, hydr.kappa_H(T), color = 'k', ls = '-', 
        label=r'$\kappa_{10}^{\mathrm{HH}}$')
    ax1.loglog(T, hydr.kappa_e(T), color = 'k', ls = '--', 
        label=r'$\kappa_{10}^{\mathrm{eH}}$') 
    
    # Linear fallback option
    ax1.loglog(T, hydr2.kappa_H(T), color = 'b', ls = '-')
    ax1.loglog(T, hydr2.kappa_e(T), color = 'b', ls = '--') 
        
    # Tidy up
    ax1.set_xlim(1, 1e4)
    ax1.set_ylim(1e-13, 2e-8)
    ax1.set_xlabel(r'$T \ (\mathrm{K})$')
    ax1.set_ylabel(r'$\kappa \ (\mathrm{cm}^3 \mathrm{s}^{-1})$')
    ax1.legend(loc = 'lower right', frameon = False)
    ax1.annotate('Points from Zygelman (2005)', (1.5, 1e-8), ha='left', va='top',
        fontsize=14)
    ax1.annotate('Lines are spline fit to points', (1.5, 5e-9), ha='left', va='top',
        fontsize=14)
    
    pl.draw()
    
    ok = 1
    for suffix in ['H', 'e']:
    
        tab = hydr.tabulated_coeff['kappa_%s' % suffix]
        
        if suffix == 'H':
            interp = hydr.kappa_H(hydr.tabulated_coeff['T_%s' % suffix])
        else:
            interp = hydr.kappa_e(hydr.tabulated_coeff['T_%s' % suffix])
                    
        # Numbers small so ignore absolute tolerance
        ok *= np.allclose(tab, interp, atol=0.0)
        
    pl.savefig('%s.png' % (__file__.rstrip('.py')))
    pl.close()        
        
    assert ok, "Error in computation of coupling coefficients."
    
if __name__ == '__main__':
    test()
    
    
