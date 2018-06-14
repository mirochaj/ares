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
    Tarr = np.logspace(-1, 2)
    
    res = []
    ls = ['-', '--', '--', ':', '-.']
    labels = 'Chuzhoy+ \'05', 'Furlanetto \& Pritchard \'06', None, 'Hirata \'06'
    for i, method in enumerate([2,3,3.5,4]):
        hydr = ares.physics.Hydrogen(approx_Salpha=method)
    
        Sa = np.array([hydr.Sa(20., Tarr[k]) for k in range(Tarr.size)])
        pl.plot(Tarr, 1 - Sa, color='k', ls=ls[i], label=labels[i])
        res.append(Sa)
     
    pl.xlim(0.5, 100)
    pl.ylim(1e-1, 1.1)
    pl.xscale('log')
    pl.yscale('log')
    pl.xlabel(r'$T_K / \mathrm{K}$')
    pl.ylabel(r'$1-S_{\alpha}$')
    pl.legend(loc='lower left', frameon=False, fontsize=14)
     
    pl.savefig('{!s}.png'.format(__file__[0:__file__.rfind('.')]))
    pl.close()    
    
    # Compare at T > 1 K
    ok = Tarr > 1.
    diff = np.abs(np.diff(res, axis=1))    
    
    # Just set to a level that I know is tight enough to pickup
    # any errors we might accidentally introduce later.
    assert np.all(diff.ravel() < 0.3)

if __name__ == '__main__':
    test()
   
