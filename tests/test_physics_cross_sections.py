"""

test_physics_cross_sections.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Apr 22 10:54:18 2013

Description: 

"""

import numpy as np
import matplotlib.pyplot as pl
from ares.physics.CrossSections import *

def test():

    E = np.logspace(np.log10(13.6), 4)
    
    sigma = PhotoIonizationCrossSection
    sigma_approx = ApproximatePhotoIonizationCrossSection
    
    fig1, ax1 = pl.subplots(1, 1)
    
    ax1.loglog(E, [sigma(EE, 0) for EE in E], color='k', ls='-', label=r'H')
    ax1.loglog(E, [sigma(EE, 1) for EE in E], color='k', ls='--', label=r'HeI')
    ax1.loglog(E, [sigma_approx(EE, 0) for EE in E], color='b', ls='-')
    ax1.loglog(E, [sigma_approx(EE, 1) for EE in E], color='b', ls='--')
    
    ax1.legend(frameon=False)
    
    ax1.set_xlim(10, 1e4)
    ax1.set_ylim(1e-25, 1e-16)
    
    ax1.set_xlabel(r'$h\nu \ (\mathrm{eV})$')
    ax1.set_ylabel(r'$\sigma_{\nu} \ (\mathrm{cm}^2)$')
    
    ax1.annotate(r'Verner & Ferland (1996)', (20, 1e-24), ha='left')
    ax1.annotate(r'Approximate', (20, 1e-25), color='b', ha='left')
    
    pl.savefig('{!s}.png'.format(__file__[0:__file__.rfind('.')]))
    pl.close()
    
    assert True

if __name__ == '__main__':
    test()
