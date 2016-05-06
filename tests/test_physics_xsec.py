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

E = np.logspace(np.log10(13.6), 4)

sigma = PhotoIonizationCrossSection
sigma_approx = ApproximatePhotoIonizationCrossSection

pl.loglog(E, map(lambda EE: sigma(EE, 0), E),
    color='k', ls='-', label=r'H')
pl.loglog(E, map(lambda EE: sigma(EE, 1), E),
    color='k', ls='--', label=r'HeI')
pl.loglog(E, map(lambda EE: sigma_approx(EE, 0), E),
    color='b', ls='-')
pl.loglog(E, map(lambda EE: sigma_approx(EE, 1), E),
    color='b', ls='--')

pl.legend(frameon=False)

pl.xlabel(r'$h\nu \ (\mathrm{eV})$')
pl.ylabel(r'$\sigma_{\nu} \ (\mathrm{cm}^2)$')

pl.annotate(r'Verner & Ferland (1996)', (20, 1e-24), ha='left')
pl.annotate(r'Approximate', (20, 1e-25), color='b', ha='left')
