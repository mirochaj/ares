"""

test_hod.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Jul 17 19:19:12 MDT 2015

Description: 

"""

import ares
import numpy as np
import matplotlib.pyplot as pl
from ares.physics.Constants import rhodot_cgs

pop_fcoll = ares.populations.GalaxyPopulation()

z = np.arange(6, 30, 0.2)
sfrd_fcoll = map(lambda zz: pop_fcoll.SFRD(zz) * rhodot_cgs, z)
pl.semilogy(z, sfrd_fcoll, label=r'$f_{\mathrm{coll}}$')

for Mmin in [1e5, 1e6, 1e7]:
    pop_hod = ares.populations.GalaxyPopulation(pop_halo_model='hod',
        pop_Mmin=Mmin)
        
    sfrd_hod = map(lambda zz: pop_hod.SFRD(zz) * rhodot_cgs, z) 
    pl.semilogy(z, sfrd_hod, 
        label=r'HOD; $M_{\min} = 10^{%i} \ M_{\odot}$' % np.log10(Mmin))

pl.xlabel(r'$z$')
pl.ylabel(ares.util.labels['sfrd'])
pl.legend(loc='lower left', fontsize=14)




