"""

test_physics_secondary_ionization_coeff.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Apr  3 16:58:43 MDT 2014

Description: Reproduce Figures 2-3 in Furlanetto & Stoever (2010).

"""

import rt1d
import matplotlib.pyplot as pl
import numpy as np
from multiplot import multipanel

# First, compare at fixed ionized fraction
xe = [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 0.9]
E = np.logspace(1, 4, 400)
channels = ['heat', 'h_1', 'lya']
channels.reverse()

colors = ['k', 'b', 'r', 'g', 'm', 'c', 'y']

mp = multipanel(dims=(2, 2), padding=(0.2, 0.))

esec1 = rt1d.physics.SecondaryElectrons(method=1)
esec2 = rt1d.physics.SecondaryElectrons(method=2)
esec3 = rt1d.physics.SecondaryElectrons(method=3)

for j, channel in enumerate(channels):
    
    for k, x in enumerate(xe):       
                
        if j == 1:
            if x < 0.5:
                label = r'$x_e = 10^{%i}$' % (np.log10(x))
            else:
                label = r'$x_e = %.2g$' % x
        else:
            label = None                   
        
        if channel not in ['exc', 'lya']:                  
            f2 = map(lambda EE: esec2.DepositionFraction(xHII=x, E=EE, 
                channel=channel), E)
            mp.grid[j].semilogx(E, f2, color=colors[k], ls='--')
            
        f3 = map(lambda EE: esec3.DepositionFraction(xHII=x, E=EE, 
            channel=channel), E)
        mp.grid[j].semilogx(E, f3, color=colors[k], ls='-', label=label)
        
        mp.grid[j].semilogx([10, 1e4], 
            [esec1.DepositionFraction(xHII=x, channel=channel)]*2, 
            color=colors[k], ls=':')
    
    mp.grid[j].set_ylabel(r'$f_{\mathrm{%s}}$' % channel)
    mp.grid[j].set_yscale('linear')
    mp.grid[j].set_ylim(0, 1.05)
    
    if j == 2:
        mp.grid[1].legend(loc='upper right', ncol=2)

for i in range(2):
    mp.grid[i].set_xlabel(r'Electron Energy (eV)')

#mp.fix_ticks()

#pl.draw()