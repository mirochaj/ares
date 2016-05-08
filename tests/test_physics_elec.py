"""

test_secondary_electrons.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Apr  3 16:58:43 MDT 2014

Description: Reproduce Figures 1-3 (kind of) in Furlanetto & Stoever (2010).

"""

import ares
import numpy as np
import matplotlib.pyplot as pl
from ares.analysis import MultiPanel

# First, compare at fixed ionized fraction
xe = [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 0.9]
E = np.logspace(1, 4, 400)
channels = ['heat', 'h_1', 'exc', 'lya']
channels.reverse()

colors = ['k', 'b', 'r', 'g', 'm', 'c', 'y']

def test():

    esec1 = ares.physics.SecondaryElectrons(method=1)
    esec2 = ares.physics.SecondaryElectrons(method=2)
    esec3 = ares.physics.SecondaryElectrons(method=3)
    
    # Re-make Figure 1 from FJS10
    fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)
    
    for channel in ['heat', 'h_1', 'he_1', 'lya', 'exc']:
        func = lambda EE: esec3.DepositionFraction(E=EE, channel=channel, xHII=0.01)
        ax1.semilogx(E, map(func, E), label=channel)
    
    ax1.set_xlabel(r'$E \ (\mathrm{eV})$')
    ax1.set_ylabel('Fraction')
    ax1.legend(loc='upper right', frameon=False, fontsize=14)    
        
    pl.draw()
    
    # Now, Figures 2 and 3 
    mp = MultiPanel(fig=2, dims=(2, 2), padding=(0.2, 0))
    
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
        mp.grid[j].set_xlim(10, 1e4)
        
        if j == 2:
            mp.grid[1].legend(loc='upper right', ncol=2, fontsize=14)
    
    for i in range(2):
        mp.grid[i].set_xlabel(r'Electron Energy (eV)')
    
    mp.fix_ticks()
    
    pl.savefig('%s.png' % (__file__.rstrip('.py')))
    pl.close('all')
        
    assert True
    
if __name__ == '__main__':
    test()
    