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

colors = ['k', 'b', 'r', 'g', 'm', 'c', 'y']

def test():

    esec1 = ares.physics.SecondaryElectrons(method=1)
    esec2 = ares.physics.SecondaryElectrons(method=2)
    esec3 = ares.physics.SecondaryElectrons(method=3)
    
    # Re-make Figure 1 from FJS10
    fig1 = pl.figure(1); ax1 = fig1.add_subplot(111)
    
    for channel in ['heat', 'h_1', 'he_1', 'lya', 'exc']:
        func = lambda EE: esec3.DepositionFraction(E=EE, channel=channel, xHII=0.01)
        ax1.semilogx(E, list(map(func, E)), label=channel)
    
    ax1.set_xlabel(r'$E \ (\mathrm{eV})$')
    ax1.set_ylabel('Fraction')
    ax1.legend(loc='upper right', frameon=False, fontsize=14)    
        
    pl.draw()
    
    # Now, Figures 2 and 3 
    mp = MultiPanel(fig=2, dims=(2, 2), padding=(0.3, 0.05))

    results = {channel: [] for channel in channels}
    
    elements = [3, 2, 0, 1]
    for j, channel in enumerate(channels):
        
        l = elements[j]
        
        for k, x in enumerate(xe):       
                    
            if j == 1:
                if x < 0.5:
                    label = r'$x_e = 10^{{{}}}$'.format(int(np.log10(x)))
                else:
                    label = r'$x_e = {:.2g}$'.format(x)
            else:
                label = None                   
            
            if channel == 'lya' and x >= 0.5:
                continue
            
            # Compare to high-energy limit from Ricotti et al.
            if channel not in ['exc', 'lya']:                  
                f2 = list(map(lambda EE: esec2.DepositionFraction(xHII=x, E=EE, 
                    channel=channel), E))
                mp.grid[l].semilogx(E, f2, color=colors[k], ls='--')
                
            f3 = np.array(list(map(lambda EE: esec3.DepositionFraction(xHII=x, E=EE, 
                channel=channel), E)))
                
            if channel == 'lya':
                last = np.array(results['exc'][k])
                mp.grid[l].semilogx(E, f3 / last, color=colors[k], ls='-', label=label)
            else:
                mp.grid[l].semilogx(E, f3, color=colors[k], ls='-', label=label)
                
            # Just need this to do flya/fexc
            results[channel].append(np.array(f3))
            
            # Compare to high-energy limit from Shull 
            if channel == 'lya':
                continue
                
            mp.grid[l].semilogx([10, 1e4], 
                [esec1.DepositionFraction(xHII=x, channel=channel)]*2, 
                color=colors[k], ls=':')
                
        mp.grid[l].set_ylabel(r'$f_{{\mathrm{{{!s}}}}}$'.format(channel))
        mp.grid[l].set_yscale('linear')
        mp.grid[l].set_ylim(0, 1.05)
        mp.grid[l].set_xlim(10, 1e4)
        
        if j == 2:
            mp.grid[2].legend(loc='upper right', ncol=2, fontsize=14)
    
    for i in range(2):
        mp.grid[i].set_xlabel(r'Electron Energy (eV)')
    
    mp.fix_ticks()
    
    for i in range(1,3):
        pl.figure(i)
        pl.savefig('{0!s}_{1}.png'.format(__file__.rstrip('.py'), i))
        
    pl.close('all')
        
    assert True
    
if __name__ == '__main__':
    test()
    
