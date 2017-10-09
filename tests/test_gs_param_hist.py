"""

test_21cm_parameterized.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Aug  6 08:54:15 MDT 2014

Description: 21-cm signal in absence of astrophysical sources.

"""

import ares
import numpy as np
import matplotlib.pyplot as pl
from ares.analysis.MultiPlot import MultiPanel

def test():

    # Create instance of Hydrogen class
    hydr = ares.physics.Hydrogen()
    
    # Analytic approximation to thermal history
    Tk = lambda z: hydr.cosm.Tgas(z)
    
    # Spin temperature (arguments: z, Tk, Ja, xHII, ne)
    Ts = lambda z: hydr.SpinTemperature(z, Tk(z), 0.0, 0.0, 0.0)
    
    # Brightness temperature (arguments: z, xHII, Ts)
    dTb = lambda z: hydr.DifferentialBrightnessTemperature(z, 0.0, Ts(z))
    
    # Define redshift interval of interest
    z = np.linspace(10, 1e3, 500)
    
    # Start plotting stuff
    mp = MultiPanel(dims=(2, 1), panel_size=(1, 0.5))

    # Plot temperatures
    mp.grid[1].loglog(z, list(map(hydr.cosm.TCMB, z)), color='k', ls=':')
    mp.grid[1].loglog(z, list(map(Tk, z)), color='k', ls='--')
    mp.grid[1].loglog(z, list(map(Ts, z)), color='k', ls='-')
    mp.grid[1].set_ylim(1, 1e4)

    # Plot 21-cm signature
    mp.grid[0].semilogx(z, list(map(dTb, z)), color='k', label='analytic')
    
    CR = ares.util.ReadData._load_inits()
    
    # Assume neutral medium for simplicity
    Ts_CR = hydr.SpinTemperature(CR['z'], CR['Tk'], 0.0, 0.0, 0.0)
    dTb_CR = hydr.DifferentialBrightnessTemperature(CR['z'], 0.0, Ts_CR)
    
    mp.grid[0].semilogx(CR['z'], dTb_CR, color='b', label='CosmoRec')
    mp.grid[0].legend(loc='lower right')

    mp.grid[1].loglog(CR['z'], CR['Tk'], color='b', ls='--')
    
    # Labels
    mp.grid[0].set_xlabel(r'$z$')
    mp.grid[0].set_ylabel('Temperature')
    mp.grid[1].set_xticklabels([])
    mp.grid[1].set_ylabel(r'$\delta T_b \ (\mathrm{mK})$')

    # Limits
    mp.grid[0].set_xlim(z.min(), z.max())
    mp.grid[1].set_xlim(z.min(), z.max())

    pl.draw()
    
    pl.savefig('{!s}.png'.format(__file__.rstrip('.py')))
    pl.close()
    
    assert True
    
if __name__ == '__main__':
    test()    

