"""

test_gs_lfcal.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Thu Feb 23 15:54:29 PST 2017

Description: 

"""

import ares
import matplotlib.pyplot as pl

def test():
    pars = ares.util.ParameterBundle('mirocha2016:dpl')
    
    # minimal build doesn't include 1000-element tau table, need to downgrade
    pars['initial_redshift'] = 50
    pars['pop_tau_Nz{0}'] = 400
    
    sim = ares.simulations.Global21cm(**pars)
    sim.run()
    sim.GlobalSignature()
    
    pl.savefig('{!s}.png'.format(__file__.rstrip('.py')))
    pl.close()

    assert True
    
if __name__ == '__main__':
    test()
