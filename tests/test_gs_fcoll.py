"""

test_21cm_basic.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Oct 1 15:23:53 2012

Description: Make sure the global 21-cm signal calculator works.

"""

import ares
import matplotlib.pyplot as pl

def test():

    sim = ares.simulations.Global21cm(verbose=False, progress_bar=False)
    sim.run()
    sim.GlobalSignature()
    
    pl.savefig('%s.png' % (__file__.rstrip('.py')))
        
    assert True
    
if __name__ == '__main__':
    test()    

