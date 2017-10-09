"""

test_gs_basic.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Thu Jan 19 10:35:33 PST 2017

Description: 

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

def test():
    sim = ares.simulations.Global21cm()
    sim.run()
    sim.GlobalSignature()
    
    pl.savefig('{!s}.png'.format(__file__.rstrip('.py')))
    pl.close()
    
    # Make sure it's not a null signal.
    z = sim.history['z']
    dTb = sim.history['dTb'][z < 50]
    assert len(np.unique(np.sign(dTb))) == 2
    assert max(dTb) > 5 and min(dTb) < -5
        
if __name__ == '__main__':
    test()


