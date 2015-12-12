"""

test_21cm_basic.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Oct 1 15:23:53 2012

Description: Make sure the global 21-cm signal calculator works.

"""

import ares

def test():

    sim = ares.simulations.Global21cm()
    sim.run()
    sim.GlobalSignature()
        
    assert True
    
if __name__ == '__main__':
    test()    

