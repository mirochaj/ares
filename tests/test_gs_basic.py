"""

test_gs_basic.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Thu Jan 19 10:35:33 PST 2017

Description: 

"""

import ares

def test():
    sim = ares.simulations.Global21cm()
    sim.run()
    
    assert True
    
if __name__ == '__main__':
    test()


