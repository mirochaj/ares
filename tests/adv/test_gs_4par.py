"""

test_gs_4par.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Thu May 26 15:53:18 PDT 2016

Description: 

"""

import ares
import numpy as np

def test():
    
    # Use N* parameters
    sim_N = ares.simulations.Global21cm(fstar=0.1, fesc=0.1, 
        Nlw=9690, Nion=4e3, fX=1, problem_type=101)
    sim_N.run()
    
    # Use xi* parameters
    sim_x = ares.simulations.Global21cm(xi_UV=40., 
        xi_LW=969, xi_XR=1e-1, problem_type=101)
    sim_x.run()
    
    for field in sim_N.history:
        assert np.allclose(sim_N.history[field], sim_x.history[field]), \
            "Backward compatibility issue w/ field '%s'." % field

if __name__ == '__main__':
    test()