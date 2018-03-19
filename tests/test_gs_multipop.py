"""

test_gs_multipop.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Fri Jul  1 15:42:30 PDT 2016

Description: 

"""

import ares
import matplotlib.pyplot as pl

def test():

    fcoll = ares.util.ParameterBundle('pop:fcoll')
    popIII = ares.util.ParameterBundle('sed:uv')
    
    pop = fcoll + popIII
    
    # Restrict to halos below the atomic cooling threshold
    pop['pop_fstar'] = 1e-3
    pop['pop_Tmin'] = 300
    pop['pop_Tmax'] = 1e4
    
    # Tag with ID number
    pop.num = 3
    
    sim1 = ares.simulations.Global21cm()
    sim2 = ares.simulations.Global21cm(**pop)
    
    sim1.run()
    sim2.run()
    
    ax, zax = sim1.GlobalSignature(color='k')
    sim2.GlobalSignature(ax=ax, color='b')
    
    pl.draw()
    
    assert True
    
if __name__ == '__main__':
    test()
