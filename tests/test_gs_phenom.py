"""

test_21cm_tanh.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Tue Sep  9 20:03:58 MDT 2014

Description: 

"""

import ares
import matplotlib.pyplot as pl

def test():

    sim = ares.simulations.Global21cm(tanh_model=True)
    sim.run()
    ax, zax = sim.GlobalSignature(label='tanh')
        
    sim2 = ares.simulations.Global21cm(gaussian_model=True)
    sim2.run()
    sim2.GlobalSignature(ax=ax, label='gaussian')
    
    ax.legend(loc='lower right', fontsize=14)
    pl.savefig('{!s}.png'.format(__file__[0:__file__.rfind('.')]))
    pl.close()        
        
    assert True
    
if __name__ == "__main__":
    test()
