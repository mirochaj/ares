"""

test_gs_backcompat.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Fri Aug 12 12:57:41 PDT 2016

Description: Just checking to make sure the four parameter model works by
varying fX and making sure the realizations are different.

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

def test():
    data = []
    ax = None
    for fX in [0.2, 1.]:
        sim = ares.simulations.Global21cm(problem_type=101, fX=fX)
        sim.run()
        
        data.append((sim.history['z'], sim.history['dTb']))
        
        # Plot the global signal
        ax = sim.GlobalSignature(ax=ax,
            label=r'$f_X=%.2g$' % (fX))
        
    ax.legend(loc='lower right', fontsize=14)
    pl.draw()
    pl.savefig('%s.png' % (__file__.rstrip('.py')))
    pl.close()
    
    # Most common problem: breaking backward compatibility means the 
    # parameters aren't parsed, and have no affect on the signal. In this
    # case, all models will be identical, so we need to check to make sure
    # the parameters are having an effect.
    
    for i in range(len(data) - 1):
        
        z1, T1 = data[i]
        z2, T2 = data[i+1]
        
        assert np.any(T1 != T2), "Simulations yielded identical results!"
    
if __name__ == '__main__':
    test()