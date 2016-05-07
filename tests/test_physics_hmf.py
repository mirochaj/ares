"""

test_hmf_PS.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri May  3 16:03:57 2013

Description: Use Press-Schechter mass function to test numerical integration, 
since PS has an analytic solution for the collapsed fraction.

"""

import ares
import numpy as np

def test(rtol=1e-2):

    # Two HMFs: one analytic, one numerical
    hmf_a = ares.populations.HaloPopulation(hmf_func='PS', hmf_analytic=True)
    hmf_n = ares.populations.HaloPopulation(hmf_func='PS', hmf_analytic=False,
        hmf_load=True)
    
    ok = True
    for i, z in enumerate([5, 10, 15, 20]):
        
        fcoll_a = hmf_a.halos.fcoll_tab[np.argmin(np.abs(z-hmf_a.halos.z))]
        
        try:
            fcoll_n = hmf_n.halos.fcoll_tab[np.argmin(np.abs(z-hmf_a.halos.z))]
        except AttributeError:
            fcoll_n = hmf_n.halos.fcoll(z, hmf_a.halos.logM)
        
        mask = np.logical_and(hmf_a.halos.M >= 1e8, hmf_a.halos.M <= 1e11)
        
        ok_z = np.allclose(fcoll_n[mask], fcoll_a[mask], rtol=rtol, atol=0)
    
        if not ok_z:
            ok = False
            break
                
    assert ok, "Relative error between analytical and numerical solutions exceeds %.3g." % rtol        
    
if __name__ == '__main__':
    test()
    