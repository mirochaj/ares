"""

test_21cm_extrema.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Tue May  6 18:10:46 MDT 2014

Description: Make sure our extrema-finding routines work.

"""
    
import ares
import numpy as np
from ares.physics.Constants import nu_0_mhz

def test():

    sim = ares.simulations.Global21cm(gaussian_model=True, gaussian_nu=70.,
        gaussian_A=-100.)
    sim.run()
                            
    # In this case, we know exactly where C happens
    absorption_OK = np.allclose(nu_0_mhz / (1. + sim.turning_points['C'][0]), 
        sim.pf['gaussian_nu'])
    absorption_OK = np.allclose(sim.turning_points['C'][1], 
        sim.pf['gaussian_A'], rtol=1e-3, atol=1e-3)
        
    no_nonsense = 1
    
    # Check to make sure no turning points are absurd
    things = ['redshift', 'amplitude', 'curvature']
    for tp in list('BCD'):
        if tp not in sim.turning_points:
            continue
            
        for i, element in enumerate(sim.turning_points[tp]):
    
            if -500 <= element <= 100:
                continue
    
            print 'Absurd turning point! %s of %s' % (things[i], element)
            no_nonsense *= 0
    
    # Now, check the turning-point-finding on the tanh model
    # Test sensitivity to frequency sampling
    for dnu in [0.05, 0.1, 0.5, 1]:
        freq = np.arange(40, 120+dnu, dnu)
        sim = ares.simulations.Global21cm(tanh_model=True, 
            output_frequencies=freq)
                                
    # Everything good?
    assert absorption_OK and no_nonsense
    
if __name__ == "__main__":
    test()
