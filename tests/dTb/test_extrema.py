"""

test_21cm_extrema.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Tue May  6 18:10:46 MDT 2014

Description: Make sure our extrema-finding routines work.

"""
    
import ares
import numpy as np
import matplotlib.pyplot as pl
from ares.physics.Constants import nu_0_mhz

def test():

    sim = ares.simulations.Global21cm(gaussian_model=True, gaussian_nu=70.,
        gaussian_A=-100.)
    sim.run()
                        
    anl = ares.analysis.Global21cm(sim)
    
    # Check that turning-point-finder works on Gaussian
    absorption_OK = np.allclose(nu_0_mhz / (1. + anl.turning_points['A'][0]), 
        sim.pf['gaussian_nu'])
    absorption_OK = np.allclose(anl.turning_points['A'][1], 
        sim.pf['gaussian_A'], rtol=1e-3, atol=1e-3)
        
    no_nonsense = 1
    
    # Check to make sure no turning points are absurd
    things = ['redshift', 'amplitude', 'curvature']
    for tp in list('BCD'):
        if tp not in anl.turning_points:
            continue
            
        for i, element in enumerate(anl.turning_points[tp]):
    
            if -500 <= element <= 100:
                continue
    
            print 'Absurd turning point! %s of %s' % (things[i], element)
            no_nonsense *= 0
    
    # Everything good?
    assert absorption_OK and no_nonsense



