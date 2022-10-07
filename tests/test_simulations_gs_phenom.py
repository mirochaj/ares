"""

test_21cm_tanh.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Tue Sep  9 20:03:58 MDT 2014

Description:

"""

import ares
import numpy as np

def test():

    sim = ares.simulations.Simulation(tanh_model=True)
    sim.get_21cm_gs()

    sim2 = ares.simulations.Simulation(gaussian_model=True)
    sim2.get_21cm_gs()

    p = \
     {
      'parametric_model': True,
      'pop_Ja': lambda z: 1e-2 * ((1. + z) / 10.)**-4.,
      'pop_Tk': lambda z: 1e2 * (1. - np.exp(-(15. / z)**4)),
      'pop_xi': lambda z: 1. - np.exp(-(10. / z)**4),
     }

    sim3 = ares.simulations.Simulation(**p)
    sim3.get_21cm_gs()

if __name__ == "__main__":
    test()
