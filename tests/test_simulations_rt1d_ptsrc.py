"""

test_simulations_rt1d.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Mon 20 Dec 2021 11:54:45 EST

Description:

"""

import ares
import numpy as np

def test():

    # Uniform density, isothermal, point source Q=5e48
    sim = ares.simulations.RaySegment(problem_type=1,
        stop_time=100, grid_cells=32)
    sim.run()

    # Make sure I-front is made over time
    assert np.mean(sim.history['h_2'][-1]) > sim.history['h_2'][0,0]

    # Same thing but now isothermal=False
    sim = ares.simulations.RaySegment(problem_type=2,
        stop_time=100, grid_cells=32)
    sim.run()

    # Make sure heating happens!
    assert np.mean(sim.history['Tk'][-1]) > sim.history['Tk'][0,0]

    # This run will have generated a lookup table for Gamma. Write to disk.
    sim.save_tables(prefix='test_rt1d')

    # Eventually, test read capability. Currently broken.

    # Same thing but now w/ secondary ionization/heating
    sim = ares.simulations.RaySegment(problem_type=2,
        stop_time=100, grid_cells=32, secondary_ionization=1)
    sim.run()

    # Make sure heating happens!
    assert np.mean(sim.history['Tk'][-1]) > sim.history['Tk'][0,0]

if __name__ == "__main__":
    test()
