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

    updates = {'stop_time': 100, 'grid_cells': 32}

    # Uniform density, isothermal, point source Q=5e48
    pars = ares.util.ParameterBundle('rt1d:isothermal')
    pars.update(updates)
    sim = ares.simulations.RaySegment(**pars)
    sim.run()

    # Make sure I-front is made over time
    assert np.mean(sim.history['h_2'][-1]) > sim.history['h_2'][0,0]

    # Same thing but now isothermal=False
    pars = ares.util.ParameterBundle('rt1d:heating')
    pars.update(updates)
    sim = ares.simulations.RaySegment(**pars)
    sim.run()

    # Make sure heating happens!
    assert np.mean(sim.history['Tk'][-1]) > sim.history['Tk'][0,0]

    # This run will have generated a lookup table for Gamma. Write to disk.
    sim.save_tables(prefix='test_rt1d')

    # Eventually, test read capability. Currently broken.

    # Same thing but now w/ secondary ionization/heating
    pars['secondary_ionization'] = 1
    sim = ares.simulations.RaySegment(**pars)
    sim.run()

    # Make sure heating happens!
    assert np.mean(sim.history['Tk'][-1]) > sim.history['Tk'][0,0]

if __name__ == "__main__":
    test()
