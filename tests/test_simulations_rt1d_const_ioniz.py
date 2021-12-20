"""

test_const_ionization.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Feb 18 10:30:47 MST 2015

Description:

"""

import ares
import numpy as np

pf = \
{
 'grid_cells': 64,
 'isothermal': True,
 'stop_time': 1e2,
 'secondary_ionization': False,
 'density_units': 1.0,
 'initial_timestep': 1,
 'max_timestep': 1e2,
 'initial_temperature': np.logspace(3, 5, 64),
 'initial_ionization': [1. - 1e-4, 1e-4],         # neutral
}

def test():

    sim = ares.simulations.GasParcel(radiative_transfer=True, **pf)

    # Set constant ionization rate (coefficient) by hand
    k_ion = np.ones([sim.grid.dims,1]) * 1e-14

    # Initialize all rate coefficients
    sim.update_rate_coefficients(sim.grid.data)
    sim.set_radiation_field()

    # Evolve in time
    all_data = []
    for t, dt, data in sim.step():

        sim.update_rate_coefficients(sim.grid.data, k_ion=k_ion)

        all_data.append(data)

    # Re-run without radiative transfer for comparison
    sim2 = ares.simulations.GasParcel(radiative_transfer=False, **pf)
    sim2.run()


if __name__ == "__main__":
    test()
