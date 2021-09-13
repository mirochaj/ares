"""

test_physics_rates.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sun Apr 13 16:38:44 MDT 2014

Description:

"""

import ares, sys
import numpy as np

def test():
    species = 0
    dims = 32
    T = np.logspace(3.5, 6, 500)

    colors = list('kb')
    for i, src in enumerate(['fk94']):

        # Initialize grid object
        grid = ares.static.Grid(grid_cells=dims)

        # Set initial conditions
        grid.set_physics(isothermal=True)
        grid.set_chemistry()
        grid.set_density(1)
        grid.set_ionization(x=[1. - 1e-8, 1e-8])
        grid.set_temperature(T)

        coeff = coeffB = ares.physics.RateCoefficients(grid=grid, rate_src=src, T=T)
        coeffA = ares.physics.RateCoefficients(grid=grid, rate_src=src, T=T,
            recombination='A')

        # First: collisional ionization, recombination
        CI = [coeff.CollisionalIonizationRate(species, TT) for TT in T]
        RRB = [coeff.RadiativeRecombinationRate(species, TT) for TT in T]
        RRA = [coeffA.RadiativeRecombinationRate(species, TT) for TT in T]

        # Second: Cooling processes
        CIC = [coeff.CollisionalIonizationCoolingRate(species, TT) for TT in T]
        CEC = [coeff.CollisionalExcitationCoolingRate(species, TT) for TT in T]
        RRC = [coeff.RecombinationCoolingRate(species, TT) for TT in T]

    assert True

if __name__ == '__main__':
    test()
