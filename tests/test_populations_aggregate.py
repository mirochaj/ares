"""

test_populations_bh.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Fri  3 Apr 2020 12:54:41 EDT

Description:

"""

import ares
import numpy as np
from ares.physics.Constants import rhodot_cgs

def test():
    pop = ares.populations.GalaxyPopulation()

    zarr = np.arange(5, 40)

    sfrd = pop.get_sfrd(zarr) * rhodot_cgs

    pop2 = ares.populations.GalaxyPopulation(pop_sfr_model='sfrd-tab',
        pop_sfrd=(zarr, sfrd), pop_sfrd_units='internal')

    assert abs(sfrd[5] - pop2.get_sfrd(zarr[5])) < 1e-8, \
        "{:.3e} {:.3e}".format(sfrd[5], pop2.SFRD(zarr[5]))

    pop3 = ares.populations.GalaxyPopulation(pop_sfr_model='sfrd-func',
        pop_sfrd=lambda z: np.interp(z, zarr, sfrd), pop_sfrd_units='internal')

    assert abs(sfrd[5] - pop3.get_sfrd(zarr[5])) < 1e-8, \
        "{:.3e} {:.3e}".format(sfrd[5], pop3.SFRD(zarr[5]))


if __name__ == '__main__':
    test()
