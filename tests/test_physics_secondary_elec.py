"""

test_secondary_electrons.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Apr  3 16:58:43 MDT 2014

Description: Reproduce Figures 1-3 (kind of) in Furlanetto & Stoever (2010).

"""

import ares
import numpy as np

# First, compare at fixed ionized fraction
xe = [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 0.9]
E = np.logspace(1, 4, 400)
channels = ['heat', 'h_1', 'exc', 'lya']

def test():

    esec1 = ares.physics.SecondaryElectrons(method=1)
    esec2 = ares.physics.SecondaryElectrons(method=2)
    esec3 = ares.physics.SecondaryElectrons(method=3)

    for channel in ['heat', 'h_1', 'he_1', 'lya', 'exc']:
        func = lambda EE: esec3.DepositionFraction(E=EE, channel=channel, xHII=0.01)
        f = list(map(func, E))


    results = {channel: [] for channel in channels}

    elements = [3, 2, 0, 1]
    for j, channel in enumerate(channels):

        l = elements[j]

        for k, x in enumerate(xe):

            if channel == 'lya' and x >= 0.5:
                continue

            # Compare to high-energy limit from Ricotti et al.
            if channel not in ['exc', 'lya']:
                f2 = list(map(lambda EE: esec2.DepositionFraction(xHII=x, E=EE,
                    channel=channel), E))

            f3 = np.array(list(map(lambda EE: esec3.DepositionFraction(xHII=x, E=EE,
                channel=channel), E)))

            if channel == 'lya':
                last = np.array(results['exc'][k])
            else:
                pass
                
            # Just need this to do flya/fexc
            results[channel].append(np.array(f3))

    assert True

if __name__ == '__main__':
    test()
