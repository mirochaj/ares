"""

test_physics_cross_sections.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Apr 22 10:54:18 2013

Description:

"""

import numpy as np
from ares.physics.CrossSections import PhotoIonizationCrossSection, \
    ApproximatePhotoIonizationCrossSection

def test():

    E = np.logspace(np.log10(13.6), 4)

    sigma = PhotoIonizationCrossSection
    sigma_approx = ApproximatePhotoIonizationCrossSection

    for species in [0, 1]:
        _sigma = np.array([sigma(EE, species) for EE in E])
        _sigma_approx = np.array([sigma_approx(EE, species) for EE in E])

        assert np.allclose(_sigma, _sigma_approx, rtol=3e-1)

if __name__ == '__main__':
    test()
