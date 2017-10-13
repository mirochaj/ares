"""

test_physics_cosm.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Sep 24 15:39:44 MDT 2014

Description: 

"""

import numpy as np
from ares.physics import Cosmology
from ares.physics.Constants import s_per_gyr, m_H, m_He

def test(rtol=1e-3):
    
    cosm = Cosmology()
    
    # Check critical density
    assert cosm.CriticalDensity(0.) == cosm.CriticalDensityNow
    
    # Make sure energy densities sum to unity
    assert np.allclose(cosm.omega_m_0, 1. - cosm.omega_l_0)
    
    # Make sure the age of the Universe is OK
    assert 13.5 <= cosm.t_of_z(0.) / s_per_gyr <= 14.
    
    # Check some high-z limits
    cosm_hi_z = Cosmology(omega_m_0=0.99999, omega_l_0=1e-5)
    
    # Age of the Universe
    assert np.allclose(cosm_hi_z.t_of_z(0.), 2. / 3. / cosm_hi_z.hubble_0)
    
    # Check high-z limit for Hubble parameter. Better than 1%?
    H_n = cosm.HubbleParameter(30.)
    H_a = cosm.hubble_0 * np.sqrt(cosm.omega_m_0) * (1. + 30.)**1.5
    assert abs(H_n - H_a) / H_a < rtol, \
        "Hubble parameter @ high-z not accurate to < {:.3g}%.".format(rtol)
        
if __name__ == '__main__':
    test()    

