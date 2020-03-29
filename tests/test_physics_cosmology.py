"""

test_physics_cosm.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Sep 24 15:39:44 MDT 2014

Description: 

"""

import numpy as np
from ares.physics import Cosmology
from ares.physics.Constants import s_per_gyr, m_H, m_He, cm_per_mpc

def test(rtol=1e-3):
    
    cosm = Cosmology()
    
    # Check some high-z limits
    cosm_appr = Cosmology(approx_highz=True)
    
    # Check critical density
    assert cosm.CriticalDensity(0.) == cosm.CriticalDensityNow
    
    # Make sure energy densities sum to unity
    assert np.allclose(cosm.omega_m_0, 1. - cosm.omega_l_0)
    
    # Make sure the age of the Universe is OK
    assert 13.5 <= cosm.t_of_z(0.) / s_per_gyr <= 14.
    
    # Check high-z limit for Hubble parameter. Better than 1%?
    H_n = cosm.HubbleParameter(30.)
    H_a = cosm_appr.HubbleParameter(30.)    
    assert abs(H_n - H_a) / H_a < rtol, \
        "Hubble parameter @ high-z not accurate to < {:.3g}%.".format(rtol)
        
    # Check high-z limit for comoving radial distance
    R_n = cosm_appr.ComovingRadialDistance(20., 30.) / cm_per_mpc
    R_a = cosm.ComovingRadialDistance(20., 30.) / cm_per_mpc
    
    assert abs(R_a - R_n) / R_a < rtol, \
        "Comoving radial distance @ high-z not accurate to < {:.3g}%.".format(rtol)
        
    # Test a user-supplied cosmology and one that grabs a row from Planck chain
    # Remember: test suite doesn't have CosmoRec, so don't use get_inits_rec.
    cosm = Cosmology(cosmology_name='user', cosmology_id='jordan')
    
    cosm = Cosmology(cosmology_id=100)
                
if __name__ == '__main__':
    test()    

