"""

test_populations_ensemble.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Fri 27 Mar 2020 10:16:25 EDT

Description: 

"""

import ares
import numpy as np
from ares.physics.Constants import rhodot_cgs

def test():
    pars = ares.util.ParameterBundle('mirocha2020:univ')
    
    # Can't actually do this test yet because we don't have access to 
    # even time-spaced HMFs/HGHs on travis.ci
    
    #pop = ares.populations.GalaxyPopulation(**pars)
    #
    #sfrd = pop.SFRD(6.) * rhodot_cgs
    #
    #assert 1e-3 <= sfrd <= 1, "SFRD is unreasonable"
    #
    ## Just check dust masss etc.
    #Md = pop.get_field(6., 'Md')
    #
    #assert 1e4 <= np.mean(Md) <= 1e10
    #
    #AUV = pop.AUV(6.)
    #
    #assert np.all(AUV > 0)
    #assert 0 < np.mean(AUV) <= 3
    

if __name__ == '__main__':
    test()    


    