"""

test_pop_models.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Fri Jul 15 15:23:11 PDT 2016

Description: 

"""

import ares
import matplotlib.pyplot as pl

PB = ares.util.ParameterBundle

def test():

    # Create a simple population
    pars_1 = PB('pop:fcoll') + PB('sed:bpass')
    pop_fcoll = ares.populations.GalaxyPopulation(**pars_1)
    #pop_fcoll_XR = ares.populations.GalaxyPopulation(**pars_1)
    
    # Mimic the above population to check our different SFRD/SED techniques
    sfrd_pars = {'pop_sfr_model': 'sfrd-func'}
    sfrd_pars['pop_sfrd'] = pop_fcoll.SFRD
    sfrd_pars['pop_sfrd_units'] = 'internal'
    
    sed = PB('sed:toy')
    sed['pop_Nion'] = pop_fcoll.src.Nion
    sed['pop_Nlw'] = pop_fcoll.src.Nlw
    # pop_Ex?
    sed['pop_ion_src_igm'] = False
    sed['pop_heat_src_igm'] = False
    
    pars_2 = sed + sfrd_pars
    
    pop_sfrd = ares.populations.GalaxyPopulation(**pars_2)
    
    assert pop_fcoll.SFRD(20.) == pop_sfrd.SFRD(20.), "Error in SFRD."

    # Check the emissivities too 
    
    #print(pop_fcoll.PhotonLuminosityDensity(20., Emin=10.2, Emax=13.6))
    #print(pop_sfrd.PhotonLuminosityDensity(20., Emin=10.2, Emax=13.6))
    
    #assert pop_fcoll.PhotonLuminosityDensity(20., Emin=10.2, Emax=13.6) \
    #    == pop_sfrd.PhotonLuminosityDensity(20., Emin=10.2, Emax=13.6), \
    #    "Error in photon luminosity density."

if __name__ == '__main__':
    test()


