"""

test_halo_model.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Fri Jul  8 12:24:24 PDT 2016

Description: 

"""

import ares
from ares.simulations.PowerSpectrum import PowerSpectrum

def test():

    #pars = ares.util.ParameterBundle('mirocha2016:dpl')
    #
    #for dr in [0.01, 0.05, 0.1, 0.5]:
    #
    #    sim = PowerSpectrum(powspec_logkmin=-3, powspec_logkmax=3, 
    #        powspec_dlogk=0.2, hmf_load_ps=True, powspec_dlogr=dr, **pars)
    #    
    #    sim.run(z=10)
    #    
    #    ax = sim.PowerSpectrum(10)
    #
    #pl.savefig('{!s}.png'.format(__file__[0:__file__.rfind('.')])
    #pl.close()

    assert True

if __name__ == '__main__':
    test()

