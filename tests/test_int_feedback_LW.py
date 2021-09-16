"""

test_feedback_LW.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Sun Mar 25 16:32:57 PDT 2018

Description:

"""


import ares
import numpy as np
from ares.physics.Constants import rhodot_cgs

def test():

    print("Skipping this test. Must use parametric SED.")
    return


    pars = ares.util.ParameterBundle('mirocha2017:base') \
         + ares.util.ParameterBundle('mirocha2018:high')

    pars['pop_sfr{2}'] = 1e-6        # ~1 100 Msun star every 10 Myr
    pars['pop_time_limit{2}'] = 20.
    pars['pop_bind_limit{2}'] = None

    # So we don't need SPS models in test suite
    #pars['pop_sed{0}'] = 'pl'
    #pars['pop_rad_yield{0}'] = 4000
    #pars['pop_rad_yield_units{0}'] = 'photons/baryon'
    #pars['pop_Emin{0}'] = 10.2
    #pars['pop_Emax{0}'] = 13.6
    #pars['pop_calib_L1600{0}'] = None

    # just to speed things up a bit
    pars['feedback_LW_maxiter'] = 4
    pars['sam_atol'] = 1e-1
    pars['sam_rtol'] = 1e-1
    pars['feedback_LW_sfrd_rtol'] = 1e-1
    pars['tau_redshift_bins'] = 200
    del pars['problem_type']

    # Strip away the X-ray sources for now.
    pop2 = pars.pars_by_pop(0, True)
    pop2.num = 0
    pop3 = pars.pars_by_pop(2, True)
    pop3.num = 1

    bkw = pars.get_base_kwargs()

    pars = bkw + pop2 + pop3

    # Correct ID number of link
    pars['pop_Mmin{0}'] = 'link:Mmax:1'
    pars['feedback_LW_sfrd_popid'] = 1

    sim = ares.simulations.MetaGalacticBackground(**pars)
    sim.run()

    popII  = sim.pops[0]
    popIII = sim.pops[1]

    # Convert SFRD from g/s/cm^3 to Msun/yr/cMpc^3
    sfrdII = popII._tab_sfrd_total * rhodot_cgs
    sfrdIII = popIII._tab_sfrd_total * rhodot_cgs

    # Could check Mmin iterations
    for i, Mmin in enumerate(sim._Mmin_bank):
        print(i, Mmin)

    assert True

if __name__ == '__main__':
    test()
