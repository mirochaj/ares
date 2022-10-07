"""

test_gs_multipop.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Fri Jul  1 15:42:30 PDT 2016

Description:

"""

import ares

def test():

    base = ares.util.ParameterBundle('global_signal:basic')
    fcoll = ares.util.ParameterBundle('pop:fcoll')
    popIII = ares.util.ParameterBundle('sed:uv')

    pop = fcoll + popIII

    # Restrict to halos below the atomic cooling threshold
    pop['pop_fstar'] = 1e-3
    pop['pop_Tmin'] = 300
    pop['pop_Tmax'] = 1e4

    # Tag with ID number
    pop.num = 3

    sim1 = ares.simulations.Simulation(**base)

    new = base + pop
    sim2 = ares.simulations.Simulation(**new)

    sim1.get_21cm_gs()
    sim2.get_21cm_gs()

    T1 = sim1.sim_gs.history['dTb']
    T2 = sim2.sim_gs.history['dTb']

    if T1.size != T2.size:
        pass
    else:
        neq = np.not_equal(T1, T2)

        assert np.any(neq), \
            "Addition of fourth population should change signal!"

    # Adding source population should shift timing of features earlier
    assert sim2.sim_gs.nu_B <= sim1.sim_gs.nu_B
    assert sim2.sim_gs.nu_C <= sim1.sim_gs.nu_C



if __name__ == '__main__':
    test()
