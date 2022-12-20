"""

test_gs_backcompat.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Wed Apr  4 09:37:25 PDT 2018

Description: Make sure changes in input parameters result in changes in
the signal! Also a test of backward compatibility.

"""


import ares
import numpy as np

def test():

    oldp = ['fstar', 'fX', 'Tmin', 'Nion', 'Nlw']
    newp = ['pop_fstar{0}', 'pop_rad_yield{1}', 'pop_Tmin{0}',
        'pop_Nion{2}', 'pop_Nlw{0}']
    oldv = [(0.05, 0.2), (0.1, 1.), (1e3, 1e4), (1e3, 1e4), (1e3, 1e4)]
    newv = [(0.05, 0.2), (2.6e38, 2.6e39), (1e3, 1e4), (1e3, 1e4), (1e3, 1e4)]

    pars = {'old': oldp, 'new': newp}
    vals = {'old': oldv, 'new': newv}

    base = ares.util.ParameterBundle('global_signal:basic')
    kw = ares.util.ParameterBundle('speed:careless')

    for h, approach in enumerate(['new']):
        ax = None

        for i, par in enumerate(pars[approach]):

            data = []
            for val in vals[approach][i]:
                p = base.copy()
                p[par] = val
                p.update(kw)
                sim = ares.simulations.Simulation(**p)
                sim_gs = sim.get_21cm_gs()

                data.append((sim_gs.history['z'], sim_gs.history['dTb']))

            for j in range(len(data) - 1):

                z1, T1 = data[j]
                z2, T2 = data[j+1]

                # In this case, the sims were definitely different, since the
                # only way to change the number of redshift points is through
                # real-time timestep adjustment (driven by sources)
                if T1.size != T2.size:
                    continue

                neq = np.not_equal(T1, T2)

                assert np.any(neq), "Changes in par={} had no effect!".format(par)

if __name__ == '__main__':
    test()
