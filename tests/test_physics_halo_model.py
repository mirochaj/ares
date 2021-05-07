"""

test_halo_model.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Fri Jul  8 12:24:24 PDT 2016

Description:

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

def test(rtol=1e-3):

    hm = ares.physics.HaloModel()

    # Just check large k range, k < 1 / Mpc
    k = np.logspace(-2, 0, 10)

    # Compare linear matter power spectrum to 2-h term of halo model
    for z in [6, 10, 20]:

        iz = np.argmin(np.abs(hm.tab_z - z))
        plin = np.exp(np.interp(np.log(k), np.log(hm.tab_k_lin),
            np.log(hm.tab_ps_lin[iz,:])))

        # Will default to using k-values used for plin if nothing supplied
        ps2h = hm.get_ps_2h(z, k)

        pl.loglog(k, k**3 * plin / 2. / np.pi**2, color='k')
        pl.loglog(k, k**3 * ps2h / 2. / np.pi**2, color='b',
            ls='--', lw=3)
        pl.xlabel(r'$k \ [\mathrm{cMpc}^{-1}]$')
        pl.ylabel(r'$\Delta^2(k)$')

        rerr = np.abs((plin - ps2h) / plin)
        assert np.allclose(plin, ps2h, rtol=rtol), \
            "2h term != linear matter PS at z={}. err_rel(k)={}".format(z,
                rerr)

    pl.savefig('{!s}.png'.format(__file__[0:__file__.rfind('.')]))
    pl.close()

if __name__ == '__main__':
    test()
