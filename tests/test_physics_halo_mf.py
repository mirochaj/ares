"""

test_physics_hmf.py

Author: Jordan Mirocha, Timothy Morton
Affiliation: McGill, USC
Created on: Wed 25 Mar 2020 11:15:35 EDT

Description: 

"""

import ares
import numpy as np
import pytest

try:
    import classy

    has_classy = True
except ImportError:
    has_classy = False


def tests():
    pop = ares.populations.HaloPopulation()  # noqa


@pytest.mark.parametrize("cosmology_package", [None, "ccl"])
@pytest.mark.parametrize("hmf_package", ["hmf", "ccl"])
@pytest.mark.parametrize("hmf_model", ["Tinker10"])
@pytest.mark.parametrize("use_class", [True, False])
def test_hmf(cosmology_package, hmf_package, hmf_model, use_class, hmf_load=False):
    kwargs = dict(
        cosmology_package=cosmology_package, hmf_load=hmf_load, hmf_package=hmf_package, hmf_model=hmf_model
    )

    if use_class:
        if not has_classy:
            return
        if not (cosmology_package == "ccl" and hmf_package == "ccl"):
            return

        kmax = 200
        class_params = {"P_k_max_1/Mpc": kmax, "output": "dTk,mPk,tCl"}
        cl = classy.Class()
        cl.set(class_params)
        cl.compute()
        kwargs["cosmology_helper"] = cl
        kwargs["kmax"] = kmax

    # Shouldn't the below hmf/cosmology consistency be caught here?
    pop = ares.populations.HaloPopulation(**kwargs)

    for attr in ["tab_k_lin", "tab_ps_lin", "tab_ngtm", "tab_M"]:
        try:
            val = getattr(pop.halos, attr)
        except AssertionError:
            if hmf_package == "ccl" and cosmology_package != "ccl":
                return
        assert np.isnan(val).sum() == 0

    pop.halos.TabulateHMF(save_MAR=False)

    assert np.isnan(pop.halos.tab_dndm).sum() == 0


if __name__ == "__main__":
    tests()
