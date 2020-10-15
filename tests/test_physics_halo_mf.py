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


@pytest.mark.parametrize("cosmology_package", ["ccl", None])
@pytest.mark.parametrize("hmf_package", ["ccl", "hmf"])
@pytest.mark.parametrize("hmf_model", ["Tinker10"])
@pytest.mark.parametrize("use_class", [True, False])
def test_hmf(
    cosmology_package,
    hmf_package,
    hmf_model,
    use_class,
    hmf_load=False,
    hmf_zmin=1,
    hmf_zmax=60.0,
    hmf_dz=10.0,
):
    kwargs = dict(
        cosmology_package=cosmology_package,
        hmf_load=hmf_load,
        hmf_package=hmf_package,
        hmf_model=hmf_model,
        hmf_zmin=hmf_zmin,
        hmf_zmax=hmf_zmax,
        hmf_dz=hmf_dz,
    )

    if use_class:
        if not has_classy:
            return
        if not (cosmology_package == "ccl" and hmf_package == "ccl"):
            return

        kmax = 200
        z_pk = 60.0
        class_params = {"P_k_max_1/Mpc": kmax, "output": "dTk,mPk,tCl", "z_pk": z_pk}
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
        print(f"{attr}...")
        assert np.isnan(val).sum() / np.size(val) == 0
        print(f"...ok.")

    pop.halos.TabulateHMF(save_MAR=False)

    assert np.isnan(pop.halos.tab_dndm).sum() / np.size(val) == 0


if __name__ == "__main__":
    tests()
