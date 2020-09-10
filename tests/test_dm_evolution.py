import ares
import numpy as np
import os
from astropy.cosmology import Planck18_arXiv_v2 as Planck

h = Planck.h
class LCDM:
    h = h
    H0 = Planck.H0.value
    omega_b = Planck.Ob0 * h ** 2
    omega_cdm = (Planck.Om0 - Planck.Ob0) * h ** 2
    omega_nu = Planck.Onu0
    omega_k = Planck.Ok0

    m_ncdm = sum(Planck.m_nu).value
    Neff = Planck.Neff
    N_ncdm = 1
    N_ur = Planck.Neff - N_ncdm

    Tcmb = Planck.Tcmb0.value
    A_s = 2.097e-9
    tau_reio = 0.0540
    n_s = 0.9652
    YHe = 0.24537116583825905
    reion_exponent = 1.5
    reion_width = 0.5

def generate_inputs(inputs_fname):
    import classy
    class_params = {
        'h': LCDM.h, 'omega_b': LCDM.omega_b, 'omega_cdm': LCDM.omega_cdm,
        'Omega_k': LCDM.omega_k, 'N_ur': LCDM.N_ur, 'N_ncdm': LCDM.N_ncdm,
        'm_ncdm': LCDM.m_ncdm, 'A_s': LCDM.A_s, 'n_s': LCDM.n_s,
        'T_cmb': LCDM.Tcmb, 'tau_reio': LCDM.tau_reio, 'YHe': LCDM.YHe,
        'reionization_exponent': LCDM.reion_exponent,
        'reionization_width': LCDM.reion_width, 'P_k_max_1/Mpc': 200,
        'output': 'dTk,mPk,tCl',
    }

    dm_params = {
        'omega_dmeff': 0.12038, 'omega_cdm': 1e-10, 'm_dmeff': 1.0,
        'sigma_dmeff': 1e-25, 'npow_dmeff': 0,
    }

    cl = classy.Class()
    cl.set(class_params)
    cl.set(dm_params)
    print("Running Class Compute...")
    cl.compute()

    thermo = cl.get_thermodynamics()
    input_params = {
        'z': thermo['z'], 'xe': thermo['x_e'], 'Tk': thermo['Tb [K]'],
        'Tchi': thermo['T_dmeff'],
    }
    np.savez(inputs_fname, **input_params)
    return input_params

def main():
    inputs_fname = 'class_inputs.npz'
    if not os.path.exists(inputs_fname):
        print(f"Generating inputs file {inputs_fname}")
        inputs = generate_inputs(inputs_fname)
    else:
        print(f"Found existing input file.")
        inputs = dict(np.load(inputs_fname))

    Vrms = 29 * 1e5

    inputs['Vchib'] = Vrms
    inputs['Vchib'] = 0
    print("Creating Ares simulation")
    dm_params = {
        'omega_dmeff': 0.12038, 'omega_cdm': 1e-10, 'm_dmeff': 1.0,
        'sigma_dmeff': 1e-25, 'npow_dmeff': 0,
    }
    hmf_table = '/Users/jacoblashner/dm/lib/ares//input/hmf/hmf_ST_planck_TTTEEE_lowl_lowE_best_logM_1400_4-18_z_1201_0-60.hdf5'
    sim = ares.simulations.Global21cm(
        cosmology_inits=inputs, include_dm=True, verbose=True,
        m_dmeff=1.0, npow_dmeff=0,
        sigma_dmeff = 1e-25,
        cosmology_name='user', hmf_table=hmf_table
    )
    sim.run()

if __name__ == "__main__":
    main()
