"""
SetDefaultParameterValues.py

Author: Jordan Mirocha / Joshua Hibbard
Affiliation: University of Colorado at Boulder
Created on 2010-10-19.

Description: Defaults for all different kinds of parameters.

"""
import os

import numpy as np

from .. import rcParams
from ..data import ARES
from ..physics.Constants import m_H, cm_per_kpc, s_per_myr, E_LL

inf = np.inf

pgroups = [
    'Grid',
    'Physics',
    'Cosmology',
    'Source',
    'Population',
    'Control',
    'HaloMassFunction',
    'Tanh',
    'Gaussian',
    'Slab',
    'MultiPhase',
    'Dust',
    'ParameterizedQuantity',
    'Old',
    'PowerSpectrum',
    'Halo',
    'Absorber',
]

# Start setting up list of parameters to be set
defaults = []

for grp in pgroups:
    defaults.append('{!s}Parameters()'.format(grp))

def SetAllDefaults():
    pf = {}

    for pset in defaults:
        pf.update(eval('{!s}'.format(pset)))

    return pf

def GridParameters():
    pf = {
        "grid_cells": 64,
        "start_radius": 0.01,
        "logarithmic_grid": False,

        "density_units": 1e-3,            # H number density
        "length_units": 10.0 * cm_per_kpc,
        "time_units": s_per_myr,

        "include_He": False,
        "include_H2": False,

        # For MultiPhaseMedium calculations
        "include_cgm": True,
        "include_igm": True,

        # Line photons
        "include_injected_lya": True,

        "initial_ionization": [1.0 - 1e-8, 1e-8, 1.0 - 2e-8, 1e-8, 1e-8],
        "initial_temperature": 1e4,

        # These have shape len(absorbers)
        "tables_logNmin": [None],
        "tables_logNmax": [None],
        "tables_dlogN": [0.1],

        # overrides above parameters
        "tables_logN": None,

        "tables_xmin": [1e-8],

        "tables_discrete_gen": False,
        "tables_energy_bins": 100,
        "tables_prefix": None,

        "tables_logxmin": -4,
        "tables_dlogx": 0.1,
        "tables_dE": 5.0,

        "tables_times": None,
        "tables_dt": s_per_myr,
    }

    pf.update(rcParams)

    return pf

def MultiPhaseParameters():
    """
    These are grid parameters -- we'll strip off the prefix in
    MultiPhaseMedium calculations.
    """
    pf = {
        "cgm_grid_cells": 1,
        "cgm_expansion": True,
        "cgm_initial_temperature": [1e4],
        "cgm_initial_ionization": [1.0 - 1e-8, 1e-8, 1.0 - 2e-8, 1e-8, 1e-8],
        "cgm_isothermal": True,
        "cgm_recombination": 'A',
        "cgm_collisional_ionization": False,
        "cgm_cosmological_ics": False,

        "monotonic_EoR": 1e-6,

        "igm_grid_cells": 1,
        "igm_expansion": True,
        "igm_initial_temperature": None,
        'igm_initial_ionization': [1.0 - 1e-8, 1e-8, 1.0 - 2e-8, 1e-8, 1e-8],
        "igm_isothermal": False,
        "igm_recombination": 'B',
        "igm_compton_scattering": True,
        "igm_collisional_ionization": True,
        "igm_cosmological_ics": False,
    }

    pf.update(rcParams)

    return pf

def SlabParameters():

    pf = {
        "slab": 0,
        "slab_position": 0.1,
        "slab_radius": 0.05,
        "slab_overdensity": 100,
        "slab_temperature": 100,
        "slab_ionization": [1.0 - 1e-8, 1e-8],
        "slab_profile": 0,
    }

    pf.update(rcParams)

    return pf

# BoundaryConditionParameters?
def FudgeParameters():
    pf = {
        "z_heII_EoR": 3.0,
    }

    pf.update(rcParams)

    return pf

def AbsorberParameters():
    pf = {
        'cddf_C': 0.25,
        'cddf_beta': 1.4,
        'cddf_gamma': 1.5,
        'cddf_zlow': 1.5,
        'cddf_gamma_low': 0.2,
    }

    pf.update(rcParams)

    return pf

def PhysicsParameters():
    pf = {
        "radiative_transfer": 1,
        "photon_conserving": 1,
        "plane_parallel": 0,
        "infinite_c": 1,

        "collisional_ionization": 1,

        "secondary_ionization": 1,  # 0 = Deposit all energy as heat
                                    # 1 = Shull & vanSteenberg (1985)
                                    # 2 = Ricotti, Gnedin, & Shull (2002)
                                    # 3 = Furlanetto & Stoever (2010)

        "secondary_lya": False,     # Collisionally excited Lyman alpha?

        "isothermal": 1,
        "expansion": 0,             # Referring to cosmology
        "collapse": 0,              # Referring to single-zone collapse
        "compton_scattering": 1,
        "recombination": 'B',
        "exotic_heating": False,
        'exotic_heating_func': None,

        "clumping_factor": 1,

        "approx_H": False,
        "approx_He": False,
        "approx_sigma": False,
        "approx_Salpha": 1, # 1 = Salpha = 1
                            # 2 = Chuzhoy, Alvarez, & Shapiro (2006),
                            # 3 = Furlanetto & Pritchard (2006)
                            # 4 = Hirata (2006)
                            # 5 = Mittal & Kulkarni (2018)

        "lya_heating": False,
        "approx_lya_Ii": False,
        "spin_exchange": False,
        "approx_tau_21cm": True,
        "extrapolate_coupling": False,

        "approx_thermal_history": False,
        "inits_Tk_p0": None,
        "inits_Tk_p1": None,
        "inits_Tk_p2": None, # Set to -4/3 if thermal_hist = 'exp' to recover
                             # adiabatic cooling
        "inits_Tk_p3": 0.0,
        "inits_Tk_p4": inf,
        "inits_Tk_p5": None,
        "inits_Tk_dz": 1.0,

        "Tbg": None,
        "Tbg_p0": None,
        "Tbg_p1": None,
        "Tbg_p2": None,
        "Tbg_p3": None,
        "Tbg_p4": None,

        # Ad hoc way to make a flattened signal
        "floor_Ts": False,
        "floor_Ts_p0": None,
        "floor_Ts_p1": None,
        "floor_Ts_p2": None,
        "floor_Ts_p3": None,
        "floor_Ts_p4": None,
        "floor_Ts_p5": None,

        # Lyman alpha sources
        "lya_nmax": 23,

        "rate_source": 'fk94', # fk94, option for development here

        # Feedback parameters

        # LW
        'feedback_clear_solver': True,

        'feedback_LW': False,
        'feedback_LW_dt': 0.0,  # instantaneous response
        'feedback_LW_Mmin': 'visbal2014',
        'feedback_LW_fsh': None,
        'feedback_LW_Tcut': 1e4,
        'feedback_LW_mean_err': False,
        'feedback_LW_maxiter': 15,
        'feedback_LW_miniter': 0,
        'feedback_LW_softening': 'sqrt',
        'feedback_LW_tol_zrange': (0, np.inf),

        'feedback_LW_Mmin_monotonic': False,
        'feedback_LW_Mmin_smooth': 0,
        'feedback_LW_Mmin_fit': 0,
        'feedback_LW_Mmin_afreq': 0,
        'feedback_LW_Mmin_rtol': 0.0,
        'feedback_LW_Mmin_atol': 0.0,
        'feedback_LW_sfrd_rtol': 1e-1,
        'feedback_LW_sfrd_atol': 0.0,
        'feedback_LW_sfrd_popid': None,
        'feedback_LW_zstart': None,
        'feedback_LW_mixup_freq': 5,
        'feedback_LW_mixup_delay': 20,
        'feedback_LW_guesses': None,
        'feedback_LW_guesses_from': None,
        'feedback_LW_guesses_perfect': False,

        # Assume that uniform background only emerges gradually as
        # the typical separation of halos becomes << Hubble length
        "feedback_LW_ramp": 0,

        'feedback_streaming': False,
        'feedback_vel_at_rec': 30.0,

        'feedback_Z': None,
        'feedback_Z_Tcut': 1e4,
        'feedback_Z_rtol': 0.0,
        'feedback_Z_atol': 1.0,
        'feedback_Z_mean_err': False,
        'feedback_Z_Mmin_uponly': False,
        'feedback_Z_Mmin_smooth': False,

        'feedback_tau': None,
        'feedback_tau_Tcut': 1e4,
        'feedback_tau_rtol': 0.0,
        'feedback_tau_atol': 1.0,
        'feedback_tau_mean_err': False,
        'feedback_tau_Mmin_uponly': False,
        'feedback_tau_Mmin_smooth': False,

        'feedback_ion': None,
        'feedback_ion_Tcut': 1e4,
        'feedback_ion_rtol': 0.0,
        'feedback_ion_atol': 1.0,
        'feedback_ion_mean_err': False,
        'feedback_ion_Mmin_uponly': False,
        'feedback_ion_Mmin_smooth': False,
    }

    pf.update(rcParams)

    return pf

def ParameterizedQuantityParameters():
    pf = {
        "pq_func": 'dpl',
        "pq_func_fun": None,  # only used if pq_func == 'user'
        "pq_func_var": 'Mh',
        "pq_func_var2": None,
        "pq_func_var_lim": None,
        "pq_func_var2_lim": None,
        "pq_func_var_fill": 0.0,
        "pq_func_var2_fill": 0.0,
        "pq_func_par0": None,
        "pq_func_par1": None,
        "pq_func_par2": None,
        "pq_func_par3": None,
        "pq_func_par4": None,
        "pq_func_par5": None,
        "pq_func_par6": None,
        "pq_func_par7": None,
        "pq_func_par8": None,
        "pq_func_par9": None,
        "pq_func_par10": None,
        "pq_func_par11": None,
        "pq_func_par12": None,
        "pq_func_par13": None,
        "pq_func_par14": None,
        "pq_func_par15": None,
        "pq_func_par16": None,

        "pq_boost": 1.0,
        "pq_iboost": 1.0,
        "pq_val_ceil": None,
        "pq_val_floor": None,
        "pq_var_ceil": None,
        "pq_var_floor": None,
    }

    pf.update(rcParams)

    return pf

def DustParameters():
    pf = {}

    tmp = {
        'dustcorr_method': None,

        'dustcorr_beta': -2.0,

        # Only used if method is a list
        'dustcorr_ztrans': None,

        # Intrinsic scatter in the AUV-beta relation
        'dustcorr_scatter_A': 0.0,
        # Intrinsic scatter in the beta-mag relation (gaussian)
        'dustcorr_scatter_B': 0.34,

        'dustcorr_Bfun_par0': -2.0,
        'dustcorr_Bfun_par1': None,
        'dustcorr_Bfun_par2': None,
    }

    pf.update(tmp)
    pf.update(rcParams)

    return pf

def PowerSpectrumParameters():
    pf = {}

    tmp = {
        'ps_output_z': np.arange(6, 20, 1),
        'ps_output_waves': None,

        "ps_output_k": None,
        "ps_output_lnkmin": -4.6, # 0.01
        "ps_output_lnkmax": 2.3,  # 10
        "ps_output_dlnk": 0.1,

        "ps_output_R": None,
        "ps_output_lnRmin": -8.0,
        "ps_output_lnRmax": 8.0,
        "ps_output_dlnR": 0.01,

        # New parameters as of 02.09.2022
        "ps_space_ion": 'real',
        "ps_space_temp": 'fourier',
        "ps_space_lya": 'fourier',
        "ps_method": 1,

        'ps_use_wick': False,

        'ps_igm_model': 1, # 1=3-zone IGM, 2=other

        'ps_include_acorr': True,
        'ps_include_xcorr': False,
        'ps_include_bias': True,

        'ps_include_xcorr_wrt': None,

        # Save all individual pieces that make up 21-cm PS?
        "ps_output_components": False,

        'ps_include_21cm': True,
        'ps_include_density': True,
        'ps_include_ion': True,
        'ps_include_temp': False,
        'ps_include_lya': False,
        'ps_lya_cut': inf,

        # Binary model switches
        'ps_include_xcorr_ion_rho': False,
        'ps_include_xcorr_hot_rho': False,
        'ps_include_xcorr_ion_hot': False,

        'ps_include_3pt': True,
        'ps_include_4pt': True,

        'ps_temp_model': 1,  # 1=Bubble shells, 2=FZH04
        'ps_saturated': 10.0,

        'ps_correct_gs_ion': True,
        'ps_correct_gs_temp': True,

        'ps_assume_saturated': False,

        'ps_split_transform': True,
        'ps_fht_rtol': 1e-4,
        'ps_fht_atol': 1e-4,

        'ps_include_lya_lc': False,

        "ps_volfix": True,

        "ps_rescale_Qlya": False,
        "ps_rescale_Qhot": False,
        "ps_rescale_dTb": False,

        "bubble_size": None,
        "bubble_density": None,

        # Important that the number is at the end! ARES will interpret
        # numbers within underscores as population ID numbers.
        "bubble_shell_rvol_zone_0": None,
        "bubble_shell_rdens_zone_0": 0.0,
        "bubble_shell_rsize_zone_0": None,
        "bubble_shell_asize_zone_0": None,
        "bubble_shell_ktemp_zone_0": None,
        #"bubble_shell_tpert_zone_0": None,
        #"bubble_shell_rsize_zone_1": None,
        #"bubble_shell_asize_zone_1": None,
        #"bubble_shell_ktemp_zone_1": None,
        #"bubble_shell_tpert_zone_1": None,
        #"bubble_shell_rsize_zone_2": None,
        #"bubble_shell_asize_zone_2": None,
        #"bubble_shell_ktemp_zone_2": None,
        #"bubble_shell_tpert_zone_2": None,

        "bubble_shell_include_xcorr": True,

        #"bubble_pod_size": None,
        #"bubble_pod_size_rel": None,
        #"bubble_pod_size_abs": None,
        #"bubble_pod_size_func": None,
        #"bubble_pod_temp": None,
        #"bubble_pod_Nsc": 1e3,

        "ps_lya_method": 'lpt',
        "ps_ion_method": None,  # unused

        #"powspec_lya_approx_sfr": 'exp',

        "bubble_shell_size_dist": None,
        "bubble_size_dist": 'fzh04', # or FZH04, PC14
    }

    pf.update(tmp)
    pf.update(rcParams)

    return pf

def PopulationParameters():
    """
    Parameters associated with populations of objects, which give rise to
    meta-galactic radiation backgrounds.
    """

    pf = {}

    # Grab all SourceParameters and replace "source_" with "pop_"
    srcpars = SourceParameters()
    for par in srcpars:
        pf[par.replace('source', 'pop')] = srcpars[par]

    tmp = {
        "pop_type": 'galaxy',

    "pop_type": 'galaxy',

    "pop_target_volume": None,
    "pop_target_redshift": None,
    "pop_target_density": 0.,
    "pop_target_seed": None,

    "pop_tunnel": None,

    "pop_sfr_model": 'fcoll', # or sfrd-func, sfrd-tab, sfe-func, sfh-tab, rates,

    "pop_ham_z": None,

    # Mass accretion rate
    "pop_MAR": 'hmf',
    "pop_MAR_interp": 'linear',
    "pop_MAR_corr": None,
    "pop_MAR_delay": None,
    "pop_MAR_from_hist": True,

    "pop_interp_MAR": 'linear',
    "pop_interp_sfrd": 'linear',
    "pop_interp_lf": 'linear',

    "pop_tdyn": 1e7,
    "pop_tstar": None,
    "pop_ssfr": None,


    "pop_uvlf": None,
    'pop_lf_Mmax': 1e15,

    "pop_fduty": None,
    "pop_fduty_seed": None,
    "pop_fduty_dt": None, # if not None, SF occurs in on/off bursts, i.e.,
                          # it's coherent.

    "pop_centrals": True,
    "pop_ihl": None,

    "pop_focc": 1.0,
    "pop_focc_inv": False,

    "pop_fsurv": 1.0,
    "pop_fsurv_inv": False,

    "pop_fsup": 0.0,  # Suppression of star-formation at threshold

    # Set the emission interval and SED
    "pop_sed": 'pl',
    "pop_sed_table": None,

    "pop_sed_sharp_at": None,

    "pop_sed_null_except": None,

    # Can degrade spectral resolution of stellar population synthesis models
    # just to speed things up.
    "pop_sed_degrade": None,

    # If pop_sed == 'user'
    "pop_E": None,
    "pop_L": None,

    # For synthesis models
    "pop_Z": 0.02,
    "pop_imf": 2.35,
    "pop_tracks": None,
    "pop_tracks_fn": None,
    "pop_stellar_aging": False,
    "pop_nebular": False,
    "pop_nebular_only": False,
    "pop_nebular_continuum": True,
    "pop_nebular_lines": True,
    "pop_nebular_ff": True,
    "pop_nebular_fb": True,
    "pop_nebular_2phot": True,
    "pop_nebular_lookup": None,
    "pop_ssp": False,             # a.k.a., continuous SF
    "pop_psm_instance": None,
    "pop_src_instance": None,

    # Cache tricks: must be pickleable for MCMC to work.
    "pop_sps_data": None,

    "pop_age": 100.,
    "pop_tneb": None,
    "pop_binaries": False,        # for BPASS
    "pop_sed_by_Z": None,

    # Used only for approximate SFH treatments in GalaxyCohort
    # "real" spectral synthesis done in GalaxyEnsemble by default
    "pop_sfh": 'const',
    "pop_sfh_degrade": 1,
    "pop_sfh_fallback": None,
    "pop_age_definition": None,


    # Numerics of specral synthesis
    "pop_ssp_oversample": False,
    "pop_ssp_oversample_age": 30.,

    # Option of setting Z, t, or just supplying SSP table?

    "pop_Emin": 2e2,
    "pop_Emax": 3e4,
    "pop_EminNorm": 5e2,
    "pop_EmaxNorm": 8e3,
    "pop_Enorm": None,

    "pop_lmin": None,
    "pop_lmax": None,
    "pop_dlam": None,
    "pop_wavelengths": None,
    "pop_times": None,

    # Reserved for delta function sources
    "pop_E": None,
    "pop_LE": None,

    # Artificially kill emission in some band.
    "pop_Ekill": None,
    "pop_Emin_xray": 2e2,

    # Controls IGM ionization for approximate CXB treatments
    "pop_Ex": 500.,
    "pop_Euv": 30.,

    "pop_lf": None,
    "pop_emissivity": None,

    # By-hand parameterizations
    "pop_Ja": None,
    "pop_Tk": None,
    "pop_xi": None,
    "pop_ne": None,

    #
    "pop_ion_rate_cgm": None,
    "pop_ion_rate_igm": None,
    "pop_heat_rate": None,

    "pop_k_ion_cgm": None,
    "pop_k_ion_igm": None,
    "pop_k_heat_igm": None,

    # Set time interval over which emission occurs
    "pop_zform": 60.,
    "pop_zdead": 0.0,

    # Main parameters in our typical global 21-cm models
    "pop_fstar": 0.1,
    'pop_fstar_cloud': 1.,  # cloud-scale star formation efficiency
    "pop_fstar_max": 1.0,
    "pop_fstar_negligible": 1e-5, # relative to maximum

    "pop_facc": 0.0,

    # Next 3: relative to fraction of halo acquiring the material
    'pop_acc_frac_metals': 1.0,
    'pop_acc_frac_stellar': 1.0,
    'pop_acc_frac_gas': 1.0,
    'pop_metal_retention': 1.0,

    "pop_star_formation": True,

    "pop_sfe": None,
    "pop_mlf": None,
    "pop_sfr": None,
    "pop_frd": None,
    "pop_fshock": 1.0,

    # Halo model stuff
    "pop_prof_1h": None,
    "pop_1h_nebular_only": False,
    "pop_mask": None, # should be (wavelength or filter, limiting mag)
    "pop_mask_interp": None,

    # For GalaxyEnsemble
    "pop_aging": False,
    "pop_enrichment": False,
    "pop_quench": None,
    "pop_quench_method": 'zreion',
    "pop_flag_sSFR": None,
    "pop_flag_tau": None,

    "pop_mag_bin": 0.5,
    "pop_mag_min": -25,
    "pop_mag_max": 0,

    "pop_synth_dz": 0.5,
    "pop_synth_zmax": 15.,
    "pop_synth_zmin": 3.5,
    "pop_synth_Mmax": 1e14,
    "pop_synth_minimal": False,  # Can turn off for testing (so we don't need MF)
    "pop_synth_cache_level": 1, # Bigger = more careful
    "pop_synth_age_interp": 'cubic',
    "pop_synth_cache_phot": {},

    # Need to avoid doing synthesis in super duper detail for speed.
    # Still need to implement 'full' method.
    "pop_synth_lwb_method": 0,

    "pop_tau_bc": 0,
    "pop_age_bc": 10.,

    "pop_mergers": False,

    # For Clusters
    "pop_mdist": None,
    "pop_age_res": 1.,
    "pop_dlogM": 0.1,

    "pop_histories": None,
    "pop_halos": None,
    "pop_density": None,
    "pop_volume": None,

    "pop_guide_pop": None,
    "pop_thin_hist": False,
    "pop_scatter_mar": 0.0,
    "pop_scatter_mar_seed": None,
    "pop_scatter_sfr": 0.0,
    "pop_scatter_sfe": 0.0,
    "pop_scatter_env": 0.0,

    "pop_update_dt": 'native',

    # Cluster-centric model
    "pop_feedback_rad": False,
    "pop_feedback_sne": False,
    "pop_delay_rad_feedback": 0,
    "pop_delay_sne_feedback": 0,
    "pop_force_equilibrium": np.inf,
    "pop_sample_imf": False,
    "pop_sample_cmf": False,
    "pop_imf": 2.35,     # default to standard SSPs.
    "pop_imf_bins": None,#np.arange(0.1, 150.01, 0.01),  # bin centers
    "pop_cmf": None,

    # Feedback for single-stars
    "pop_coupling_sne": 0.1,
    "pop_coupling_rad": 0.1,

    # Energy per SNe in units of 1e51 erg.
    "pop_omega_51": 0.01,

    # Baryon cycling
    "pop_multiphase": False,

    "pop_fobsc": 0.0,
    "pop_fobsc_by": None, # or 'age' or 'lum'

    "pop_tab_z": None,
    "pop_tab_Mh": None,
    "pop_tab_sfe": None,
    "pop_tab_sfr": None,

    "pop_Tmin": 1e4,
    "pop_Tmax": None,
    "pop_Mmin": None,
    "pop_Mmin_ceil": None,
    "pop_Mmin_floor": None,
    "pop_Tmin_ceil": None,
    "pop_Tmin_floor": None,
    "pop_Tmax_ceil": None,
    "pop_Tmax_floor": None,
    "pop_Mmax_ceil": None,
    "pop_Mmax_floor": None,
    "pop_Mmax": None,

    "pop_time_limit": None,
    "pop_time_limit_delay": True,
    "pop_mass_limit": None,
    "pop_abun_limit": None,
    "pop_bind_limit": None,
    "pop_temp_limit": None,
    "pop_lose_metals": False,
    "pop_limit_logic": 'and',

    "pop_time_ceil": None,

    "pop_initial_Mh": 1, # In units of Mmin. Zero means unused

    "pop_sfrd": None,

    # For BHs
    "pop_bh_formation": False,
    "pop_bh_md": None,
    "pop_bh_ard": None,
    "pop_bh_seed_ratio": 1e-3,
    "pop_bh_seed_mass": None,
    "pop_bh_seed_eff": None,
    "pop_bh_facc": None,

    # Scales SFRD
    "pop_Nlw": 9690.,
    "pop_Nion": 4e3,
    "pop_fesc": 0.1,
    "pop_fX": 1.0,
    "pop_cX": 2.6e39,
    "pop_qdot": 5e48,
    "pop_lifetime": 1e10,
    "pop_temperature": 1e5,

    # Should
    "pop_fesc_LW": 1.,

    # Parameters that sweep fstar under the rug
    "pop_xi_XR": None,     # product of fstar and fX
    "pop_xi_LW": None,     # product of fstar and Nlw
    "pop_xi_UV": None,     # product of fstar, Nion, and fesc

    # Override luminosity density
    "pop_rhoL": None,

    # For multi-frequency calculations
    "pop_E": None,
    "pop_LE": None,

    # What radiation does this population emit?
    # These are passive fields
    "pop_oir_src": False,
    "pop_lw_src": True,
    "pop_lya_src": True,
    "pop_radio_src": False,

    # These are active fields (i.e., they change the IGMs properties)
    "pop_ion_src_cgm": True,
    "pop_ion_src_igm": True,
    "pop_heat_src_cgm": False,
    "pop_heat_src_igm": True,

    "pop_lya_fl": False,
    "pop_ion_fl": False,
    "pop_temp_fl": False,

    "pop_one_halo_term": True,
    "pop_two_halo_term": True,

    "pop_emissivity_tricks": True,

    # Generalized normalization
    # Mineo et al. (2012) (from revised 0.5-8 keV L_X-SFR)
    "pop_rad_yield": 2.6e39,
    "pop_rad_yield_units": 'erg/s/sfr',
    "pop_rad_yield_Z_index": None,

    # Parameters for simple galaxy SAM
    "pop_sam_method": 0,
    "pop_sam_nz": 1,
    "pop_mass_yield": 0.5,
    "pop_metal_yield": 0.1,

    "pop_gas_fraction": None,

    "pop_mzr": None,
    "pop_fox": 0.03,

    "pop_msr": None,

    "pop_dust_holes": 'big',
    "pop_dust_yield": None,     # Mdust = dust_yield * metal mass
    "pop_dust_yield_delay": 0,
    "pop_dust_growth": None,
    "pop_dust_scale": 0.1,    # 100 pc
    "pop_dust_fcov": 1.0,
    "pop_dust_geom": 'screen',  # or 'mixed'


    "pop_dust_scatter": None,
    "pop_dust_scatter_seed": None,
    "pop_dust_kill_redshift": np.inf,

    "pop_Av": None,
    'pop_dust_template': None,
    "pop_dust_absorption_coeff": None,   # opacity in [cm^2 / g]



    "pop_fpoll": 1.0,         # uniform pollution
    "pop_fstall": 0.0,
    "pop_mass_rec": 0.0,
    "pop_mass_loss": 0.0,
    "pop_fstar_res": 0.0,
    "pop_metal_loss": 0.0,

    # Transition mass
    "pop_transition": 0,

    "pop_dE": None,

    "pop_calib_wave": 1600,
    "pop_calib_lum": None,
    "pop_lum_per_sfr": None,

    "pop_calib_Z": None,        # not implemented

    "pop_Lh_scatter": 0.0,

    'pop_fXh': None,

    'pop_frep': 1.0,
    'pop_reproc': False,

    'pop_frec_bar': 0.0,   # Neglect injected photons by default if we're
                           # treating background in approximate way

    # Nebular emission stuff
    "pop_nebular_Tgas": 2e4,
    "pop_nebular_caseBdeparture": 1.,

    "pop_lmin": None,
    "pop_lmax": None,
    "pop_dlam": None,
    "pop_wavelengths": None,
    "pop_times": None,

    "pop_toysps_method": 0,
    "pop_toysps_beta": -2.,
    "pop_toysps_norm": 2e33,    # at 1600A
    "pop_toysps_gamma": -0.8,
    "pop_toysps_delta": -0.25,
    "pop_toysps_alpha": 8.,
    "pop_toysps_t0": 100.,
    "pop_toysps_lmin": 912.,
    "pop_toysps_trise": 3,

    "pop_solve_rte": False,
    "pop_lya_permeable": False,

    # Pre-created splines
    "pop_fcoll": None,
    "pop_dfcolldz": None,

    # Get passed on to litdata instances
    "source_kwargs": {},
    "pop_kwargs": {},

    "pop_test_param": None,


    #HOD model - set for z = 0, no quiescent galaxies
    # 'pop_sf_fract': 1.0,
    'pop_lf': 3e-4,
    'pop_smhm_beta': 1.06,
    'pop_smhm_n': 0.0282,
    'pop_smhm_gamma': 0.556,
    'pop_smhm_m': 11.88,
    'pop_sfr_1': 0.84,
    'pop_sfr_2': 6.51,

    #HOD model - set for z = 0, some quiescent galaxies
    'pop_sf_A': -1,
    'pop_sf_B': -10.5,
    'pop_sf_C': 3.0,
    'pop_sf_D': 2.0,

    }

    pf.update(tmp)
    pf.update(rcParams)

    return pf

def SourceParameters():
    pf = {
        "source_type": 'star',
        "source_sed": 'bb',
        "source_position": 0.0,


        "source_sed_sharp_at": None,
        "source_sed_null_except": None,
        "source_sed_degrade": None,

        "source_sfr": 1.0,
        "source_fesc": 0.1,

        # only for schaerer2002 right now
        "source_piecewise": True,
        "source_model": 'tavg_nms', # or "zams" or None

        "source_tbirth": 0,
        "source_lifetime": 1e10,

        "source_dlogN": [0.1],
        "source_logNmin": [None],
        "source_logNmax": [None],
        "source_table": None,

        "source_E": None,
        "source_LE": None,

        "source_multigroup": False,

        "source_Emin": E_LL,
        "source_Emax": 1e2,
        "source_Enorm": None,
        "source_EminNorm": None,
        "source_EmaxNorm": None,
        "source_dE": None,

        "source_dlam": None,
        "source_lmin": None,
        "source_lmax": None,
        "source_wavelengths": None,
        "source_times": None,

        "source_toysps_method": 0,
        "source_toysps_beta": -2.5,
        "source_toysps_norm": 3e33,  # at 1600A
        "source_toysps_gamma": -1.0,
        "source_toysps_delta": -0.25,
        "source_toysps_alpha":8.0,
        "source_toysps_t0": 350.0,
        "source_toysps_lmin": 912.0,
        "source_toysps_trise": 3,

        "source_Ekill": None,

        "source_logN": -inf,
        "source_hardening": 'extrinsic',

        # Synthesis models
        "source_Z": 0.02,
        "source_imf": 2.35,
        "source_imf_Mmax": 300,
        "source_tracks": None,
        "source_tracks_fn": None,
        "source_stellar_aging": False,
        "source_nebular": False,
        "source_nebular_only": False,
        "source_nebular_continuum": True,
        "source_nebular_lines": True,
        "source_nebular_ff": True,
        "source_nebular_fb": True,
        "source_nebular_2phot": True,
        "source_nebular_lookup": None,
        "source_nebular_Tgas": 2e4,
        "source_nebular_caseBdeparture": 1.0,
        "source_prof_1h": None,
        "source_ssp": False,             # a.k.a., continuous SF
        "source_sfh": 'const',
        "source_sfh_axes": None,

        "source_psm_instance": None,
        "source_age": 100.0,
        "source_tneb": None,
        "source_binaries": False,        # for BPASS
        "source_sed_by_Z": None,
        "source_rad_yield": 'from_sed',
        "source_interpolant": None,

        "source_sps_data": None,

        # Log masses
        "source_imf_bins": np.arange(-1, 2.52, 0.02),  # bin centers

        "source_degradation": None,      # Degrade spectra to this \AA resolution
        "source_aging": False,

        # Stellar
        "source_temperature": 1e5,
        "source_qdot": 5e48,

        # SFH
        "source_sfh": None,
        "source_sfh_axes": None,
        "source_meh": None,

        # BH
        "source_mass": 1,         # Also normalizes ssp's, so set to 1 by default.
        "source_rmax": 1e3,
        "source_alpha": -1.5,

        "source_evolving": False,

        # SIMPL
        "source_fsc": 0.1,
        "source_uponly": True,
        "source_dlogE": 0.1,

        "source_Lbol": None,
        "source_fduty": 1.0,

        "source_eta": 0.1,
        "source_isco": 6,
        "source_rmax": 1e3,
    }

    pf.update(rcParams)

    return pf

def StellarParameters():
    pf = {
        "source_temperature": 1e5,
        "source_qdot": 5e48,
    }

    pf.update(SourceParameters())
    pf.update(rcParams)

    return pf

def BlackHoleParameters():
    pf = {
        #"source_mass": 1e5,
        "source_rmax": 1e3,
        "source_alpha": -1.5,

        "source_fsc": 0.1,
        "source_uponly": True,

        "source_Lbol": None,
        "source_mass": 10,
        "source_fduty": 1.0,

        "source_eta": 0.1,
        "source_isco": 6,
        "source_rmax": 1e3,
    }

    pf.update(SourceParameters())
    pf.update(rcParams)

    return pf

def SynthesisParameters():
    pf = \
    {
        # For synthesis models
        "source_sed": None,
        "source_sfh": 'constant',
        "source_sfh_axes": None,
        "source_sfh_degrade": 1,
        "source_sfh_fallback": None,
        "source_sed_degrade": None,
        "source_sed_null_except": None,
        "source_Z": 0.02,
        "source_imf": 2.35,
        "source_tracks": None,
        "source_tracks_fn": None,
        "source_stellar_aging": False,
        "source_nebular": False,
        "source_nebular_only": False,
        "source_nebular_continuum": False,
        "source_nebular_lines": False,

        # If doing nebular emission with ARES
        "source_nebular_ff": True,
        "source_nebular_fb": True,
        "source_nebular_2phot": True,
        "source_nebular_lookup": None,
        "source_nebular_Tgas": 2e4,
        "source_nebular_caseBdeparture": 1.0,

        "source_fesc": 0.0,

        "source_ssp": False,             # a.k.a., continuous SF
        "source_sfh": 'const',
        "source_sfh_axes": None,
        "source_psm_instance": None,
        "source_age": 100.0,
        "source_aging": False,
        "source_tneb": None,
        "source_binaries": False,        # for BPASS
        "source_sed_by_Z": None,
        "source_rad_yield": 'from_sed',
        "source_sps_data": None,

        # Only used by toy SPS
        "source_dE": None,
        "source_Emin": 1.0,
        "source_Emax": 54.4,
        "source_EminNorm": 1.0,
        "source_EmaxNorm": 54.4,

        "source_lifetime": 1e10,
        "source_qdot": 5e48,
        "source_temperature": 1e5,

        "source_dlam": None,
        "source_lmin": None,
        "source_lmax": None,
        "source_times": None,
        "source_wavelengths": None,

        "source_mass": 1.0,

        "source_toysps_method": 0,
        "source_toysps_beta": -2.0,
        "source_toysps_norm": 2e33,  # at 1600A
        "source_toysps_gamma": -0.8,
        "source_toysps_delta": -0.25,
        "source_toysps_alpha": 8.0,
        "source_toysps_t0": 100.0,
        "source_toysps_lmin": 912.0,
        "source_toysps_trise": 3,

        # Coefficient of Bpass in Hybrid synthesis model
        "source_coef": 0.5,
    }

    return pf

def HaloMassFunctionParameters():
    pf = {
        "halo_mf": 'ST',
        "halo_mf_sub": "Tinker08",

        "halo_mf_instance": None,
        "halo_mf_load": True,
        "halo_mf_cache": None,
        "halo_ps_load": True,
        "halo_load_growth": False,
        "halo_use_splined_growth": True,
        "halo_mf_table": None,
        "halo_mf_analytic": False,
        "halo_mf_params": None,

        # Table resolution
        "halo_logMmin": 4,
        "halo_logMmax": 18,
        "halo_dlogM": 0.01,
        "halo_zmin": 0,
        "halo_zmax": 60,
        "halo_dz": 0.05,

        # Optional: time instead of redshift
        "halo_tmin": 30.0,
        "halo_tmax": 1000.0,
        "halo_dt": None,     # if not None, will switch this one.

        # Augment suite of halo growth histories
        "halo_hist_dlogM": 0.1,
        'halo_hist_Mmax': 10, # 10x the

        # to CAMB
        'halo_dlna': 2e-6,           # hmf default value is 1e-2
        'halo_dlnk': 1e-2,
        'halo_lnk_min': -20.0,
        'halo_lnk_max': 10.0,
        'halo_transfer_k_per_logint': 11,
        'halo_transfer_kmax': 100.0, # hmf default value is 5

        "halo_dfcolldz_smooth": False,
        "halo_dfcolldz_trunc": False,

        "halo_mf_path": None,

        # For, e.g., fcoll, etc
        "halo_interp": 'cubic',

        "halo_mf_func": None,
        "halo_extra_par0": None,
        "halo_extra_par1": None,
        "halo_extra_par2": None,
        "halo_extra_par3": None,
        "halo_extra_par4": None,

        # Mean molecular weight of collapsing gas
        "mu": 0.61,

        "halo_database": None,

        "halo_ps_linear": True,

        'halo_dlnk': 0.001,
        'halo_dlnR': 0.001,
        'halo_lnk_min': -9.0,
        'halo_lnk_max': 9.0,
        'halo_lnR_min': -9.0,
        'halo_lnR_max': 9.0,

        # Note that this is not passed to hmf yet.
        "halo_mf_window": 'tophat',
        "halo_wdm_mass": None,
        "halo_wdm_interp": True,

        #For various DM models
        'halo_dm_model': 'CDM',

        "halo_cosmology_location": None,
        # PCA eigenvectors
        "halo_mf_pca": None,
        "halo_mf_pca_coef0":None,
        "halo_mf_pca_coef1":None,
        "halo_mf_pca_coef2":None,
        "halo_mf_pca_coef3":None,
        "halo_mf_pca_coef4":None,
        "halo_mf_pca_coef5":None,
        "halo_mf_pca_coef6":None,
        "halo_mf_pca_coef7":None,
        "halo_mf_pca_coef8":None,
        "halo_mf_pca_coef9":None,
        "halo_mf_pca_coef10": None,
        "halo_mf_pca_coef11": None,
        "halo_mf_pca_coef12": None,
        "halo_mf_pca_coef13": None,
        "halo_mf_pca_coef14": None,
        "halo_mf_pca_coef15": None,
        "halo_mf_pca_coef16": None,
        "halo_mf_pca_coef17": None,
        "halo_mf_pca_coef18": None,
        "halo_mf_pca_coef19": None,

        # If a new tab_MAR should be computed when using the PCA
        "halo_mf_gen_MAR":False,

        "filter_params" : None,

        "halo_MAR_from_CDM": True,
    }

    pf.update(rcParams)

    return pf

def CosmologyParameters():
    # Last column of Table 4 in Planck XIII. Cosmological Parameters (2015)
    pf = {
        "cosmology_propagation": False,
        "cosmology_inits_location": None,
        "omega_m_0": 0.3089,
        "omega_b_0": round(0.0223 / 0.6774**2, 5),  # O_b / h**2
        "omega_l_0": 1.0 - 0.3089,
        "omega_k_0": 0.0,
        "hubble_0": 0.6774,
        "helium_by_number": 0.0813,
        "helium_by_mass": 0.2453,   # predicted by BBN
        "cmb_temp_0": 2.7255,
        "sigma_8": 0.8159,
        "primordial_index": 0.9667,
        'relativistic_species': 3.04,
        "approx_highz": False,
        "cosmology_id": 'best',
        "cosmology_name": 'planck_TTTEEE_lowl_lowE',  # Can pass 'named cosmologies'
        "cosmology_number": None,
        "path_to_CosmoRec": None,
        "interpolate_cosmology_in_z": False,

        # As you might have guessed, these parameters are all unique to CosmoRec
        'cosmorec_nz': 1000,
        'cosmorec_z0': 3000,
        'cosmorec_zf': 0,
        'cosmorec_recfast_fudge': 1.14,
        'cosmorec_nshells_H': 3,
        'cosmorec_nS': 500,
        'cosmorec_dm_annhil': 0,
        'cosmorec_A2s1s': 0,                   # will use internal default if zero
        'cosmorec_nshells_He': 3,
        'cosmorec_HI_abs': 2,                  # during He recombination
        'cosmorec_spin_forb': 1,
        'cosmorec_feedback_He': 0,
        'cosmorec_run_pde': 1,
        'cosmorec_corr_2s1s': 2,
        'cosmorec_2phot': 3,
        'cosmorec_raman': 2,
        'cosmorec_path': None,
        'cosmorec_output': 'input/inits/outputs/',
        'cosmorec_fmt': '.dat',
    }

    pf.update(rcParams)

    return pf

def HaloParameters():
    # Last column of Table 4 in Planck XIII. Cosmological Parameters (2015)
    pf = {
        "halo_profile": 'nfw',
        "halo_cmr": 'duffy',
        "halo_delta": 200.0,
    }

    pf.update(rcParams)

    return pf

def ControlParameters():
    pf = {
        'revision': None,

        'nthreads': None,

        # Start/stop/IO
        "dtDataDump": 1.0,
        "dzDataDump": None,
        'logdtDataDump': None,
        'logdzDataDump': None,
        "stop_time": 500,

        "initial_redshift": 60.0,
        "final_redshift": 5,
        "fallback_dz": 0.1, # only used when no other constraints
        "kill_redshift": 0.0,
        "first_light_redshift": 60.0,

        "save_rate_coefficients": 1,

        "optically_thin": 0,

        # Solvers
        "solver_rtol": 1e-8,
        "solver_atol": 1e-8,
        "interp_tab": 'cubic',
        "interp_cc": 'linear',
        "interp_rc": 'linear',
        "interp_Z": 'cubic',
        "interp_hist": 'linear',
        "interp_all": 'linear',  # backup
        #"interp_sfrd": 'cubic',
        #"interp_hmf": 'cubic',
        "master_interp": None,

        # Not implemented
        "extrap_Z": False,

        # Experimental
        "conserve_memory": False,

        # Initialization
        "load_ics": 'cosmorec',
        "cosmological_ics": False,
        "load_sim": False,

        "cosmological_Mmin": ['filtering', 'tegmark'],

        # Timestepping
        "max_timestep": 1.0,
        "epsilon_dt": 0.05,
        "initial_timestep": 1e-2,
        "tau_ifront": 0.5,
        "restricted_timestep": ['ions', 'neutrals', 'electrons', 'temperature'],

        "compute_fluxes_at_start": False,

        # Real-time analysis junk
        "stop": None,           # 'B', 'C', 'trans', or 'D'

        "stop_igm_h_2": 0.999,
        "stop_cgm_h_2": 0.999,

        "track_extrema": False,
        "delay_extrema": 5,      # Number of steps
        "delay_tracking": 1.0,    # dz below initial_redshift when tracking begins
        "smooth_derivative": 0,

        "blob_names": None,
        "blob_ivars": None,
        "blob_funcs": None,
        "blob_kwargs": {},

        # Real-time optical depth calculation once EoR begins
        "EoR_xavg": 1.0,        # ionized fraction indicating start of EoR (OFF by default)
        "EoR_dlogx": 0.001,
        "EoR_approx_tau": False, # 0 = trapezoidal integration,
                                 # 1 = mean ionized fraction, approx cross sections
                                 # 2 = neutral approx, approx cross sections

        # Discretizing integration
        "tau_table": None,
        "tau_arrays": None,
        "tau_prefix": None,
        "tau_instance": None,
        "tau_redshift_bins": 400,
        "tau_approx": True,
        "tau_clumpy": None,
        "tau_Emin": 2e2,
        "tau_Emax": 3e4,
        "tau_Emin_pin": True,

        "sam_dt": 1.0, # Myr
        "sam_dz": None, # Usually good enough!
        "sam_atol": 1e-4,
        "sam_rtol": 1e-4,

        # File format
        "preferred_format": 'hdf5',

        # Finding SED tables
        "load_sed": False,
        "sed_prefix": None,

        "unsampled_integrator": 'quad',
        "sampled_integrator": 'simps',
        "integrator_rtol": 1e-6,
        "integrator_atol": 1e-4,
        "integrator_divmax": 1e2,

        "interpolator": 'spline',

        "progress_bar": True,
        "verbose": True,
        "debug": False,

        'use_mcfit': True,
    }

    pf.update(rcParams)

    return pf

_sampling_parameters = {
    'parametric_model': False,
    'output_frequencies': None,
    'output_freq_min': 30.0,
    'output_freq_max': 200.0,
    'output_freq_res': 1.0,
    'output_dz': None,  # Redshift sampling
    'output_redshifts': None,
}

# Old != Deprecated
def OldParameters():
    pf = {
        'xi_LW': None,
        'xi_UV': None,
        'xi_XR': None,
    }

    return pf

def TanhParameters():
    pf = {
        'tanh_model': False,
        'tanh_J0': 10.0,
        'tanh_Jz0': 20.0,
        'tanh_Jdz': 3.0,
        'tanh_T0': 1e3,
        'tanh_Tz0': 8.0,
        'tanh_Tdz': 4.0,
        'tanh_x0': 1.0,
        'tanh_xz0': 10.0,
        'tanh_xdz': 2.0,
        'tanh_bias_temp': 0.0,   # in mK
        'tanh_bias_freq': 0.0,   # in MHz
        'tanh_scale_temp': 1.0,
        'tanh_scale_freq': 1.0
    }

    pf.update(rcParams)
    pf.update(_sampling_parameters)

    return pf

def GaussianParameters():
    pf = {
        'gaussian_model': False,
        'gaussian_A': -100.0,
        'gaussian_nu': 70.0,
        'gaussian_sigma': 10.0,
        'gaussian_bias_temp': 0,
    }

    pf.update(rcParams)
    pf.update(_sampling_parameters)

    return pf
