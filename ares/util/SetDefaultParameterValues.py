"""
SetDefaultParameterValues.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-10-19.

Description: Defaults for all different kinds of parameters.
     
"""

import os, imp
import numpy as np
from ares import rcParams
from ..physics.Constants import m_H, cm_per_kpc, s_per_myr

inf = np.inf
ARES = os.environ.get('ARES')
    
tau_prefix = os.path.join(ARES,'input','optical_depth') \
    if (ARES is not None) else '.'
    
pgroups = ['Grid', 'Physics', 'Cosmology', 'Source', 'Population', 
    'Control', 'HaloMassFunction', 'Tanh', 'Gaussian', 'Slab',
    'MultiPhase', 'Dust', 'ParameterizedQuantity', 'Old']

# Blob stuff
_blob_redshifts = list('BCD')
_blob_redshifts.extend([6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40])

# Nothing population specific
_blob_names = ['z', 'dTb', 'curvature', 'igm_Tk', 'igm_Ts', 'cgm_h_2', 
    'igm_h_1', 'cgm_k_ion', 'igm_k_heat', 'Ja', 'tau_e']
    
default_blobs = (_blob_names, _blob_names)

# Start setting up list of parameters to be set
defaults = []

for grp in pgroups:
    defaults.append('%sParameters()' % grp)

def SetAllDefaults():
    pf = {'problem_type': 1}
    
    for pset in defaults:
        exec('pf.update(%s)' % pset)
        
    return pf
    
def GridParameters():
    pf = \
    {
    "grid_cells": 64,
    "start_radius": 0.01,
    "logarithmic_grid": False,
    
    "density_units": 1e-3,            # H number density
    "length_units": 10. * cm_per_kpc,  
    "time_units": s_per_myr,  
    
    "include_He": False,
    "include_H2": False,
    
    # For MultiPhaseMedium calculations
    "include_cgm": True,
    "include_igm": True,
    
    # Line photons
    "include_injected_lya": True,

    "initial_ionization": [1. - 1e-8, 1e-8, 1.-2e-8, 1e-8, 1e-8],
    "initial_temperature": 1e4,
            
    # These have shape len(absorbers)
    "tables_logNmin": [None],
    "tables_logNmax": [None],
    "tables_dlogN": [0.1],   
    
    # overrides above parameters
    "tables_logN": None, 
    
    "tables_xmin": [1e-8],
    #
    
    "tables_discrete_gen": False,
    "tables_energy_bins": 100,
    "tables_prefix": None,
        
    "tables_logxmin": -4,
    "tables_dlogx": 0.1,
    "tables_dE": 5.,
    
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
    pf = \
    {
     "cgm_grid_cells": 1,
     "cgm_expansion": True,
     "cgm_initial_temperature": [1e4],
     "cgm_initial_ionization": [1.-1e-8, 1e-8, 1.-2e-8, 1e-8, 1e-8],
     "cgm_isothermal": True,
     "cgm_recombination": 'A',
     "cgm_collisional_ionization": False,
     "cgm_cosmological_ics": False,
     
     "photon_counting": False,
     "monotonic_EoR": 1e-6,

     "igm_grid_cells": 1,     
     "igm_expansion": True,
     "igm_initial_temperature": None,         
     'igm_initial_ionization': [1.-1e-8, 1e-8, 1.-2e-8, 1e-8, 1e-8],
     "igm_isothermal": False,
     "igm_recombination": 'B',
     "igm_compton_scattering": True,
     "igm_collisional_ionization": True,
     "igm_cosmological_ics": False,
     
    }    
    
    pf.update(rcParams)
    
    return pf
    
def SlabParameters():
    
    pf = \
    {
    "slab": 0,
    "slab_position": 0.1,
    "slab_radius": 0.05,
    "slab_overdensity": 100,
    "slab_temperature": 100,
    "slab_ionization": [1. - 1e-8, 1e-8],
    "slab_profile": 0,
    }
    
    pf.update(rcParams)
    
    return pf

# BoundaryConditionParameters?
def FudgeParameters():
    pf = \
    {
    "z_heII_EoR": 3.,
    }
    
    pf.update(rcParams)
    
    return pf

def AbsorberParameters():
    pf = \
    {
    'cddf_C': 0.25,
    'cddf_beta': 1.4,
    'cddf_gamma': 1.5,
    'cddf_zlow': 1.5,
    'cddf_gamma_low': 0.2,
    }
    
    pf.update(rcParams)
    
    return pf

def PhysicsParameters():
    pf = \
    {        
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

    "clumping_factor": 1,

    "approx_H": False,
    "approx_He": False,
    "approx_sigma": False,
    "approx_Salpha": 1, # 1 = Salpha = 1
                        # 2 = Chuzhoy, Alvarez, & Shapiro (2005),
                        # 3 = Furlanetto & Pritchard (2006)

    # Lyman alpha sources
    "lya_nmax": 23,
    
    "rate_source": 'fk94', # fk94, option for development here
    
    # Feedback parameters
    'feedback_maxiter': 10,
    'feedback_rtol': 0.,
    'feedback_atol': 1.,
    'feedback_mean_err': False,
    
    # LW
    'feedback_LW_Mmin': None,        # 'visbal2014'
    'feedback_LW_fsh': None,
    'feedback_LW_felt_by': None,
    'feedback_LW_Tcut': 1e4,    
    'feedback_LW_Mmin_uponly': False,
    'feedback_LW_Mmin_smooth': False,
    
    'feedback_Z': None,
    'feedback_Z_Tcut': 1e4,
    'feedback_Z_rtol': 0.,
    'feedback_Z_atol': 1.,
    'feedback_Z_mean_err': False,
    'feedback_Z_Mmin_uponly': False,
    'feedback_Z_Mmin_smooth': False,
    
    'feedback_tau': None,
    'feedback_tau_Tcut': 1e4,
    'feedback_tau_rtol': 0.,
    'feedback_tau_atol': 1.,
    'feedback_tau_mean_err': False,
    'feedback_tau_Mmin_uponly': False,
    'feedback_tau_Mmin_smooth': False,
    
    'feedback_EoR': None,
    'feedback_EoR_Tcut': 1e4,
    'feedback_EoR_rtol': 0.,
    'feedback_EoR_atol': 1.,
    'feedback_EoR_mean_err': False,
    'feedback_EoR_Mmin_uponly': False,
    'feedback_EoR_Mmin_smooth': False,

    }

    pf.update(rcParams)

    return pf

def ParameterizedQuantityParameters():
    pf = {}

    tmp = \
    {
     "pq_func": 'dpl',
     "pq_func_var": 'Mh',
     "pq_func_par0": None,
     "pq_func_par1": None,
     "pq_func_par2": None,
     "pq_func_par3": None,
     "pq_func_par4": None,
     "pq_func_par5": None,

     'pq_faux': None,
     'pq_faux_var': None,
     'pq_faux_meth': 'multiply',
     'pq_faux_par0': None,
     'pq_faux_par1': None,
     'pq_faux_par2': None,
     'pq_faux_par3': None,
     'pq_faux_par4': None,
     'pq_faux_par5': None,
     
     'pq_faux_A': None,
     'pq_faux_A_var': None,
     'pq_faux_A_meth': 'multiply',
     'pq_faux_A_par0': None,
     'pq_faux_A_par1': None,
     'pq_faux_A_par2': None,
     'pq_faux_A_par3': None,
     'pq_faux_A_par4': None,
     'pq_faux_A_par5': None,
     
     'pq_faux_B': None,
     'pq_faux_B_var': None,
     'pq_faux_B_meth': 'multiply',
     'pq_faux_B_par0': None,
     'pq_faux_B_par1': None,
     'pq_faux_B_par2': None,
     'pq_faux_B_par3': None,
     'pq_faux_B_par4': None,
     'pq_faux_B_par5': None,
     
     "pq_boost": 1.,
     "pq_iboost": 1.,
     "pq_val_ceil": None,
     "pq_val_floor": None,
     "pq_var_ceil": None,
     "pq_var_floor": None,      
         
    }  
    
    # Hrm...can't remember what this is about.
    for i in range(6):
        for j in range(6):
            tmp['pq_func_par%i_par%i' % (i,j)] = None
    
    pf.update(tmp)
    pf.update(rcParams)

    return pf

def DustParameters():
    pf = {}
    
    tmp = \
    {     
     'dustcorr_method': None,

     'dustcorr_beta': -2.,
     
     # Only used if method is a list
     'dustcorr_ztrans': None,
     
     # Intrinsic scatter in the AUV-beta relation
     'dustcorr_scatter_A': 0.0,
     # Intrinsic scatter in the beta-mag relation (gaussian)
     'dustcorr_scatter_B': 0.34,
          
     'dustcorr_Bfun_par0': -2.,
     'dustcorr_Bfun_par1': None,
     'dustcorr_Bfun_par2': None,

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
    
    tmp = \
    {
    
    "pop_type": 'galaxy',
    
    "pop_tunnel": None,

    "pop_sfr_model": 'fcoll', # or sfrd-func, sfrd-tab, sfe-func, sfh-tab, rates,
    "pop_sed_model": True,    # or False
    
    "pop_sfr_above_threshold": True,
    "pop_sfr_cross_threshold": True,
    "pop_sfr_cross_upto_Tmin": np.inf,
    
    # Mass accretion rate
    "pop_MAR": 'hmf',
    "pop_MAR_conserve_norm": False,

    "pop_tdyn": 1e7,
    "pop_sSFR": None,

    # the pop_lf pars here might be deprecated
    "pop_lf_z": None,
    "pop_lf_M": None,
    "pop_lf_Mstar": None,
    "pop_lf_pstar": None,
    "pop_lf_alpha": None,
    "pop_lf_mags": None,

    'pop_lf_Mmax': 1e15,

    "pop_fduty": 1.0,
    "pop_focc": 1.0,
    
    "pop_fsup": 0.0,  # Suppression of star-formation at threshold
        
    # Set the emission interval and SED
    "pop_sed": 'pl',
    
    # If pop_sed == 'user'
    "pop_E": None,
    "pop_L": None,
    
    # For synthesis models
    "pop_Z": 0.02,
    "pop_imf": 2.35,
    "pop_nebular": False,
    "pop_ssp": False,             # a.k.a., continuous SF
    "pop_psm_instance": None,
    "pop_tsf": 100.,
    "pop_binaries": False,        # for BPASS
    "pop_sed_by_Z": None,

    # Option of setting Z, t, or just supplying SSP table?
    
    "pop_Emin": 2e2,
    "pop_Emax": 3e4,
    "pop_EminNorm": 5e2,
    "pop_EmaxNorm": 8e3,

    "pop_Emin_xray": 2e2,
    
    # Controls IGM ionization for approximate CXB treatments
    "pop_Ex": 500.,
    "pop_EminX": 2e2,
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
    "pop_zform": 50.,
    "pop_zdead": 0.0,

    # Main parameters in our typical global 21-cm models
    "pop_fstar": 0.1,
    "pop_fstar_max": 1.0,
    "pop_fstar_negligible": 1e-5, # relative to maximum
    
    "pop_sfe": None,
    "pop_mlf": None,
    "pop_sfr": None,
    "pop_fshock": 1.0,
    
    "pop_fobsc": 0.0,
    "pop_fgrowth": 1.0,
    
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
    "pop_Mmax": None,
    "pop_sfrd": None,
    "pop_sfrd_units": 'msun/yr/mpc^3',
    
    # Scales SFRD
    "pop_Nlw": 9690.,
    "pop_Nion": 4e3,
    "pop_fesc": 0.1,
    "pop_fX": 1.0,
    "pop_cX": 2.6e39,
    
    # Should
    "pop_fesc_LW": 1.,
    "pop_fesc_LyC": 0.1,

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
    # Should these be deprecated?
    "pop_lya_src": True,
    "pop_ion_src_cgm": True,
    "pop_ion_src_igm": True,
    "pop_heat_src_cgm": False,
    "pop_heat_src_igm": True,
    
    # Generalized normalization    
    # Mineo et al. (2012) (from revised 0.5-8 keV L_X-SFR)
    "pop_rad_yield": 2.6e39,
    "pop_rad_yield_units": 'erg/s/sfr',
    "pop_rad_yield_Z_index": None,
    
    # Parameters for simple galaxy SAM
    "pop_sam_nz": 1,
    "pop_sam_dz": 1.0,
    "pop_mass_yield": 0.5,
    "pop_metal_yield": 0.01,
    "pop_fpoll": 1.0,         # uniform pollution
    "pop_fstall": 0.0,
    "pop_mass_rec": 0.0,
    "pop_mass_escape": 0.0,
    "pop_fstar_res": 0.0,
    
    # Transition mass
    "pop_transition": 0,
    
    # deprecated?
    "pop_kappa_UV": 1.15e-28,
    
    "pop_L1600_per_sfr": None,
    "pop_calib_L1600": None,
    
    "pop_Lh_scatter": 0.0,
    
    'pop_fXh': None,
    
    'pop_frec_bar': 0.0,   # Neglect injected photons by default if we're
                           # treating background in approximate way

    "pop_approx_tau": True,     # shouldn't be a pop parameter?
    "pop_solve_rte": False,
    
    "pop_tau_Nz": 400,
    
    # Pre-created splines
    "pop_fcoll": None,
    "pop_dfcolldz": None,

    # Get passed on to litdata instances
    "source_kwargs": {},
    "pop_kwargs": {},

    "pop_test_param": None,

    }

    pf.update(tmp)
    pf.update(rcParams)

    return pf

def SourceParameters():
    pf = \
    {
    "source_type": 'star',
    "source_sed": 'bb',
    "source_position": 0.0,
    
    "source_tbirth": 0,
    "source_lifetime": 1e10,
    
    "source_dlogN": [0.1],
    "source_logNmin": [None],
    "source_logNmax": [None],
    "source_table": None,
    
    "source_E": None,
    "source_LE": None,
    "source_multigroup": False,

    "source_Emin": 13.6,  
    "source_Emax": 1e2,  
    "source_EminNorm": None,
    "source_EmaxNorm": None,
    
    "source_logN": -inf,
    "source_hardening": 'extrinsic',
    
    # Stellar
    "source_temperature": 1e5,  
    "source_qdot": 5e48,
    
    # BH
    "source_mass": 1e5,
    "source_rmax": 1e3,
    "source_alpha": -1.5,
    
    "source_fsc": 0.1,
    "source_uponly": True,
    
    "source_Lbol": None,
    "source_mass": 10,  
    "source_fduty": 1.,
    
    "source_eta": 0.1,
    "source_isco": 6,  
    "source_rmax": 1e3,

    }
    
    pf.update(rcParams)
    
    return pf

def StellarParameters():
    pf = \
    {        
    "source_temperature": 1e5,  
    "source_qdot": 5e48,
    }

    pf.update(SourceParameters())
    pf.update(rcParams)

    return pf

def BlackHoleParameters():
    pf = \
    {
    #"source_mass": 1e5,
    "source_rmax": 1e3,
    "source_alpha": -1.5,
    
    "source_fsc": 0.1,
    "source_uponly": True,
    
    "source_Lbol": None,
    "source_mass": 10,  
    "source_fduty": 1.,
    
    "source_eta": 0.1,
    "source_isco": 6,  
    "source_rmax": 1e3,
    
    }
    
    pf.update(SourceParameters())
    pf.update(rcParams)
    
    return pf    
    
def HaloMassFunctionParameters():
    pf = \
    {
    "hmf_model": 'ST',
    
    "hmf_instance": None,
    "hmf_load": True,
    "hmf_table": None,
    "hmf_analytic": False,
    
    # Table resolution
    "hmf_logMmin": 4,
    "hmf_logMmax": 16,
    "hmf_dlogM": 0.01,
    "hmf_zmin": 3,
    "hmf_zmax": 60,
    "hmf_dz": 0.05,
    
    # to CAMB
    'hmf_dlna': 2e-6,           # hmf default value is 1e-2
    'hmf_dlnk': 1e-2,
    'hmf_lnk_min': -20.,
    'hmf_lnk_max': 10.,
    'hmf_transfer__k_per_logint': 11.,
    'hmf_transfer__kmax': 100., # hmf default value is 5
    
    "hmf_dfcolldz_smooth": False,
    "hmf_dfcolldz_trunc": False,
    
    # Mean molecular weight of collapsing gas
    "mu": 0.61,
    
    }
    
    pf.update(rcParams)

    return pf

def CosmologyParameters():
    # Last column of Table 4 in Planck XIII. Cosmological Parameters (2015)
    pf = \
    {
    "omega_m_0": 0.3089,
    "omega_b_0": round(0.0223 / 0.6774**2, 5),  # O_b / h**2
    "omega_l_0": 1. - 0.3089,
    "hubble_0": 0.6774,
    "helium_by_number": 0.0813,
    "helium_by_mass": 0.2453,   # predicted by BBN
    "cmb_temp_0": 2.7255,
    "sigma_8": 0.8159,
    "primordial_index": 0.9667,
    "approx_highz": False,    
    }

    pf.update(rcParams)

    return pf
    
def ControlParameters():
    pf = \
    {
    
    # Start/stop/IO
    "dtDataDump": 1,
    "dzDataDump": None,
    'logdtDataDump': None,
    'logdzDataDump': None,
    "stop_time": 500,
    
    "initial_redshift": 50.,
    "final_redshift": 5,
    "kill_redshift": 0.0,
    
    "save_rate_coefficients": 1,
    
    "need_for_speed": False,

    "optically_thin": 0,

    # Solvers
    "solver_rtol": 1e-8,
    "solver_atol": 1e-8,
    "interp_tab": 'cubic',
    "interp_cc": 'linear',
    "interp_Z": 'linear',
    "interp_hist": 'linear',
    
    # Not implemented
    "extrap_Z": False,

    # Initialization
    "load_ics": True,
    "cosmological_ics": False,
    "load_sim": False,

    # Timestepping
    "max_dz": None,
    "max_timestep": 1.,
    "min_timestep": 1e-8,
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
    "delay_tracking": 1.,    # dz below initial_redshift when tracking begins
    "smooth_derivative": 0, 

    "blob_names": None,
    "blob_ivars": None,
    "blob_funcs": None,

    # Real-time optical depth calculation once EoR begins
    "EoR_xavg": 1.0,        # ionized fraction indicating start of EoR (OFF by default)
    "EoR_dlogx": 0.001,
    "EoR_approx_tau": False, # 0 = trapezoidal integration,
                             # 1 = mean ionized fraction, approx cross sections
                             # 2 = neutral approx, approx cross sections

    # Discretizing integration
    "tau_table": None,
    "tau_arrays": None,
    "tau_prefix": tau_prefix,
    "tau_instance": None,

    "sam_dz": 0.05,

    # File format
    "preferred_format": 'npz',

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
    }

    pf.update(rcParams)

    return pf
    
_sampling_parameters = \
{
 'parametric_model': False,
 'output_frequencies': None,
 'output_freq_min': 30.,
 'output_freq_max': 200.,
 'output_freq_res': 1.,    
 'output_dz': None,  # Redshift sampling    
}

# Old != Deprecated
def OldParameters():
    pf = \
    {
     'xi_LW': None,
     'xi_UV': None,
     'xi_XR': None,
    }
    
    return pf

def TanhParameters():
    pf = \
    {
    'tanh_model': False,
    'tanh_J0': 10.0,
    'tanh_Jz0': 20.0,
    'tanh_Jdz': 3.,
    'tanh_T0': 1e3,
    'tanh_Tz0': 8.,
    'tanh_Tdz': 4.,
    'tanh_x0': 1.0,
    'tanh_xz0': 10.,
    'tanh_xdz': 2.,
    'tanh_bias_temp': 0.0,   # in mK
    'tanh_bias_freq': 0.0,   # in MHz
    }

    pf.update(rcParams)
    pf.update(_sampling_parameters)

    return pf
    
def GaussianParameters():
    pf = \
    {
     'gaussian_model': False,
     'gaussian_A': -100., 
     'gaussian_nu': 70.,
     'gaussian_sigma': 10.,
    }
    
    pf.update(rcParams)
    pf.update(_sampling_parameters)
    
    return pf

