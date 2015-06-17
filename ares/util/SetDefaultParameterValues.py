"""
SetDefaultParameterValues.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-10-19.

Description: Defaults for all different kinds of parameters, now sorted
into groups.
     
"""

import os, imp
from numpy import inf
from ares import rcParams
from ..physics.Constants import m_H, cm_per_kpc, s_per_myr

ARES = os.environ.get('ARES')
    
tau_prefix = "%s/input/optical_depth" % ARES \
    if (ARES is not None) else '.'
    
pgroups = ['Grid', 'Physics', 'Cosmology', 'Source', 'Population', 
    'Control', 'HaloMassFunction', 'Tanh', 'Slab']

# Blob stuff
_blob_redshifts = list('BCD')
_blob_redshifts.extend([6, 10, 20, 30, 40])

# Nothing population specific
_blob_names = ['z', 'igm_dTb', 'curvature', 'igm_Tk', 'igm_Ts', 'cgm_h_2', 
    'igm_h_1']

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
    
    "include_cgm": True,
    
    # Line photons
    "include_H_Lya": False,

    "initial_ionization": [1. - 1e-8, 1e-8],
    "initial_temperature": 1e4,
            
    # These have shape len(absorbers)
    "tables_logNmin": [None],
    "tables_logNmax": [None],
    "tables_dlogN": [0.1],        
    "tables_xmin": [1e-8],
    #
    
    "tables_discrete_gen": False,
    "tables_energy_bins": 100,
    "tables_prefix": None,
    
    ""
    
    "tables_logxmin": -4,
    "tables_dlogx": 0.1,
    "tables_dE": 5.,
    
    "tables_times": None,
    "tables_dt": s_per_myr,
            
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

def AbsorberParameters():
    pf = \
    {
    'cddf_C': 0.25,
    'cddf_beta': 1.4,
    'cddf_gamma': 1.5,
    'cdff_zlow': 1.5,
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

    # Approximations to radiation fields
    #"approx_irb": True,
    #"approx_lwb": True,
    #"approx_uvb": True,
    #'approx_xrb': True,
        
    # If doing "full" calculation, discretize in redshift and energy?
    #"discrete_irb": True,
    #"discrete_lwb": True,
    #"discrete_uvb": True,
    #'discrete_xrb': True,
    #
    #"tau_irb": False,
    #"tau_lwb": False,
    #"tau_uvb": False,
    #"tau_xrb": False,

    "tau_dynamic": False,
    
    # How many redshift bins for static optical depth tables
    #"redshifts_irb": 400,
    #"redshifts_lwb": 1e4,
    #"redshifts_uvb": 400,
    #"redshifts_xrb": 400,

    "sawtooth_nmax": 8,

    # Lyman alpha sources

    "lya_nmax": 23,
    "lya_injected": True,
    'lya_continuum': True,
    'lya_frec_bar': 0.0,   # Neglect injected photons by default if we're
                           # treating background in approximate way
                     
    "rate_source": 'fk94', # fk94, option for development here
    
    # Approximations to radiation field                   
    "norm_by": 'xray',
    "xray_Emin": 2e2,
    
    # Feedback!
    "feedback": False,
    "feedback_dz": 0.5,
    "feedback_method": ['jeans_mass', 'filtering_mass', 'critical_Jlw'],
    "feedback_analytic": True,
    
    }
    
    pf.update(rcParams)
            
    return pf
    
def PopulationParameters():
    
    pf = {}
    srcpars = SourceParameters()
    for par in srcpars:
        pf[par.replace('source', 'pop')] = srcpars[par]
    
    
    tmp = \
    {
    
    "pop_type": 'galaxy',
    
    "pop_rhoL": None,
    
    "pop_sed": 'pl',
    
    "pop_sawtooth": False,
    "pop_solve_rte": False,
    "pop_tau_Nz": 400,
    
    "pop_Emin": 2e2,
    "pop_Emax": 3e4,
    "pop_EminNorm": 5e2,
    "pop_EmaxNorm": 8e3,
    
    "pop_lf": None,
    "pop_emissivity": None,
    
    "pop_lf_Lstar": 1e42,
    "pop_lf_slope": -1.5,
    
    "pop_lf_zdep": None,
    
    "pop_lf_Lmin": 1e38,
    "pop_lf_Lmax": 1e42,
    "pop_lf_LminNorm": 1e38,
    "pop_lf_LmaxNorm": 1e42,    
    
    
    
    "pop_zform": 50.,
    "pop_zdead": 0.0,
    
    "pop_focc": None,
    
    # Main parameters in our typical global 21-cm models
    "pop_fstar": 0.1,
    "pop_Tmin": 1e4,
    "pop_Mmin": None,
    "pop_sfrd": None,
    
    # Parameters that sweep fstar under the rug
    "pop_xi_XR": None,     # product of fstar and fX
    "pop_xi_LW": None,     # product of fstar and Nlw
    "pop_xi_UV": None,     # product of fstar, Nion, and fesc
    
    # For multi-frequency calculations
    "pop_E": None,
    "pop_LE": None,

    # What radiation does this population emit?
    "pop_lya_src": True,
    "pop_ion_src_cgm": True,
    "pop_ion_src_igm": True,
    "pop_heat_src_cgm": False,
    "pop_heat_src_igm": True,

    "pop_src_irb": False,
    "pop_src_lwb": True,
    "pop_src_uvb": False,
    "pop_src_xrb": True,

    # Generalized normalization
    "pop_yield": 2.6e39,
    "pop_yield_units": 'erg/s/SFR', # alternatively, 'photons / baryon'

    # Scales X-ray emission
    "pop_cX": 3.4e40, # Furlanetto (2006) extrapolation
    "pop_fX": 0.2,    # Mineo et al. (2012) (from revised 0.5-8 keV L_X-SFR)
    'pop_fXh': None,

    # Scales Lyman-Werner emission
    "pop_Nlw": 9690.,

    # Scales ionizing emission
    "pop_fion": 1.0,                
    "pop_Nion": 4e3,
    "pop_fesc": 0.1,

    # Controls IGM ionization for approximate CXB treatments
    "pop_Ex": 500.,
    "pop_Euv": 30.,

    # Bypass fcoll prescriptions, use parameterizations
    "heat_igm": None,
    "Gamma_igm": None,
    "Gamma_cgm": None,
    "gamma_igm": None,
    'Ja': None,
    
    # Pre-created splines
    "pop_fcoll": None,
    "pop_dfcolldz": None,
    
    # Get passed on to litdata instances
    "source_kwargs": {},
    "pop_kwargs": {},

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
    
    pf.update(SourceParameters())
    pf.update(rcParams)
    
    return pf
        
def HaloMassFunctionParameters():
    pf = \
    {
    "hmf_func": 'ST',
    
    "hmf_load": True,
    "hmf_table": None,
    "hmf_analytic": False,
    
    # Table resolution
    "hmf_logMmin": 4,
    "hmf_logMmax": 16,
    "hmf_dlogM": 0.01,
    "hmf_zmin": 4,
    "hmf_zmax": 60,
    "hmf_dz": 0.05,
    
    # Mean molecular weight of collapsing gas
    "mu": 0.61,
    
    # Compute the full mass function? 
    "hmf_dndm": False,
    
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
    "final_redshift": 6,
    "save_rate_coefficients": 1,

    "optically_thin": 0,

    # Solvers
    "solver_rtol": 1e-8,
    "solver_atol": 1e-8,
    "interp_method": 'cubic',

    # Initialization
    "load_ics": False,
    "cosmological_ics": False,
    "load_sim": False,
    #"first_light_redshift": 50.,

    # Timestepping
    "max_dt": 1.,
    "max_dz": None,
    "max_timestep": 1.,
    "epsilon_dt": 0.05,
    "initial_timestep": 1e-2,
    "tau_ifront": 0.5,
    "restricted_timestep": ['ions', 'neutrals', 'electrons', 'temperature'],
    
    # Real-time analysis junk
    "stop": None,           # 'B', 'C', 'trans', or 'D'
    "stop_xavg": 0.99999,   # stop at given ionized fraction
    "track_extrema": False,
    "stop_delay": 0.5,      # differential redshift step
    "inline_analysis": None,
    "auto_generate_blobs": False,
    "override_blob_names": None,
    "override_blob_redshifts": None,

    # Real-time optical depth calculation once EoR begins
    "EoR_xavg": 1.0,        # ionized fraction indicating start of EoR (OFF by default)
    "EoR_dlogx": 0.001,
    "EoR_approx_tau": False, # 0 = trapezoidal integration,
                             # 1 = mean ionized fraction, approx cross sections
                             # 2 = neutral approx, approx cross sections

    # Discretizing integration
    "redshift_bins": None,
    "tau_table": None,
    "tau_prefix": tau_prefix,

    "approx_tau": None,

    # File format
    "preferred_format": 'pkl',

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
    }

    pf.update(rcParams)

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
    'tanh_dz': 0.1,  # Redshift sampling
    'tanh_bias_temp': 0.0,   # in mK
    'tanh_bias_freq': 0.0,   # in MHz
    'tanh_nu': None, # Array of frequencies in MHz
    }
    
    pf.update(rcParams)

    return pf
    

    


