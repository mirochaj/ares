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
    
tau_prefix = os.path.join(ARES,'input','optical_depth') \
    if (ARES is not None) else '.'
    
pgroups = ['Grid', 'Physics', 'Cosmology', 'Source', 'Population', 'Spectrum', 
    'Control', 'HaloMassFunction', 'Tanh', 'Halo', 'LuminosityFunction',
    'Gaussian']

# Blob stuff
_blob_redshifts = list('BCD')
_blob_redshifts.extend([6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40])

# Nothing population specific
_blob_names = ['z', 'dTb', 'curvature', 'igm_Tk', 'Ts', 'cgm_h_2', 'igm_h_1',
 'tau_e', 'cgm_Gamma_h_1', 'igm_heat_h_1', 'Ja']

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

    "initial_ionization": [1.2e-3],
    "initial_temperature": 1e4,
            
    "clump": 0,
    "clump_position": 0.1,
    "clump_radius": 0.05,
    "clump_overdensity": 100,
    "clump_temperature": 100,
    "clump_ionization": 1e-6,
    "clump_profile": 0,
    
    # These have shape len(absorbers)
    "tables_logNmin": [None],
    "tables_logNmax": [None],
    "tables_dlogN": [0.1],        
    "tables_xmin": [1e-8],
    #
    
    "tables_logxmin": -4,
    "tables_dlogx": 0.1,
    "tables_dE": 5.,
    
    "tables_times": None,
    "tables_dt": s_per_myr,
            
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
    "approx_highz": False,
    "approx_Salpha": 1, # 1 = Salpha = 1
                        # 2 = Chuzhoy, Alvarez, & Shapiro (2005),
                        # 3 = Furlanetto & Pritchard (2006)
    
    # Approximations to radiation fields
    "approx_lwb": True,
    'approx_xrb': True,
    "approx_uvb": True,
    
    # If doing "full" calculation, discretize in redshift and energy?
    "discrete_lwb": True,
    'discrete_xrb': True,
    "discrete_uvb": True,

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
    }

    pf.update(rcParams)

    return pf      
    
def SourceParameters():
    pf = \
    {
    "source_type": 'star',  
    
    "source_temperature": 1e5,  
    "source_qdot": 5e48,
    "source_Lbol": None,
    "source_mass": 10,  
    "source_fduty": 1,
    "source_tbirth": 0,
    "source_lifetime": 1e10,  
    "source_eta": 0.1,
    "source_isco": 6,  
    "source_rmax": 1e3,
    "source_cX": 1.0,
    
    "source_ion": 0,
    "source_ion2": 0,
    "source_heat": 0,
    "source_lya": 0,
    
    "source_table": None,
    "source_normalized": False,
    
    }
    
    pf.update(rcParams)
    
    return pf
    
def StellarParameters():
    pf = \
    {        
    "source_temperature": 1e5,  
    "source_qdot": 5e48,
    
    "spectrum_Emin": 13.6,  
    "spectrum_Emax": 1e2,  
    "spectrum_EminNorm": None,
    "spectrum_EmaxNorm": None,
    }
    
    pf.update(rcParams)
        
    return pf

def BlackHoleParameters():
    pf = \
    {
    "source_mass": 1e5,
    "source_rmax": 1e3,
    "spectrum_alpha": -1.5,
    "spectrum_Emin": 2e2,  
    "spectrum_Emax": 3e4,  
    "spectrum_EminNorm": None,
    "spectrum_EmaxNorm": None,
    "spectrum_fsc": 0.1,
    "spectrum_uponly": True,
    }
    
    pf.update(rcParams)
    
    return pf    
        
def SpectrumParameters():
    pf = \
    {        
    "spectrum_type": 0,
    "spectrum_evolving": False,
    
    "spectrum_fraction": 1,
    
    "spectrum_alpha": -1.5,
    "spectrum_Emin": 13.6,  
    "spectrum_Emax": 1e2,  
    "spectrum_EminNorm": None,
    "spectrum_EmaxNorm": None,
    
     
    "spectrum_logN": -inf,
    "spectrum_fcol": 1.7,
    "spectrum_fsc": 0.1,
    "spectrum_uponly": True,
            
    "spectrum_file": None,
    "spectrum_pars": None,
    
    "spectrum_multigroup": 0,
    "spectrum_bands": None,
      
    "spectrum_t": None,
    "spectrum_E": None,
    "spectrum_LE": None,
            
    "spectrum_table": None,
    "spectrum_function": None,
    "spectrum_kwargs": None,
                    
    }
    
    pf.update(rcParams)
    
    return pf
    
def PopulationParameters():
    pf = \
    {
    "source_type": 'star',
    "source_kwargs": None,

    "model": -1, # Only BHs use this at this point

    "zoff": 0.0,
    "formation_redshift": 50.,
    "extinction_redshift": 0.0,
    
    "is_lya_src": True,
    "is_ion_src_cgm": True,
    "is_ion_src_igm": True,
    "is_heat_src_cgm": False,
    "is_heat_src_igm": True,
    
    # Sets star formation history
    "Tmin": 1e4, 
    "Tmax": None,
    "Mmin": None,
    "Mmax": None,
    "fstar": 0.1,
    
    # Scales X-ray emission
    "cX": 3.4e40, # Furlanetto (2006) extrapolation
    "fX": 0.2,    # Mineo et al. (2012) (from revised 0.5-8 keV L_X-SFR)
    'fXh': None,
    
    # Scales Lyman-Werner emission
    "Nlw": 9690.,
    
    # Scales ionizing emission
    "fion": 1.0,                
    "Nion": 4e3,
    "fesc": 0.1,
    
    # Acknowledge degeneracy between fstar, etc.
    "xi_XR": None,     # product of fstar and fX
    "xi_LW": None,     # product of fstar and Nlw
    "xi_UV": None,     # product of fstar, Nion, and fesc
    
    # Controls IGM ionization for approximate CXB treatments
    "xray_Eavg": 500.,
    "uv_Eavg": 30.,
    
    # Bypass fcoll prescriptions, use parameterizations
    "sfrd": None,
    "emissivity": None,
    "heat_igm": None,
    "Gamma_igm": None,
    "Gamma_cgm": None,
    "gamma_igm": None,
    'Ja': None,
    
    # Black hole models
    "rhoi": 1e2,
    "fbh": 1e-5,
    "fedd": 0.1,
    "eta": 0.1,
    "Mi": 100.,
    }
    
    pf.update(rcParams)
    
    return pf      
    
def HaloMassFunctionParameters():
    pf = \
    {
    "load_hmf": True,
    "hmf_table": None,
    "hmf_analytic": False,
    
    # Table resolution
    "hmf_logMmin": 4,
    "hmf_logMmax": 16,
    "hmf_dlogM": 0.01,
    "hmf_zmin": 4,
    "hmf_zmax": 60,
    "hmf_dz": 0.05,
    
    "hmf_dndm": False,
    
    # Mean molecular weight of collapsing gas
    "mu": 0.61,
    
    "fitting_function": 'ST',
    
    # Pre-created splines
    "fcoll": None,
    "dfcolldz": None,
    }
    
    pf.update(rcParams)

    return pf

def HaloParameters():
    pf = \
    {
    'halo_M': 1e10,
    'halo_c': 15.,
    'halo_profile': 'nfw',
    }
    
    return pf
    
def LuminosityFunctionParameters():
    pf = \
    {
     "lf": None,
     "lf_norm": 1.0,
     "lf_Lmin": 1e38,
     "lf_Lstar": 1e42,
     "lf_Emin": 2e3,
     "lf_Emax": 8e3,
     "lf_slope": -1.,
     "lf_gamma1": None,
     "lf_gamma2": None,
     "lf_func": 'schecter', # or "dpl" for double power-law,
     "lf_zfunc": None,      # governs redshift evolution
    }

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
    "load_ics": True,
    "load_sim": False,
    "first_light_redshift": 50.,

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
    "stop_xavg": 0.999,   # stop at given ionized fraction
    "track_extrema": False,
    "stop_delay": 0.5,      # differential redshift step
    "inline_analysis": None,
    "one_file_per_blob": False,
    "auto_generate_blobs": False,
    "override_blob_names": None,
    "override_blob_redshifts": None,
    ""
    
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
    'tanh_dz': 0.025,  # Redshift sampling
    'tanh_bias_temp': 0.0,   # in mK
    'tanh_bias_freq': 0.0,   # in MHz
    'output_frequencies': None,
    }
    
    pf.update(rcParams)

    return pf
    
def GaussianParameters():
    pf = \
    {
     'gaussian_model': False,
     'gaussian_A': -100., 
     'gaussian_nu': 70.,
     'gaussian_sigma': 10.,
     'gaussian_dz': 0.025,  # Redshift sampling
     'output_frequencies': None,
    }
    
    return pf


