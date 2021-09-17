"""

ProblemTypes.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Mar  7 15:53:15 2012

Description: Non-default parameter sets for certain test problems.

Note: Integer problem types imply use of a continuous SED, while their
non-integer counterparts imply a discrete SED, except for ProblemType = 1 = 1.1.
I generally (for clarity) order parameters from top to bottom in the following
way:
    Units, time/data dump interval
    Integral tabulation
    Initial conditions
    Source parameters
    Physics parameters

More notes:
-A problem type > 10 (or < -10) corresponds to the same problem as
 problem_type % 10, except helium is included.

"""

import numpy as np
from .SetDefaultParameterValues import SetAllDefaults
from ..physics.Constants import m_H, cm_per_kpc, cm_per_mpc, s_per_myr, E_LL

defs = SetAllDefaults()

def RaySegmentProblem(ptype):

    ptype_int = int(ptype)

    if abs(ptype_int) > 10:
        ptype_int -= 10 * np.sign(ptype_int)
        ptype_mod1 = round(ptype - 10 - ptype_int, 1)
    else:
        ptype_mod1 = round(ptype - ptype_int, 1)

    # Single-zone, cosmological expansion test
    if ptype_int == -1:
        pf = {
              "problem_type": -1,
              "radiative_transfer": 0,
              "isothermal": 0,
              "expansion": 1,
              "compton_scattering": 1,
              "grid_cells": 1,
              "length_units": 1e-4*cm_per_kpc, # 100 milliparsecs
              "start_radius": 0.99, # cell = 1 milliparsec across
              "dtDataDump": 1.,
              "dzDataDump": 0.1,
              "initial_redshift": 1e3,
              "initial_ionization": [1.-0.049, 0.049, 1-2e-8, 1e-8, 1e-8],
              "final_redshift": 10,
              "stop_time": 500.,
              "restricted_timestep": ['electrons', 'ions', 'temperature',
                  'hubble'],
             }

    # RT06-0.3, Single zone ionization/heating, then source switches off.
    if ptype_int == 0:
        pf = {
              "problem_type": 0,
              "plane_parallel": 1,
              "isothermal": 1,
              "density_units": 1.0,
              "length_units": 1e-4 * cm_per_kpc, # 100 milliparsecs
              "time_units": s_per_myr,
              "start_radius": 0.99999999,   # cell = 1 milliparsec across
              "grid_cells": 1,

              "stop_time": 10,
              "logdtDataDump": 0.1,
              "dtDataDump": None,
              "initial_timestep": 1e-15,
              "max_timestep": 0.1,
              "restricted_timestep": ['ions', 'electrons', 'temperature'],

              "initial_temperature": 1e2,
              "initial_ionization": [1. - 1e-6, 1e-6, 1.-2e-6, 1e-6, 1e-6],

              "source_type": 'star',
              "source_qdot": 1e12,
              "source_lifetime": 0.5,

              "source_sed": 'bb',
              "tau_ifront": [0],

              "source_Emin": E_LL,
              "source_Emax": 100.,
              "source_EminNorm": 0.1,
              "source_EmaxNorm": 100.,

             }

    # RT06-1, RT1: Pure hydrogen, isothermal HII region expansion,
    # monochromatic spectrum at 13.6 eV
    if ptype_int == 1:
        pf = {
              "problem_type": 1,
              "density_units": 1e-3,
              "length_units": 6.6 * cm_per_kpc,
              "stop_time": 500.0,
              "isothermal": 1,
              "secondary_ionization": 0,
              "initial_temperature": 1e4,
              "initial_ionization": [1.-1.2e-3, 1.2e-3, 1-2e-8, 1e-8, 1e-8],
              "source_type": 'toy',
              "source_qdot": 5e48,
              "source_E": [E_LL],
              "source_LE": [1.0],
             }

    # RT06-2: Pure hydrogen, HII region expansion, temperature evolution
    # allowed, *continuous spectrum*
    if ptype_int == 2:
        pf = {
              "problem_type": 2,
              "density_units": 1e-3,
              "length_units": 6.6 * cm_per_kpc,
              "stop_time": 100.0,
              "isothermal": 0,
              "restricted_timestep": ['ions', 'temperature'],
              "initial_temperature": 1e2,
              "initial_ionization": [1.-1.2e-3, 1.2e-3, 1.-2e-8, 1e-8, 1e-8],
              "source_type": 'star',
              "source_temperature": 1e5,
              "source_sed": 'bb',
              "source_qdot": 5e48,
              "source_EminNorm": 1e-1,
              "source_EmaxNorm": 5e2
             }

    # RT06-3: I-front trapping in a dense clump and the formation of a shadow,
    # continuous blackbody spectrum
    if ptype_int == 3:
        pf = {
              "problem_type": 3,
              "plane_parallel": 1,
              "density_units": 2e-4,
              "grid_cells": 128,
              "length_units": 6.6 * cm_per_kpc,

              "initial_timestep": 1e-8,
              "tables_dlogN": [0.01],

              "stop_time": 15.0,
              "dtDataDump": 1.0,
              "isothermal": 0,
              "initial_temperature": 8e3,
              "initial_ionization": [1.-1e-4, 1e-4, 1.-2e-4, 1e-4, 1e-4],
              "source_type": 'star',
              "source_qdot": 1e6,
              "source_sed": 'bb',
              "source_temperature": 1e5,

              "restricted_timestep": ['ions', 'electrons', 'temperature'],

              "source_Emin": E_LL,
              "source_Emax": 100.,
              "source_EminNorm": 1e-1,
              "source_EmaxNorm": 5e2,

              "slab": 1,
              "slab_position": 5.0 / 6.6,
              "slab_overdensity": 200.,
              "slab_radius": 0.8 / 6.6,
              "slab_temperature": 40.,
              "slab_profile": 0,
              "slab_ionization": [1.-1e-4, 1e-4],

             }

    if ptype_mod1 != 0:
        pf.update({'source_type': 'toy'})

        # Change discrete spectrum: 0.1 = Mirocha et al. 2012
        #                           0.2 = Wise & Abel 2011
        if ptype_mod1 == 0.1:
            pf.update({'source_E': [17.98, 31.15, 49.09, 76.98]})
            pf.update({'source_LE': [0.23, 0.36, 0.24, 0.06]})
        if ptype_mod1 == 0.2:
            pf.update({'source_E': [18.29, 31.46, 49.13, 77.23]})
            pf.update({'source_LE': [0.24, 0.35, 0.23, 0.06]})

    if 10 <= ptype <= 20:
        helium_pars = \
            {
             'include_He': True,
             'tables_dlogN': defs['tables_dlogN']*3,
             'tables_xmin': defs["tables_xmin"]*3,
             'tables_logNmin': defs['tables_logNmin']*3,
             'tables_logNmax': defs['tables_logNmax']*3,
             'initial_ionization': [1.-1e-8, 1e-8, 1.-2e-8, 1e-8, 1e-8]
            }

        pf.update(helium_pars)

    return pf

def ReionizationProblem(ptype):
    """
    Problems using MultiPhaseMedium or MetaGalacticBackground.
    """

    ptype -= 100

    ptype_int = int(ptype)

    # If 110-120, include helium
    if abs(ptype_int) > 10:
        ptype_int -= 10 * np.sign(ptype_int)
        ptype_mod1 = round(ptype - 10 - ptype_int, 1)
    else:
        ptype_mod1 = round(ptype - ptype_int, 1)

    # Single-zone reionization problem
    if ptype_int == 5:
        pf = \
        {
         'problem_type': 100,
         "grid_cells": 1,

         'pop_type': 'galaxy',
         'pop_sfrd': 'robertson2015',
         'pop_sed': 'pl',
         'pop_alpha': 1.0,
         'pop_Emin': E_LL,
         'pop_Emax': 24.6,
         'pop_EminNorm': E_LL,
         'pop_EmaxNorm': 24.6,
         'pop_rad_yield': 10**53.14,
         'pop_rad_yield_units': 'photons/s/sfr',
         'initial_redshift': 30.,
         'final_redshift': 4.,
         'include_igm': False,                   # single-zone model
         'cgm_initial_temperature': 2e4,         # should be 20,000 K
         'clumping_factor': 3.,
         'pop_fesc': 0.2,
         'cgm_recombination': 'B',
         'cgm_collisional_ionization': False,
        }

    # Simple global 21-cm problem
    if ptype_int == 0:
        # Blank slate
        pf = {}

    # Simple 3-pop model (each pop only 1 type of radiation)
    elif ptype_int == 1:
        pf = \
        {

        'problem_type': 101,
        "grid_cells": 1,

        # Emits LW
        'pop_type{0}': 'galaxy',
        'pop_sfr_model{0}': 'fcoll',
        'pop_Tmin{0}': 1e4,
        'pop_fstar{0}': 0.1,
        "pop_lya_src{0}": True,
        "pop_ion_src_cgm{0}": False,
        "pop_ion_src_igm{0}": False,
        "pop_heat_src_cgm{0}": False,
        "pop_heat_src_igm{0}": False,

        "pop_Emin{0}": 10.2,
        "pop_Emax{0}": E_LL,
        "pop_EminNorm{0}": 10.2,
        "pop_EmaxNorm{0}": E_LL,
        "pop_rad_yield{0}": 9690.,
        "pop_rad_yield_units{0}": 'photons/baryon',
        "pop_solve_rte{0}": False,

        # Emits X-rays
        'pop_type{1}': 'galaxy',
        'pop_sfr_model{1}': 'link:sfrd:0',
        "pop_lya_src{1}": False,
        "pop_ion_src_cgm{1}": False,
        "pop_ion_src_igm{1}": True,
        "pop_heat_src_cgm{1}": False,
        "pop_heat_src_igm{1}": True,

        "pop_sed{1}": 'pl',
        "pop_alpha{1}": -1.5,

        "pop_Emin{1}": 2e2,
        "pop_Emax{1}": 3e4,
        "pop_EminNorm{1}": 5e2,
        "pop_EmaxNorm{1}": 8e3,

        "pop_Ex{1}": 500.,
        "pop_rad_yield{1}": 2.6e39,
        "pop_rad_yield_units{1}": 'erg/s/SFR',
        "pop_solve_rte{1}": False,

        # Emits ionizing photons
        'pop_type{2}': 'galaxy',
        'pop_sfr_model{2}': 'link:sfrd:0',
        "pop_lya_src{2}": False,
        "pop_ion_src_cgm{2}": True,
        "pop_ion_src_igm{2}": False,
        "pop_heat_src_cgm{2}": False,
        "pop_heat_src_igm{2}": False,

        "pop_fesc{2}": 0.1,

        "pop_Emin{2}": E_LL,
        "pop_Emax{2}": 1e2,
        "pop_EminNorm{2}": E_LL,
        "pop_EmaxNorm{2}": 1e2,
        "pop_rad_yield{2}": 4000.,
        "pop_rad_yield_units{2}": 'photons/baryon',
        "pop_solve_rte{2}": False,

        }

    elif ptype_int == 2:
        pf = \
        {

        'problem_type': 102,
        "grid_cells": 1,

        # Emits UV photons
        'pop_type{0}': 'galaxy',
        "pop_lya_src{0}": True,
        "pop_ion_src_cgm{0}": True,
        "pop_ion_src_igm{0}": False,
        "pop_heat_src_cgm{0}": False,
        "pop_heat_src_igm{0}": False,

        "pop_fesc{0}": 0.1,

        "pop_Emin{0}": 10.2,
        "pop_Emax{0}": 24.6,
        "pop_EminNorm{0}": E_LL,
        "pop_EmaxNorm{0}": 24.6,

        "pop_sed{0}": 'eldridge2009',
        "pop_Z{0}": 0.02,
        "pop_ssp{0}": False,
        "pop_tsf{0}": 100.,

        # Emits X-rays
        'pop_type{1}': 'galaxy',
        'pop_tunnel{1}': 0,         # Takes SFRD from population 1
        "pop_lya_src{1}": False,
        "pop_ion_src_cgm{1}": False,
        "pop_ion_src_igm{1}": True,
        "pop_heat_src_cgm{1}": False,
        "pop_heat_src_igm{1}": True,

        "pop_sed{1}": 'pl',
        "pop_alpha{1}": -1.5,

        "pop_Emin{1}": 2e2,
        "pop_Emax{1}": 3e4,
        "pop_EminNorm{1}": 5e2,
        "pop_EmaxNorm{1}": 8e3,

        "pop_Ex": 500.,
        "pop_rad_yield{1}": 2.6e39,
        "pop_rad_yield_units{1}": 'erg/s/SFR',
        "pop_solve_rte{1}": False,
        }
    elif ptype_int == 3:
        pf = \
        {

        'problem_type': 103,
        "grid_cells": 1,

        # Emits UV photons
        'pop_type': 'galaxy',
        "pop_lya_src": True,
        "pop_ion_src_cgm": True,
        "pop_ion_src_igm": False,
        "pop_heat_src_cgm": False,
        "pop_heat_src_igm": False,

        "pop_fesc": 0.1,

        "pop_Emin": 10.2,
        "pop_Emax": 24.6,
        "pop_EminNorm": E_LL,
        "pop_EmaxNorm": 24.6,

        "pop_sed": 'eldridge2009',
        "pop_Z": 0.02,
        "pop_ssp": False,
        "pop_tsf": 100.,
        }

    pf['load_ics'] = True
    pf['cosmological_ics'] = True

    return pf

def GalaxyProblem(ptype):
    pass

def ProblemType(ptype):
    """
    Storage bin for predefined problem types.

    Parameters
    ----------
    ptype : int, float
        Problem type!

    Problem Types
    -------------
    - 0-100 (1-D radiative transfer)
    - 100+  (Global 21-cm)

    Returns
    -------
    Dictionary of parameters and values for given ptype.

    """

    if ptype < 100:
        return RaySegmentProblem(ptype)
    else:
        return ReionizationProblem(ptype)

    return pf
