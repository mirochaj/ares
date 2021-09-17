import numpy as np
from mirocha2017 import dpl, base, flex
from ares.physics.Constants import E_LyA, E_LL

_popII_models = {}
for model in ['fall', 'strong_fall', 'weak_fall', 'soft_fall']:
    _popII_models[model] = flex.copy()
    _popII_models[model]['pq_func_par2{0}[1]'] = 1

for model in ['rise', 'strong_rise', 'weak_rise', 'soft_rise']:
    _popII_models[model] = flex.copy()
    _popII_models[model]['pq_func_par2{0}[1]'] = -1

for model in ['dpl', 'strong', 'weak', 'strong_weak']:
    _popII_models[model] = {}

_popII_models['soft'] = {}
_popII_models['early'] = flex.copy()
_popII_models['strong_early'] = flex.copy()

_sed_soft = \
{

 # Emits X-rays
 "pop_lya_src{2}": False,
 "pop_ion_src_cgm{2}": False,
 "pop_ion_src_igm{2}": True,
 "pop_heat_src_cgm{2}": False,
 "pop_heat_src_igm{2}": True,

 "pop_sed{2}": 'pl',
 "pop_alpha{2}": -1.5,
 'pop_logN{2}': -np.inf,

 "pop_Emin{2}": 2e2,
 "pop_Emax{2}": 3e4,
 "pop_EminNorm{2}": 5e2,
 "pop_EmaxNorm{2}": 8e3,

 #"pop_Ex{2}": 500.,
 "pop_rad_yield{2}": 2.6e40,
 "pop_rad_yield_units{2}": 'erg/s/SFR',
}

_sed_soft['pop_sfr_model{2}'] = 'link:sfrd:0'
_sed_soft['pop_alpha{2}'] = -2.5
_sed_soft['pop_solve_rte{2}'] = True
_sed_soft['tau_approx'] = 'neutral'

_popII_updates = \
{
 'dpl': {},
 'fall': {},
 'rise': {},
 'strong': {'pop_Z{0}': 1e-3, 'pop_rad_yield_Z_index{1}': -0.6},
 'strong_rise': {'pop_Z{0}': 1e-3, 'pop_rad_yield_Z_index{1}': -0.6},
 'strong_fall': {'pop_Z{0}': 1e-3, 'pop_rad_yield_Z_index{1}': -0.6},
 'weak': {'pop_logN{1}': 22.5},
 'weak_rise': {'pop_logN{1}': 22.5},
 'weak_fall': {'pop_logN{1}': 22.5},
 'strong_weak': {'pop_Z{0}': 1e-3, 'pop_rad_yield_Z_index{1}': -0.6, 'pop_logN{1}': 22.5},
 'soft': _sed_soft,
 'soft_rise': _sed_soft,
 'soft_fall': _sed_soft,
 'early': {'pop_Tmin{0}': 500.},
 'strong_early': {'pop_Tmin{0}': 500., 'pop_Z{0}': 1e-3, 'pop_rad_yield_Z_index{1}': -0.6},
}

for _update in _popII_updates.keys():
    _popII_models[_update].update(_popII_updates[_update])

popII_pars = _popII_models

popII_markers = \
{
'dpl': 's', 'strong': '^', 'weak': 'v',
'fall': '<', 'rise': '>', 'strong_weak': 'D',
'strong_fall': '^', 'weak_rise': 'v',
'strong_rise': '^', 'weak_fall': 'v', 'soft': 's',
'soft_fall': '<', 'soft_rise': '>',
'early': '<', 'strong_early': '<',
}

# Lotta trouble just to get 'dpl' first in the list...
_amp =  ['weak', 'strong']
_timing = ['rise', 'fall']
_all = _amp + _timing
popII_models = ['dpl'] + _all + ['strong_weak'] + ['early', 'strong_early']

for e1 in _amp:
    for e2 in _timing:
        popII_models.append('{0!s}_{1!s}'.format(e1, e2))

popII_models.extend(['soft', 'soft_fall', 'soft_rise'])

# relative to mirocha2016:dpl
_generic_updates = \
{
 'initial_redshift': 60,
 'pop_zform{0}': 60,
 'pop_zform{1}': 60,
 'feedback_LW': True,
 'sam_dz': None, # != None is causing crashes nowadays...
 'kill_redshift': 5.6,

 'feedback_LW_Mmin_rtol': 0,
 'feedback_LW_sfrd_rtol': 5e-2,
 'feedback_LW_sfrd_popid': 2,
 'feedback_LW_maxiter': 50,
 'feedback_LW_mixup_freq': 5,
 'feedback_LW_mixup_delay': 10,
 'pop_time_limit_delay{2}': True,
}

# This can lead to pickling issues...argh
#halos = ares.physics.HaloMassFunction()
#barrier_A = lambda zz: halos.VirialMass(1e4, zz)

"""
First model: constant SFR in minihalos.
"""
_low = \
{

 'pop_zform{2}': 60.,
 'pop_zform{3}': 60.,
 'pop_zdead{2}': 5.,
 'pop_zdead{3}': 5.,

 'pop_sfr_model{2}': 'sfr-func',
 'pop_sfr{2}': 1e-5,
 'pop_sfr_cross_threshold{2}': False,
 'pop_sed{2}': 'eldridge2009',
 'pop_binaries{2}': False,
 'pop_Z{2}': 1e-3,
 'pop_Emin{2}': E_LyA,
 'pop_Emax{2}': 24.6,
 #'pop_EminNorm{2}': 0.,
 #'pop_EmaxNorm{2}': np.inf,

 'pop_heat_src_igm{2}': False,
 'pop_ion_src_igm{2}': False,

  # Solve LWB!
 'pop_solve_rte{2}': (10.2, 13.6),

 'pop_sed{2}': 'eldridge2009',
 'pop_rad_yield{2}': 'from_sed',
 'pop_Z{2}': 0.02,

 # Radiative knobs
 'pop_fesc_LW{2}': 1.,
 'pop_fesc{2}': 0.0,
 'pop_rad_yield{3}': 2.6e39,

 # Other stuff needed for X-rays
 'pop_sfr_model{3}': 'link:sfrd:2',
 'pop_sed{3}': 'mcd',
 'pop_Z{3}': 1e-3,
 'pop_rad_yield_Z_index{3}': None,
 'pop_Emin{3}': 2e2,
 'pop_Emax{3}': 3e4,
 'pop_EminNorm{3}': 5e2,
 'pop_EmaxNorm{3}': 8e3,
 'pop_ion_src_cgm{3}': False,
 'pop_logN{3}': -np.inf,

 'pop_solve_rte{3}': True,

 #'pop_Mmin{0}': 'link:Mmax_active:2',

 # Tmin here just an initial guess -- will get modified by feedback.
 'pop_Tmin{2}': 500.,
 'pop_Tmin{0}': None,
 'pop_Mmin{0}': 'link:Mmax:2',
 'pop_Tmax_ceil{2}': 1e6,
 'pop_sfr_cross_threshold{2}': False,

 'pop_time_limit{2}': 2.5,
 'pop_bind_limit{2}': 1e51,
 'pop_abun_limit{2}': None,

 # Acknowledging that our mean metallicity is kludgey
 # Note that this renders the stellar mass meaningless (it'll be zero).
 'pop_mass_yield{2}': 1.0,
 'pop_metal_yield{2}': 1.0,
}

low = _generic_updates.copy()
low.update(_low)

high = low.copy()
high['pop_sed{2}'] = 'schaerer2002'
high['pop_mass{2}'] = 120.

med = low.copy()
med['pop_sed{2}'] = 'schaerer2002'
med['pop_mass{2}'] = 5.

LWoff = low.copy()
LWoff['pop_fesc_LW{2}'] = 0

bb = low.copy()
bb['pop_sed{2}'] = 'bb'
bb['pop_rad_yield{2}'] = 1e5
bb['pop_rad_yield_units{2}'] = 'photons/baryon'
bb['pop_EminNorm{2}'] = 11.2
bb['pop_EmaxNorm{2}'] = 13.6
bb['pop_Emin{2}'] = 1.
bb['pop_Emax{2}'] = 1e2
bb['pop_temperature{2}'] = 1e5

popII_like = low.copy()
popII_like['pop_sed{2}'] = 'eldridge2009'
popII_like['pop_Z{2}'] = 1e-3

"""
Second model: constant SFE in minihalos.
"""
#_csfe_specific = \
#{
# 'pop_sfr_model{2}': 'sfe-func',
# 'pop_sfr{2}': None,
# 'pop_fstar{2}': 1e-4,
#}
#
#csfe = dpl.copy()
#csfe.update(_generic_updates)
#csfe.update(_csfr_specific)
#csfe.update(_csfe_specific)
#
#_csff_specific = \
#{
# 'pop_sfr{2}': 'pq[2]',
# 'pq_func{2}[2]': 'pl',
# 'pq_func_var{2}[2]': 'Mh',
# 'pq_func_par0{2}[2]': 1e-4,
# 'pq_func_par1{2}[2]': 1e8,
# 'pq_func_par2{2}[2]': 1.,
#}
#
#csff = csfr.copy()
#csff.update(_csff_specific)

"""
Third model: extrapolated SFE in minihalos (i.e., same SFE as atomic halos).
"""
#_xsfe_specific = \
#{
#
# 'pop_fesc_LW{0}': 'pq[101]',
# 'pq_func{0}[101]': 'astep',
# 'pq_func_var{0}[101]': 'Mh',
# 'pq_func_par0{0}[101]': 1.,
# 'pq_func_par1{0}[101]': 1.,
# 'pq_func_par2{0}[101]': (barrier_A, 'z', 1),
#
# 'pop_fesc{0}': 'pq[102]',
# 'pq_func{0}[102]': 'astep',
# 'pq_func_var{0}[102]': 'Mh',
# 'pq_func_par0{0}[102]': 0., # No LyC from minihalos by default
# 'pq_func_par1{0}[102]': 0.1,
# 'pq_func_par2{0}[102]': (barrier_A, 'z', 1),
#
# 'pop_Tmin{0}': 500.,
# 'pop_Mmin{1}': 'pop_Mmin{0}',
#
# 'pop_sfr_model{1}': 'link:sfe:0',
#
# # X-ray sources
# 'pop_rad_yield{1}': 'pq[103]',
# 'pq_func{1}[103]': 'astep',
# 'pq_func_var{1}[103]': 'Mh',
# 'pq_func_par0{1}[103]': 2.6e39,
# 'pq_func_par1{1}[103]': 2.6e39,
# 'pq_func_par2{1}[103]': (barrier_A, 'z', 1),
#}
#
#xsfe = dpl.copy()
#xsfe.update(_generic_updates)
#xsfe.update(_xsfe_specific)

csfr_blobs = \
{
 'blob_names': ['popII_sfrd_tot', 'popIII_sfrd_tot',
                'popII_Mmin', 'popIII_Mmin',
                'popII_Mmax', 'popIII_Mmax',
                'popII_nh', 'popIII_nh'],
 'blob_ivars': ('z', np.arange(5, 60.1, 0.1)),
 'blob_funcs': ['pops[0].SFRD', 'pops[2].SFRD',
    'pops[0].Mmin', 'pops[2].Mmin',
    'pops[0].Mmax', 'pops[2].Mmax', 'pops[0].nactive', 'pops[2].nactive'],
 'blob_kwargs': [None] * 8,
}

csfe_blobs = csfr_blobs
csff_blobs = csfr_blobs

dpl_blobs = \
{
 'blob_names': [['popII_sfrd_tot', 'popII_Mmin', 'popII_Mmax']],
 'blob_ivars': [('z', np.arange(5, 60.1, 0.1))],
 'blob_funcs': [['pops[0].SFRD', 'pops[0].Mmin', 'pops[0].Mmax']],
}
