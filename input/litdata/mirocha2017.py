import numpy as np
from mirocha2016 import dpl, dpl_flex

_popII_models = \
{
 'dpl': {},
 'shallow': {'pop_Z{0}': 1e-3, 'pop_rad_yield_Z_index{1}': -0.6},
 'deep': {'pop_logN{1}': 22.},
 'late': {'pop_Tmin{0}': 1e5},
}

_popII_models['early'] = dpl_flex
_popII_models['early']['pq_func_par2{0}[1]'] = 1.

popII_pars = _popII_models#{}
#for model in _popII_models:
#    p = dpl.copy()
#    p.update(_popII_models[model])
#    popII_pars[model] = p.copy()
#    
popII_markers = {'dpl': 's', 'shallow': '^', 'deep': 'v', 'early': '<', 'late': '>'}
popII_models = ['dpl', 'shallow', 'deep', 'late', 'early']

# relative to mirocha2016:dpl
_generic_updates = \
{
 'initial_redshift': 60,
 'pop_zform{0}': 60,
 'pop_zform{1}': 60,
 'feedback_LW': True,
}

# This can lead to pickling issues...argh
#halos = ares.physics.HaloMassFunction()
#barrier_A = lambda zz: halos.VirialMass(1e4, zz)

"""
First model: constant SFR in minihalos.
"""
_csfr_specific = \
{

 'kill_redshift': 5.6,
 'sam_dz': 0.1,

 'pop_zform{0}': 60,
 'pop_zform{1}': 60,
 'pop_zform{2}': 60,
 'pop_zform{3}': 60,

 'pop_sfr_model{2}': 'sfr-func',
 'pop_sfr{2}': 2e-5,
 'pop_sfr_cross_threshold{2}': False,
 'pop_sed{2}': 'eldridge2009',
 'pop_binaries{2}': False,
 'pop_Z{2}': 1e-3,
 'pop_Emin{2}': 10.19,
 'pop_Emax{2}': 24.6,
 'pop_rad_yield{2}': 'from_sed', # EminNorm and EmaxNorm arbitrary now
                                  # should make this automatic

 'pop_heat_src_igm{2}': False,
 'pop_ion_src_igm{2}': False,

  # Solve LWB!
 'pop_solve_rte{2}': (10.2, 13.6),

 'pop_sed{2}': 'bb',
 'pop_temperature{2}': 1e5,
 'pop_rad_yield{2}': 3e3,
 'pop_EminNorm{2}': 10.2,
 'pop_EmaxNorm{2}': 13.6,
 'pop_rad_yield_units{2}': 'photons/baryon',
 
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
 'pop_tau_Nz{3}': 1e3,
 'pop_approx_tau{3}': 'neutral',
 
 #'pop_Mmin{0}': 'link:Mmax_active:2',
 
 # Tmin here just an initial guess -- will get modified by feedback.
 'pop_Tmin{2}': 500.,
 'pop_Tmin{0}': None,
 'pop_Mmin{0}': 'link:Mmax:2',
 'pop_Tmax_ceil{2}': 1e8,
 'pop_sfr_cross_threshold{2}': False,
}

csfr = dpl.copy()
csfr.update(_generic_updates)
csfr.update(_csfr_specific)

"""
Second model: constant SFE in minihalos.
"""
_csfe_specific = \
{
 'pop_sfr_model{2}': 'sfe-func',
 'pop_sfr{2}': None,
 'pop_fstar{2}': 1e-4,
}

csfe = dpl.copy()
csfe.update(_generic_updates)
csfe.update(_csfr_specific)
csfe.update(_csfe_specific)

_csff_specific = \
{
 'pop_sfr{2}': 'pq[2]',
 'pq_func{2}[2]': 'pl',
 'pq_func_var{2}[2]': 'Mh',
 'pq_func_par0{2}[2]': 1e-4,
 'pq_func_par1{2}[2]': 1e8,
 'pq_func_par2{2}[2]': 1.,
}

csff = csfr.copy()
csff.update(_csff_specific)

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
                'popII_sfrd_bc',  'popIII_sfrd_bc', 
                'popII_Mmin', 'popIII_Mmin',
                'popII_Mmax', 'popIII_Mmax',
                'popII_nh', 'popIII_nh'],
 'blob_ivars': ('z', np.arange(5, 60.1, 0.1)),
 'blob_funcs': ['pops[0].SFRD', 'pops[2].SFRD', 'pops[0].SFRD_at_threshold',
    'pops[2].SFRD_at_threshold', 'pops[0].Mmin', 'pops[2].Mmin',
    'pops[0].Mmax', 'pops[2].Mmax', 'pops[0].nactive', 'pops[2].nactive'],
}

csfe_blobs = csfr_blobs
csff_blobs = csfr_blobs

dpl_blobs = \
{
 'blob_names': ['popII_sfrd_tot', 'popII_Mmin', 'popII_Mmax'],
 'blob_ivars': ('z', np.arange(5, 60.1, 0.1)),
 'blob_funcs': ['pops[0].SFRD', 'pops[0].Mmin', 'pops[0].Mmax'],
}

"""
From here on out are the parameters that govern the PopIII space surveyed.
"""

#_dplpb = ares.util.ParameterBundle('mirocha2016:dpl')
#pop_dpl = ares.populations.GalaxyPopulation(**_dplpb.pars_by_pop(0,1))
#SFR_ref = float(pop_dpl.SFR(z=20, Mh=1e8))
#SFE_ref = float(pop_dpl.SFE(z=20, Mh=1e8))
#SFF_ref = float(pop_dpl.SFR(z=20, Mh=1e8))
# This one depends on our CSFF pivot occurring at 1e8 Msun!

SFR_ref = 0.0010497749824767248
SFE_ref = 0.0020455358405117763
SFF_ref = SFR_ref

popIII_sfr_methods = ['csfr', 'csfe', 'csff']
popIII_trans_methods = ['bind', 'time', 'mass', 'temp']

popIII_sfr_vals = \
{
 'csfr': 10**np.arange(-2., 0., 1) * SFR_ref,
 'csfe': 10**np.arange(-2., 0., 1) * SFE_ref,
 'csff': 10**np.arange(-2., 0., 1) * SFF_ref,
}

popIII_sfr_vals_hires = \
{
 'csfr': 10**np.arange(-3., 0.5, 0.5) * SFR_ref,
 'csfe': 10**np.arange(-2., 1.5, 0.5) * SFE_ref,
 'csff': 10**np.arange(-2., 1.5, 0.5) * SFF_ref,
}

popIII_trans_vals = \
{
 'bind': [1e51, 1e52],
 'time': [1e1, 1e2],
 'mass': [1e2, 1e3],
 'temp': [1e3, 1e4],
}

popIII_trans_vals_hires = \
{
 'bind': np.arange(1e51, 1e52, 2e51),
 'time': np.arange(20, 120, 20),
 'mass': np.arange(100, 1e3, 200),
 'temp': np.arange(1e3, 1e4, 2e3),
}

def popIII_prefix_maker(reg, trans, hp1=None, hp2=None, XRIII=1., 
    lwfb=1, M0=0, t0=10, popII='dpl'):
    """
    Make a prefix duh.
    """
    
    if (hp1 is not None) and (hp2 is not None):
        prefix = '%s_%.1e_%s_%.1e_XR_%i_lwfb_%i_M0_%i_t0_%i_popII_%s' \
            % (reg, hp1, trans, hp2, XRIII, lwfb, M0, t0, popII)
    else:
        prefix = '%s_%s_XR_%i_lwfb_%i_M0_%i_t0_%i_popII_%s' \
            % (reg, trans, XRIII, lwfb, M0, t0, popII)

    return prefix










