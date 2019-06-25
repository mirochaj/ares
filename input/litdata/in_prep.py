import numpy as np
from mirocha2017 import base as _base_
from mirocha2017 import dflex as _dflex_

base = _base_.copy()
base.update(_dflex_)

_base = \
{
 'pop_sfr_model{0}': 'ensemble', 
 'pop_sed{0}': 'eldridge2009',
 
 'pop_sed_degrade{0}': 10,
 'pop_thin_hist{0}': 10,
 'pop_aging{0}': True,
 'pop_ssp{0}': True,
 'pop_mass_yield{0}': 0.15,
 'pop_calib_L1600{0}': None,
 'pop_Z{0}': 0.002, 
 'pop_zdead{0}': 3.5,
 
 # Metallicity evolution!?
 'pop_enrichment{0}': False,
 'pop_metal_yield{0}': 0.1,
 'pop_mass_yield{0}': 0.15,
 'pop_fpoll{0}': 1,
 
 'pop_scatter_mar{0}': 0.3,
   
 'hmf_dt': 1.,
 'hmf_tmax': 2e3,
 'hmf_model': 'Tinker10',
 
 "sigma_8": 0.8159, 
 'primordial_index': 0.9652, 
 'omega_m_0': 0.315579, 
 'omega_b_0': 0.0491, 
 'hubble_0': 0.6726,
 'omega_l_0': 1. - 0.315579,

 # Screw with SFE.
 'pq_func_par0[1]{0}': 3e-2,         # SFE normalization [0.05 by default]
 'pq_func_par0[3]{0}': 0.5,
 'pq_func_par0[4]{0}': 0.0,          # High-mass slope
 'pq_func_par2[1]{0}': 0.,           # Redshift evolution
 
 'pq_func_par1[1]{0}': 5.,           # Pin to z = 4
}

base.update(_base)

screen = \
{
 
 # Dust opacity vs. wavelength    
    
 'pop_dust_fcov{0}': 1,  
 'pq_func_par2[22]{0}': 0.45,              # PL dependence of Rdust on Mh
 'pq_func_par0[23]{0}': 2.,                # normalization if Rdust [kpc]
 'pq_func_par2[23]{0}': 0.,                # PL dependence of Rdust on z
  
 'pop_dust_yield{0}': 0.4,
 
}

patchy = \
{
 'pop_dust_fcov{0}': 'pq[21]',
 'pq_func[21]{0}': 'log_tanh_abs',
 'pq_func_var[21]{0}': 'Mh',
 'pq_func_par0[21]{0}': 0.05,
 'pq_func_par1[21]{0}': 1.0,
 'pq_func_par2[21]{0}': 'pq[24]',
 'pq_func_par3[21]{0}': 0.2,
 
 # Redshift evolution of transition mass
 'pq_func[24]{0}': 'linear',
 'pq_func_var[24]{0}': '1+z',
 'pq_func_par0[24]{0}': 10.8,
 'pq_func_par1[24]{0}': 5.,
 'pq_func_par2[24]{0}': 0.0,
}

fduty = \
{
 'pop_fduty{0}': 'pq[40]',
 "pq_func[40]{0}": 'pl',
 'pq_func_var[40]{0}': 'Mh',
 'pq_func_par0[40]{0}': 'pq[41]',
 'pq_func_par1[40]{0}': 1e10,
 'pq_func_par2[40]{0}': 0.2,
 'pq_val_ceil[40]{0}': 1.0,
 
 # Redshift evolution of fduty normalization
 'pq_func[41]{0}': 'pl',
 'pq_func_var[41]{0}': '1+z',
 'pq_func_par0[41]{0}': 0.5,
 'pq_func_par1[41]{0}': 5.,
 'pq_func_par2[41]{0}': 0.0,
 
}


