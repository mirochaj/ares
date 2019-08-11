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
 'pq_func_par0[2]{0}': 3e11,
 'pq_func_par0[3]{0}': 0.5,
 'pq_func_par0[4]{0}': 0.0,          # High-mass slope
 'pq_func_par2[1]{0}': 0.,           # Redshift evolution
 
 'pq_func_par1[1]{0}': 5.,           # Pin to z = 4
}

base.update(_base)

_screen = \
{
 
 'pop_dust_yield{0}': 0.4,
 
 # Dust opacity vs. wavelength    
 "pop_dust_kappa{0}": 'pq[20]',   # opacity in [cm^2 / g]
 "pq_func[20]{0}": 'pl',
 'pq_func_var[20]{0}': 'wave',
 'pq_func_par0[20]{0}': 1e5,      # opacity at wavelength below
 'pq_func_par1[20]{0}': 1e3,
 'pq_func_par2[20]{0}': -1.,
 
 # Screen parameters
 'pop_dust_fcov{0}': 1,  
 "pop_dust_scale{0}": 'pq[22]',       # Scale radius [in kpc]
 "pq_func[22]{0}": 'pl',
 'pq_func_var[22]{0}': 'Mh',
 'pq_func_par0[22]{0}': 'pq[23]',     # Note that Rhalo ~ Mh^1/3 / (1+z)
 'pq_func_par1[22]{0}': 1e10,
 'pq_func_par2[22]{0}': 0.45,
 
 # Evolution of scale
 "pq_func[23]{0}": 'pl',
 'pq_func_var[23]{0}': '1+z',
 'pq_func_par0[23]{0}': 1.6,
 'pq_func_par1[23]{0}': 5.,
 'pq_func_par2[23]{0}': 0.,         # R(vir) goes like 1 / (1+z)
  
}

_screen_best = \
{
 'pq_func_par0[1]{0}': 0.0486191550793195,
 'pq_func_par0[2]{0}': 778500431735.2164,
 'pq_func_par0[3]{0}': 0.5124444731982576,
 'pq_func_par0[4]{0}': 0.03708330190131004,
 'pq_func_par2[22]{0}': 0.46260876352941155,
 'pq_func_par0[23]{0}': 1.3886146855924113,
}

screen = _screen.copy()
screen.update(_screen_best)

_patchy = \
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

patchy = screen.copy()
patchy.update(_patchy)

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

quench = \
{
 'pop_fduty{0}': 'pq[50]',
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

destruction = \
{
 'pq_func_par2[22]{0}': 0.,    
 'pq_func_par0[1]{0}': 0.0414924616502775,
 'pq_func_par0[2]{0}': 228025945619.58618,
 'pq_func_par0[3]{0}': 0.6314799815636458,
 'pq_func_par0[4]{0}': -0.5599463903428499,
 'pq_func_par0[23]{0}': 1.5452880256044765,
 'pq_func_par2[27]{0}': -1.4360103791063477,
 'pq_func_par0[28]{0}': 0.33198068910686596,
}

_scatter = \
{
 'pq_func_par2[22]{0}': 0.,
 'pq_func_par0[23]{0}': 3.3435661993369257,
 'pq_func_par0[1]{0}': 0.03599536365742767,
 'pq_func_par0[2]{0}': 167344779121.14706,
 'pq_func_par0[3]{0}': 1.1677597636641468,
 'pq_func_par0[4]{0}': 0.06069676909919603,
 
 "pop_dust_scatter{0}": 'pq[33]',
 "pq_func[33]{0}": 'pl',
 'pq_func_var[33]{0}': 'Mh',
 'pq_func_par0[33]{0}': 'pq[34]',
 'pq_func_par1[33]{0}': 1e10,
 'pq_func_par2[33]{0}': 0.,
   
 "pq_func[34]{0}": 'pl',
 'pq_func_var[34]{0}': '1+z',
 'pq_func_par0[34]{0}': 0.3,
 'pq_func_par1[34]{0}': 5.,
 'pq_func_par2[34]{0}': 0.,
}

scatter = screen.copy()
scatter.update(_scatter)

_composition = \
{
 'pq_func_par2[20]{0}': 'pq[31]',
 'pq_func[31]{0}': 'pl',
 'pq_func_var[31]{0}': 'Mh',
 'pq_func_par0[31]{0}': -1.,
 'pq_func_par1[31]{0}': 1e10,
 'pq_func_par2[31]{0}': 0.,
 
 # Best fit from scatter model
 'pq_func_par2[22]{0}': 0.,
 'pq_func_par0[23]{0}': 3.3435661993369257,
 'pq_func_par0[1]{0}': 0.03599536365742767,
 'pq_func_par0[2]{0}': 167344779121.14706,
 'pq_func_par0[3]{0}': 1.1677597636641468,
 'pq_func_par0[4]{0}': 0.06069676909919603,
}


composition = screen.copy()
composition.update(_composition)

