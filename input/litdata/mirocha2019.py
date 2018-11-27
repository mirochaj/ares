from mirocha2017 import dpl, dflex
from ares.util import ParameterBundle as PB
from ares.physics.Constants import nu_0_mhz, h_p, erg_per_ev

##
# All need this!
##
base = PB(verbose=0, **dpl) \
     + PB(verbose=0, **dflex) \
     + PB('dust:var_beta', verbose=0)
     
cold = \
{
 # New base_kwargs
 'approx_thermal_history': 'exp',
 'load_ics': 'parametric',
 
 # Copy-pasted over from best fits.
 'pq_func_par0{0}[1]': 0.018949141521,
 'pq_func_par0{0}[2]': 2.31016897023e+11,
 'pq_func_par0{0}[3]': 0.924929473207,
 'pq_func_par0{0}[4]': -0.345183026665,
 'pq_func_par2{0}[1]': -1.93710531526,
 'pq_func_par2{0}[2]': -0.0401589348891,
 'pq_func_par2{0}[3]': 0.496357034538,
 'pq_func_par2{0}[4]': -0.85185812704,
 'pq_func_par0{0}[5]': 0.0180819517762,
 'pq_func_par2{0}[5]': 0.633935667655,
 'pq_func_par0{0}[7]': 2.47340783415,
 'pq_func_par0{0}[8]': 22676116.9726,
 'pq_func_par2{0}[7]': 1.29978775053,
 'pq_func_par2{0}[8]': 2.51794223721,
 'inits_Tk_p0': 203.477407296,
 'inits_Tk_p1': 1.23612113669,
 'inits_Tk_p2': -6.89882925178,
 'pop_Tmin{0}': 18729.8083367,
 'pop_rad_yield{1}': 6.75648524393e+40,
}

E21 = nu_0_mhz * 1e6 * h_p / erg_per_ev
nu_min = 1e7  # in Hz
nu_max = 1e12
Emin = nu_min * (h_p / erg_per_ev)   # 1 GHz -> eV
Emax = nu_max * (h_p / erg_per_ev)   # 3 GHz -> eV

radio = \
{
 # New base_kwargs
 'pop_sfr_model{2}': 'link:sfrd:0',
 'pop_sed{2}': 'pl',
 'pop_alpha{2}': -0.7,
 'pop_Emin{2}': Emin,
 'pop_Emax{2}': Emax,
 'pop_EminNorm{2}': None,
 'pop_EmaxNorm{2}': None,
 'pop_Enorm{2}': E21, # 1.4 GHz
 'pop_rad_yield_units{2}': 'erg/s/sfr/hz', 
 
 'pop_solve_rte{2}': True,
 'pop_radio_src{2}': True,
 'pop_lw_src{2}': False,
 'pop_lya_src{2}': False,
 'pop_heat_src_igm{2}': False,
 'pop_ion_src_igm{2}': False,
 'pop_ion_src_cgm{2}': False,

 # Best fitting parameters
 'pq_func_par0{0}[1]': 0.0210546853809,
 'pq_func_par0{0}[2]': 3.3655857398e+11,
 'pq_func_par0{0}[3]': 0.83260420295,
 'pq_func_par0{0}[4]': -0.38893617798,
 'pq_func_par2{0}[1]': -1.65784835208,
 'pq_func_par2{0}[2]': 1.10105346147,
 'pq_func_par2{0}[3]': 0.213603033125,
 'pq_func_par2{0}[4]': -0.484706407852,
 'pq_func_par0{0}[5]': 0.0187421303099,
 'pq_func_par2{0}[5]': 0.390393567924,
 'pq_func_par0{0}[7]': 1.86134140738,
 'pq_func_par0{0}[8]': 632300.756957,
 'pq_func_par2{0}[7]': 0.404071491728,
 'pq_func_par2{0}[8]': -0.545718558631,
 'pop_rad_yield{2}': 1.23318228932e+33,
 'pop_zdead{2}': 16.3318857443,
 'pop_Tmin{0}': 23809.941444,
 'pop_rad_yield{1}': 9.74775993145e+41,
}
