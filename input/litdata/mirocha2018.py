from mirocha2016 import dpl, dflex
from ares.util import ParameterBundle as PB
from ares.physics.Constants import nu_0_mhz, h_p, erg_per_ev

##
# All need this!
##
base = PB(**dpl) + PB(**dflex) + PB('dust:var_beta')

lf = \
{
 'pq_func_par0{0}[12]': 11.6690040263,
 'pq_func_par0{0}[13]': 0.855092040361,
 'pq_func_par2{0}[0]': 1.03890453721,
 'pq_func_par2{0}[13]': -0.292169721106,
 'pq_func_par3{0}[0]': -0.372920485712,
 'pq_func_par2{0}[12]': -0.742521570574,
 'pq_func_par2{0}[11]': 0.0561650504878,
 'pq_func_par1{0}[0]': 1.8979710707e+11,
 'pq_func_par0{0}[11]': 0.249869796896,
 'pq_func_par0{0}[0]': 0.171411643892,
}

cold = \
{
 'approx_thermal_history': 'exp',
 'load_ics': 'parametric',
 'inits_Tk_p0': 194.002300947,
 'inits_Tk_p1': 1.20941098917,
 'inits_Tk_p2': -6.0088645858,
 
 'pq_func_par2{0}[4]': 0.0661410442476,
 'pq_func_par0{0}[8]': 19701832.4424,
 'pq_func_par2{0}[5]': -0.624613431068,
 'pq_func_par2{0}[8]': -0.74856754179,
 'pq_func_par2{0}[7]': 1.1001036864,
 'inits_Tk_p1': 1.20941098917,
 'pq_func_par0{0}[7]': 1.27092398097,
 'inits_Tk_p2': -6.0088645858,
 'pq_func_par0{0}[4]': -0.459851109813,
 'pq_func_par2{0}[1]': -0.978610145341,
 'pop_rad_yield{1}': 1.46150981378e+40,
 'pq_func_par0{0}[5]': 0.0338201803874,
 'pq_func_par2{0}[2]': 0.451825099378,
 'inits_Tk_p0': 194.002300947,
 'pq_func_par0{0}[2]': 2.57313940031e+11,
 'pq_func_par2{0}[3]': 0.0305460186899,
 'pq_func_par0{0}[3]': 0.899806521888,
 'pop_Tmin{0}': 18875.1354673,
 'pq_func_par0{0}[1]': 0.0195652006263,

}

E21 = nu_0_mhz * 1e6 * h_p / erg_per_ev
nu_min = 1e7  # in Hz
nu_max = 1e12
Emin = nu_min * (h_p / erg_per_ev)   # 1 GHz -> eV
Emax = nu_max * (h_p / erg_per_ev)   # 3 GHz -> eV

radio = \
{
 'pop_sfr_model{2}': 'link:sfrd:0',
 'pop_sed{2}': 'pl',
 'pop_alpha{2}': -0.7,
 'pop_Emin{2}': Emin,
 'pop_Emax{2}': Emax,
 'pop_EminNorm{2}': None,
 'pop_EmaxNorm{2}': None,
 'pop_Enorm{2}': E21, # 1.4 GHz
 'pop_rad_yield_units{2}': 'erg/s/sfr/hz', 
 
 # Solution method
 'pop_solve_rte{2}': True,
 'pop_radio_src{2}': True,
 'pop_lw_src{2}': False,
 'pop_lya_src{2}': False,
 'pop_heat_src_igm{2}': False,
 'pop_ion_src_igm{2}': False,
 'pop_ion_src_cgm{2}': False,

 # Best fitting parameters
 'pq_func_par2{0}[4]': -0.367507400514,
 'pq_func_par0{0}[8]': 88971.2720411,
 'pq_func_par2{0}[5]': 0.432165531607,
 'pq_func_par2{0}[8]': -0.459908108301,
 'pq_func_par2{0}[7]': 0.0804573261674,
 'pq_func_par0{0}[7]': 0.755153610936,
 'pop_Tmin{0}': 19345.5297538,
 'pq_func_par0{0}[4]': -0.332641612164,
 'pq_func_par2{0}[1]': -0.280720519413,
 'pop_rad_yield{1}': 4.20070368093e+40,
 'pq_func_par0{0}[5]': 0.0200865171317,
 'pq_func_par2{0}[2]': 0.100749514086,
 'pq_func_par0{0}[2]': 2.19664981719e+11,
 'pq_func_par2{0}[3]': -0.251920672275,
 'pq_func_par0{0}[3]': 0.772229741005,
 'pop_zdead{2}': 17.0252061155,
 'pop_rad_yield{2}': 1.37568148182e+32,
 'pq_func_par0{0}[1]': 0.0268026393973,
}
