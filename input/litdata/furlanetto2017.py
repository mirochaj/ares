from numpy import inf

# Calibration set!
energy = \
{

 'pop_fstar': None,
 'pop_fstar_max': 0.1,            # fstar <= this value
 
 # SFE (through mass loading factor)
 'pop_sfr_model': 'mlf-func',
 'pop_mlf': 'pq[0]',
 'pq_func[0]': 'pl_evolN',
 'pq_func_var[0]': 'Mh',
 'pq_func_var2[0]': '1+z',
 
 ##
 # Steve's Equation 13.
 ##
 'pq_func_par0[0]': 1.,
 'pq_func_par1[0]': 10**11.5,
 'pq_func_par2[0]': -2./3.,
 'pq_func_par3[0]': 9.,
 'pq_func_par4[0]': -1.,
 
 'pop_L1600_per_sfr': 1e-28,

}

momentum = energy.copy()
momentum['pop_fstar_max'] = 0.2
momentum['pq_func_par0[0]'] = 5.   # actually not sure what Steve uses here.
momentum['pq_func_par2[0]'] = -1./3.
momentum['pq_func_par4[0]'] = -0.5

fshock = \
{
 # Massive end
 'pop_fshock': 'pq[1]',
 'pq_func[1]': 'pl_evolN',
 'pq_func_var[1]': 'Mh',
 'pq_func_var2[1]': '1+z',
 
 'pq_val_ceil[1]': 1.0,           # fshock <= 1

 # Steve's Equation 6 (from Faucher-Giguere+ 2011)
 'pq_func_par0[1]': 0.47,
 'pq_func_par1[1]': 1e12,
 'pq_func_par2[1]': -0.25,
 'pq_func_par3[1]': 4.,
 'pq_func_par4[1]': 0.38,
 
}

