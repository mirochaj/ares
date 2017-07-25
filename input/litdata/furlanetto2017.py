from numpy import inf

# Calibration set!
energy = \
{

 'pop_fstar': None,
 'pop_fstar_max': 0.1,            # fstar <= this value
 
 # SFE (through mass loading factor)
 'pop_mlf': 'pq[0]',
 'pq_func[0]': 'pl',
 'pq_func_var[0]': 'Mh',
 
 ##
 # Steve's Equation 13.
 ##
 'pq_func_par0[0]': 'pq[1]',
 'pq_func_par1[0]': 10**11.5,
 'pq_func_par2[0]': -2./3.,
 
 # The redshift-dependent part
 'pq_func[1]': 'pl',
 'pq_func_var[1]': '1+z',
 'pq_func_par0[1]': 1., # really = 10 * epsilon_K * omega_49'
 'pq_func_par1[1]': 9., 
 'pq_func_par2[1]': -1.,
 
 'pop_L1600_per_sfr': 1e-28,

}

momentum = energy.copy()
momentum['pop_fstar_max'] = 0.2
momentum['pq_func_par0[1]'] = 5.   # actually not sure what Steve uses here.
momentum['pq_func_par2[0]'] = -1./3.
momentum['pq_func_par2[1]'] = -0.5

fshock = \
{
 # Massive end
 'pop_fshock': 'pq[2]',
 'pq_func[2]': 'pl',
 'pq_func_var[2]': 'Mh',
 'pq_val_ceil[2]': 1.0,           # fshock <= 1

 # Steve's Equation 6 (from Faucher-Giguere+ 2011)
 'pq_func_par0[2]': 'pq[3]',
 'pq_func_par1[2]': 1e12,
 'pq_func_par2[2]': -0.25,
 
 'pq_func[3]': 'pl',
 'pq_func_var[3]': '1+z',
 'pq_func_par0[3]': 0.47,
 'pq_func_par1[3]': 4.,
 'pq_func_par2[3]': 0.38,
}

