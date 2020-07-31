
#From Moster2010, table 7
logM_0 = 11.88 #(0.01)
mu = 0.019 #(0.002) #or not
N_0 = 0.0282 #(0.0003)
nu = -0.72 #(0.06)
gamma_0 = 0.556 #0.001
gamma_1 = -0.26 #(0.05)
beta_0 = 1.06 #(0.06)
beta_1 = 0.17 #(0.12)


model1 = \
{
	'pop_sfr_model':'hod',

'pop_sf_type': 'tot', #tot, sf, q

	#star-forming fraction, linear wrt log10(HM)
	'pop_sf_fract': 'pq[7]',
	'pq_func[7]': 'p_linear',
	'pq_func_var[7]': 'Mh',
	'pq_func_par0[7]': 11, #log10_minSM,  8 SM
	'pq_func_par1[7]': 16, #log10_maxSM, 12.5 SM
	'pq_func_par2[7]': 0.95, #percent_for_minSM
	'pq_func_par3[7]': 0.25, #percent_for_maxSM


	#for LF
	'pop_lf': 'pq[4]',
	'pq_func[4]': 'linear',
	'pq_func_var[4]': 'z',
	'pq_func_par0[4]': 3e-4,
	'pq_func_par1[4]': 0,
	'pq_func_par2[4]': 0, 


	#for SMF - parameter for dpl
	#beta
	'pop_smhm_beta': 'pq[0]',
	'pq_func[0]': 'linear',
	'pq_func_var[0]': 'z',
	'pq_func_par0[0]': beta_0,
	'pq_func_par1[0]': 0,
	'pq_func_par2[0]': beta_1, 

	#norm
	'pop_smhm_n': 'pq[1]',
	'pq_func[1]': 'pl',
	'pq_func_var[1]': '1+z',
	'pq_func_par0[1]': N_0,
	'pq_func_par1[1]': 1.0,
	'pq_func_par2[1]': nu, 

	#gamma
	'pop_smhm_gamma': 'pq[2]',
	'pq_func[2]': 'pl',
	'pq_func_var[2]': '1+z',
	'pq_func_par0[2]': gamma_0,
	'pq_func_par1[2]': 1.0,
	'pq_func_par2[2]': gamma_1, 

	#peak mass
	'pop_smhm_m': 'pq[3]',
	'pq_func[3]': 'pl_10',
	'pq_func_var[3]': '1+z',
	'pq_func_par0[3]': logM_0,
	'pq_func_par1[3]': 1.0,
	'pq_func_par2[3]': mu, 


	#SFR - added with log10(_1)
	'pop_sfr_1': 'pq[5]',
	'pq_func[5]': 'linear',
	'pq_func_var[5]': 't',
	'pq_func_par0[5]': 0.84,
	'pq_func_par1[5]': 0.,
	'pq_func_par2[5]': -0.026, 

	'pop_sfr_2': 'pq[6]',
	'pq_func[6]': 'linear',
	'pq_func_var[6]': 't',
	'pq_func_par0[6]': 6.51,
	'pq_func_par1[6]': 0.,
	'pq_func_par2[6]': -0.11, 

}