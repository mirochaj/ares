
logM_0 = 11.88 #(0.01)
mu = 0.019 #(0.002)
N_0 = 0.0282 #(0.0003)
nu = -0.72 #(0.06)
gamma_0 = 0.556 #0.001
gamma_1 = -0.26 #(0.05)
beta_0 = 1.06 #(0.06)
beta_1 = 0.17 #(0.12)


model1 = \
{
	'pop_sfr_model':'hod',

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

}