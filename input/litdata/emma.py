
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
	#beta
	'pop_smhm_beta': 'pq[0]',
	'pq_func[0]': 'linear',
	'pq_func_var[0]': 'z',
	'pq_func_par0[0]': beta_0,
	'pq_func_par1[0]': 0,
	'pq_func_par2[0]': beta_1,     
}