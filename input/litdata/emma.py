
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

	#star-forming fraction - a, b dpl z dependance, other linear
	'pop_sf_A': 'pq[7]',
	'pq_func[7]': 'pl',
	'pq_func_var[7]': '1+z',
	'pq_func_par0[7]': -1.5, #const
	'pq_func_par1[7]': 1.0,
	'pq_func_par2[7]': 0.4, #m

	'pop_sf_B': 'pq[8]',
	'pq_func[8]': 'pl',
	'pq_func_var[8]': '1+z',
	'pq_func_par0[8]': -10.7,
	'pq_func_par1[8]': 1.0,
	'pq_func_par2[8]': -0.1,

	'pop_sf_C': 'pq[9]',
	'pq_func[9]': 'linear',
	'pq_func_var[9]': 'z',
	'pq_func_par0[9]': 1.8, #const
	'pq_func_par1[9]': 0, #offset
	'pq_func_par2[9]': 0.8, #m

	'pop_sf_D': 'pq[10]',
	'pq_func[10]': 'linear',
	'pq_func_var[10]': 'z',
	'pq_func_par0[10]': 0.5, #const
	'pq_func_par1[10]': 0, #offset
	'pq_func_par2[10]': 1.0, #m

	#for LF
	'pop_lf': 'pq[4]',
	'pq_func[4]': 'linear',
	'pq_func_var[4]': 'z',
	'pq_func_par0[4]': 3e-4,
	'pq_func_par1[4]': 0,
	'pq_func_par2[4]': 0, 


	#for SMF - default parameters for dpl
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


#SMF values updated from MCMC best fits
model2 = \
{
	'pop_sfr_model':'hod',

	#star-forming fraction - a, b dpl z dependance, other linear
	'pop_sf_A': 'pq[7]',
	'pq_func[7]': 'pl',
	'pq_func_var[7]': '1+z',
	'pq_func_par0[7]': -3.5,
	'pq_func_par1[7]': 1.0,
	'pq_func_par2[7]': 0.1,

	'pop_sf_B': 'pq[8]',
	'pq_func[8]': 'pl',
	'pq_func_var[8]': '1+z',
	'pq_func_par0[8]': -10.2,
	'pq_func_par1[8]': 1.0,
	'pq_func_par2[8]': 0.1,

	'pop_sf_C': 'pq[9]',
	'pq_func[9]': 'linear',
	'pq_func_var[9]': 'z',
	'pq_func_par0[9]': 2.1, #const
	'pq_func_par1[9]': 0, #offset
	'pq_func_par2[9]': 0.6, #m

	'pop_sf_D': 'pq[10]',
	'pq_func[10]': 'linear',
	'pq_func_var[10]': 'z',
	'pq_func_par0[10]': 1.05, #const
	'pq_func_par1[10]': 0, #offset
	'pq_func_par2[10]': 0.6, #m

	#for LF
	'pop_lf': 'pq[4]',
	'pq_func[4]': 'linear',
	'pq_func_var[4]': 'z',
	'pq_func_par0[4]': 1.52e-06,
	'pq_func_par1[4]': 0.13,
	'pq_func_par2[4]': 6.69e-05, 


	#for SMF - best fit parameters for dpl
	#beta
	'pop_smhm_beta': 'pq[0]',
	'pq_func[0]': 'linear',
	'pq_func_var[0]': 'z',
	'pq_func_par0[0]': 0.8828985178317218,
	'pq_func_par1[0]': 0,
	'pq_func_par2[0]': -0.03363387618820308, 

	#norm
	'pop_smhm_n': 'pq[1]',
	'pq_func[1]': 'pl',
	'pq_func_var[1]': '1+z',
	'pq_func_par0[1]': 0.010358061397412294,
	'pq_func_par1[1]': 1.0,
	'pq_func_par2[1]': 0.28690793780049106, 

	#gamma
	'pop_smhm_gamma': 'pq[2]',
	'pq_func[2]': 'pl',
	'pq_func_var[2]': '1+z',
	'pq_func_par0[2]': 0.5633902051902832,
	'pq_func_par1[2]': 1.0,
	'pq_func_par2[2]': 0.18194904277970236, 

	#peak mass
	'pop_smhm_m': 'pq[3]',
	'pq_func[3]': 'pl_10',
	'pq_func_var[3]': '1+z',
	'pq_func_par0[3]': 11.750289778904255,
	'pq_func_par1[3]': 1.0,
	'pq_func_par2[3]': 1.855774245368317, 


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

#SMF values updated from MCMC best fits
model3 = \
{
	'pop_sfr_model':'hod',

	#star-forming fraction - all dpl z dependance
	'pop_sf_A': 'pq[7]',
	'pq_func[7]': 'pl',
	'pq_func_var[7]': '1+z',
	'pq_func_par0[7]': -3.5,
	'pq_func_par1[7]': 1.0,
	'pq_func_par2[7]': 0.1,

	'pop_sf_B': 'pq[8]',
	'pq_func[8]': 'pl',
	'pq_func_var[8]': '1+z',
	'pq_func_par0[8]': -10.2,
	'pq_func_par1[8]': 1.0,
	'pq_func_par2[8]': 0.1,

	'pop_sf_C': 'pq[9]',
	'pq_func[9]': 'pl',
	'pq_func_var[9]': '1+z',
	'pq_func_par0[9]': 2.1,
	'pq_func_par1[9]': 1.0,
	'pq_func_par2[9]': 0.4,

	'pop_sf_D': 'pq[10]',
	'pq_func[10]': 'pl',
	'pq_func_var[10]': '1+z',
	'pq_func_par0[10]': 1.05,
	'pq_func_par1[10]': 1.0,
	'pq_func_par2[10]': 0.4,

	#for LF
	'pop_lf': 'pq[4]',
	'pq_func[4]': 'linear',
	'pq_func_var[4]': 'z',
	'pq_func_par0[4]': 1.52e-06,
	'pq_func_par1[4]': 0.13,
	'pq_func_par2[4]': 6.69e-05, 


	#for SMF - best fit parameters for dpl
	#beta
	'pop_smhm_beta': 'pq[0]',
	'pq_func[0]': 'linear',
	'pq_func_var[0]': 'z',
	'pq_func_par0[0]': 0.8828985178317218,
	'pq_func_par1[0]': 0,
	'pq_func_par2[0]': -0.03363387618820308, 

	#norm
	'pop_smhm_n': 'pq[1]',
	'pq_func[1]': 'pl',
	'pq_func_var[1]': '1+z',
	'pq_func_par0[1]': 0.010358061397412294,
	'pq_func_par1[1]': 1.0,
	'pq_func_par2[1]': 0.28690793780049106, 

	#gamma
	'pop_smhm_gamma': 'pq[2]',
	'pq_func[2]': 'pl',
	'pq_func_var[2]': '1+z',
	'pq_func_par0[2]': 0.5633902051902832,
	'pq_func_par1[2]': 1.0,
	'pq_func_par2[2]': 0.18194904277970236, 

	#peak mass
	'pop_smhm_m': 'pq[3]',
	'pq_func[3]': 'pl_10',
	'pq_func_var[3]': '1+z',
	'pq_func_par0[3]': 11.750289778904255,
	'pq_func_par1[3]': 1.0,
	'pq_func_par2[3]': 1.855774245368317, 


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


model4 = \
{
	'pop_sfr_model':'hod',

	#star-forming fraction - a, b linear z dependance, c, d constant 
	'pop_sf_A': 'pq[7]',
	'pq_func[7]': 'linear',
	'pq_func_var[7]': 'z',
	'pq_func_par0[7]': -1.2, #const
	'pq_func_par1[7]': 0.5,
	'pq_func_par2[7]': -0.4, #m

	'pop_sf_B': 'pq[8]',
	'pq_func[8]': 'linear',
	'pq_func_var[8]': 'z',
	'pq_func_par0[8]': -10.7,
	'pq_func_par1[8]': 0.0,
	'pq_func_par2[8]': -0.1,

	'pop_sf_C': 'pq[9]',
	'pq_func[9]': 'linear',
	'pq_func_var[9]': 'z',
	'pq_func_par0[9]': 2.0, #const
	'pq_func_par1[9]': 0,
	'pq_func_par2[9]': 0,

	'pop_sf_D': 'pq[10]',
	'pq_func[10]': 'linear',
	'pq_func_var[10]': 'z',
	'pq_func_par0[10]': 1.0, #const
	'pq_func_par1[10]': 0,
	'pq_func_par2[10]': 0,

	#for LF
	'pop_lf': 'pq[4]',
	'pq_func[4]': 'linear',
	'pq_func_var[4]': 'z',
	'pq_func_par0[4]': 3e-4,
	'pq_func_par1[4]': 0,
	'pq_func_par2[4]': 0, 


	#for SMF - default parameters for dpl
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

#hopefully all the best fit values
model5 = \
{
	'pop_sfr_model':'hod',

	#star-forming fraction - a, b linear z dependance, c, d constant 
	'pop_sf_A': 'pq[7]',
	'pq_func[7]': 'linear',
	'pq_func_var[7]': 'z',
	'pq_func_par0[7]': -1.5, #const
	'pq_func_par1[7]': 1.0,
	'pq_func_par2[7]': 0.4, #m

	'pop_sf_B': 'pq[8]',
	'pq_func[8]': 'linear',
	'pq_func_var[8]': 'z',
	'pq_func_par0[8]': -10.7,
	'pq_func_par1[8]': 1.0,
	'pq_func_par2[8]': -0.1,

	'pop_sf_C': 'pq[9]',
	'pq_func[9]': 'linear',
	'pq_func_var[9]': 'z',
	'pq_func_par0[9]': 1.8, #const
	'pq_func_par1[9]': 0,
	'pq_func_par2[9]': 0,

	'pop_sf_D': 'pq[10]',
	'pq_func[10]': 'linear',
	'pq_func_var[10]': 'z',
	'pq_func_par0[10]': 0.5, #const
	'pq_func_par1[10]': 0,
	'pq_func_par2[10]': 0,

	#for LF
	'pop_lf': 'pq[4]',
	'pq_func[4]': 'linear',
	'pq_func_var[4]': 'z',
	'pq_func_par0[4]': 3e-4,
	'pq_func_par1[4]': 0,
	'pq_func_par2[4]': 0, 


	#for SMF - default parameters for dpl
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