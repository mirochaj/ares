import numpy as np

# Giving fsmooth 5 <= PQs < 10         
fsmooth = \
{
 'pop_fsmooth': 'pq[5]',
 'pq_func[5]': 'log_tanh_abs',
 'pq_func_var[5]': 'Mh',
 'pq_func_par0[5]': 0.7,
 'pq_func_par1[5]': 0.3,
 'pq_func_par2[5]': 12.,
 'pq_func_par3[5]': 1.,
}

fsmooth_evol = fsmooth.copy()
for j, i in enumerate(range(6, 10)):
    fsmooth_evol['pq_func_par%i[5]' % j] = 'pq[%i]' % i
    fsmooth_evol['pq_func[%i]' % i] = 'pl'
    fsmooth_evol['pq_func_var[%i]' % i] = '1+z'
    fsmooth_evol['pq_func_par0[%i]' % i] = fsmooth['pq_func_par%i[5]' % j]
    fsmooth_evol['pq_func_par1[%i]' % i] = 7.
    fsmooth_evol['pq_func_par2[%i]' % i] = 0.

# Giving fsmooth 10 <= PQs < 15
fobsc = \
{
 'pop_fobsc': 'pq[10]',
 'pq_func[10]': 'log_tanh_abs',
 'pq_func_var[10]': 'Mh',
 'pq_func_par0[10]': 0.1,
 'pq_func_par1[10]': 0.9,
 'pq_func_par2[10]': 11.5,
 'pq_func_par3[10]': 0.5,
}

fobsc_evol = fobsc.copy()
for j, i in enumerate(range(11, 15)):
    fobsc_evol['pq_func_par%i[10]' % j] = 'pq[%i]' % i
    fobsc_evol['pq_func[%i]' % i] = 'pl'
    fobsc_evol['pq_func_var[%i]' % i] = '1+z'
    fobsc_evol['pq_func_par0[%i]' % i] = fobsc['pq_func_par%i[10]' % j]
    fobsc_evol['pq_func_par1[%i]' % i] = 7.
    fobsc_evol['pq_func_par2[%i]' % i] = 0.

"""
Saves some disk space since redshift is irrelevant for scaling laws.
"""
_const_blob_n1 = ['lf']                         # function of MUV
_const_blob_n2 = ['fstar', 'fsmooth', 'fobsc']  # function of halo mass, not z
_const_blob_n3 = ['smf']                        # function of stellar mass
_const_blob_n4 = ['SFR', 'Mgas', 'MZ', 'Mstell']# can depend on redshift
_const_blob_i1 = [('z', np.array([3, 3.8, 4, 4.9, 5, 5.9, 6, 6.9, 7, 7.9, 8, 9, 10, 10.4, 11, 12, 15])),
    ('x', np.arange(-27, -8.8, 0.2))]
_const_blob_i2 = [('Mh', 10**np.arange(5., 13.6, 0.1))]
_const_blob_i3 = [('z', np.array([3, 3.8, 4, 4.9, 5, 5.9, 6, 6.9, 7, 7.9, 8, 9, 10, 10.4, 11, 12, 15])),
    ('M', 10**np.arange(5., 13.6, 0.1))]
_const_blob_i4 = [('z', np.array([3, 3.8, 4, 4.9, 5, 5.9, 6, 6.9, 7, 7.9, 8, 9, 10, 10.4, 11, 12, 15])),
    ('Mh', 10**np.arange(5., 13.6, 0.1))]    
_const_blob_f1 = ['LuminosityFunction']
_const_blob_f2 = ['fstar', 'fsmooth', 'fobsc']
_const_blob_f3 = ['StellarMassFunction']
_const_blob_f4 = ['SFR', 'GasMass', 'MetalMass', 'StellarMass']

# Case where everything evolves with z
_zevol_blob_n1 = ['lf']                                               # function of MUV
_zevol_blob_n2 = ['fstar', 'fsmooth', 'fobsc', 'SFR', 'Mgas', 'MZ', 'Mstell']  # function of halo mass
_zevol_blob_n3 = ['smf']                                              # function of stellar mass
_zevol_blob_i1 = [('z', np.array([3, 3.8, 4, 4.9, 5, 5.9, 6, 6.9, 7, 7.9, 8, 9, 10, 10.4, 11, 12, 15])),
    ('x', np.arange(-27, -8.8, 0.2))]
_zevol_blob_i2 = [('z', np.array([3, 3.8, 4, 4.9, 5, 5.9, 6, 6.9, 7, 7.9, 8, 9, 10, 10.4, 11, 12, 15])),
    ('Mh', 10**np.arange(5., 13.6, 0.1))]
_zevol_blob_i3 = [('z', np.array([3, 3.8, 4, 4.9, 5, 5.9, 6, 6.9, 7, 7.9, 8, 9, 10, 10.4, 11, 12, 15])),
    ('M', 10**np.arange(5., 13.6, 0.1))]
_zevol_blob_f1 = ['LuminosityFunction']
_zevol_blob_f2 = ['fstar', 'fsmooth', 'fobsc', 'SFR', 'GasMass', 'MetalMass', 'StellarMass']
_zevol_blob_f3 = ['StellarMassFunction']

sfe_const = \
{ 
 'blob_names': [_const_blob_n1, _const_blob_n2, _const_blob_n3, _const_blob_n4],
 'blob_ivars': [_const_blob_i1, _const_blob_i2, _const_blob_i3, _const_blob_i4],
 'blob_funcs': [_const_blob_f1, _const_blob_f2, _const_blob_f3, _const_blob_f4],
}

sfe_zevol = \
{ 
 'blob_names': [_zevol_blob_n1, _zevol_blob_n2, _zevol_blob_n3],
 'blob_ivars': [_zevol_blob_i1, _zevol_blob_i2, _zevol_blob_i3],
 'blob_funcs': [_zevol_blob_f1, _zevol_blob_f2, _zevol_blob_f3],
}

lfonly = \
{ 
 'blob_names': [_const_blob_n1],
 'blob_ivars': [_const_blob_i1],
 'blob_funcs': [_const_blob_f1],
}

smfonly = \
{ 
 'blob_names': [_zevol_blob_n2],
 'blob_ivars': [_zevol_blob_i2],
 'blob_funcs': [_zevol_blob_f2],
}

ddpl = \
{
 'pq_func[0]': 'ddpl',
 'pq_func_par4[0]': 5e-1, 
 'pq_func_par5[0]': 5e11,
 'pq_func_par6[0]': 1.,
 'pq_func_par7[0]': -0.5,
}

