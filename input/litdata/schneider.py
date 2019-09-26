import sys
import numpy as np
from ares.util import ParameterBundle
from scipy.interpolate import interp1d
from ares.populations import GalaxyPopulation

try:
    import h5py
except ImportError:
    pass

try:
    import genmassfct as gmf
    from classy import Class
except ImportError:
    sys.exit() 

# Grab a vanilla ST mass function. dndm has shape (z, M)
pop = GalaxyPopulation()
tab_z, tab_M, tab_dndm = pop.halos.tab_z, pop.halos.tab_M, pop.halos.tab_dndm
#

zmin = tab_z.min()
zmax = tab_z.max()
Nz = tab_z.size

tiny_dndm = 1e-60

def get_ps(**kwargs):
    
    Om = kwargs['omega_m_0']
    Ob = kwargs['omega_b_0']
    s8 = kwargs['sigma_8']
    h0 = kwargs['hubble_0']
    ns = kwargs['primordial_index']
    
    Omega_ncdm = Om - Ob
    #m_ncdm = 77319.85488

    T_ncdm = 0.715985
    
    mTH = kwargs['hmf_extra_par1']
    
    #Bozek2015
    msn = 3.9*(mTH)**1.294*(0.1198/0.1225)**(-0.33333)*1000
    
    params = {
         'h': h0,
         'T_cmb': 2.726,
         'Omega_b': Ob,
         'Omega_cdm': 1e-10,
         'Omega_ncdm': Omega_ncdm,
         'N_eff': 3.04,
         'N_ncdm': 1,
         'm_ncdm': msn,
         #'m_ncdm_in_eV': msn * 1e3,
         'T_ncdm': 0.715985, # ?
         'n_s': ns,
         #'sigma8': s8,
         'A_s': 2.02539e-09,
         'P_k_max_h/Mpc': 500.,         # Shouldn't need to be more than ~500? 
         'k_per_decade_for_pk': 20,     # Shouldn't need to be more than ~20? 
         'output': 'mPk',
         'ncdm_fluid_approximation': 3,
         'use_ncdm_psd_files': 0,
         # tolerances
         'tol_ncdm_bg': 1e-3,
         #'tol_ncdm': 1e-3,
         
         'tol_perturb_integration': 1e-6,
         'perturb_sampling_stepsize': 0.01,
         
         'format': 'camb',
                  
    }
        
    classinst = Class()
    classinst.set(params)
    classinst.compute()
    
    #lo = -2
    #hi = np.log10(params['P_k_max_h/Mpc'] * h0)
    #Nk = (hi - lo) * params['k_per_decade_for_pk']
    #k_bin = 10**np.linspace(lo, hi, Nk+1)
    k_bin = np.logspace(-5, 2.5, 500)
    #k_bin = np.logspace(-2, 1.7, 100)

    pk_bin = np.array([classinst.pk_lin(k_bin[i],0) for i in range(len(k_bin))])
    
    hcorr = True
    
    #if hcorr:
    return k_bin / h0, pk_bin * h0**3
    #else:
    #    return k_bin, pk_bin

def hmf_wrapper(**kwargs):
    """
    Compute the HMF from some kwargs. In this case, just truncate below
    some mass threshold and return modified HMF.
    
    In practice this will be a wrapper around some other function. We just
    need to rename parameters so ARES understands them, hence the 
    `hmf_extra_par0` parameter below, which is serving as our mass cutoff.
    """
    
    assert 'hmf_extra_par0' in kwargs
    
    # 'hmf_extra_par0' = psfct
    # 'hmf_extra_par0' = mf_table
    
    #ok = tab_M > kwargs['hmf_extra_par0']
    
    # Use Aurel's code to compute dndm
    par = gmf.par()
    
    # Should generate filename from hmf_extra_par? parameters.
    
    # 0 = filename supplied directly
    # 1 = WDM mass supplied, PS generated on-the-fly    
    par0 = kwargs['hmf_extra_par0']
    par1 = kwargs['hmf_extra_par1']
    
    if type(par0) is str:
        # Interpolate from a grid
        if par0.endswith('hdf5'):
            f = h5py.File(par0, 'r')
            zvals = np.array(f[('tab_z')])
            mxvals = np.array(f[('m_x')])
            dndm_all = np.array(f[('tab_dndm')])
            f.close()
            
            # Could interpolate at some point.
            
            i1 = np.argmin(np.abs(mxvals - par1))
            
            if mxvals[i1] > par1:
                i1 -= 1
                
            i2 = i1 + 1
                
            dndm_lo = dndm_all[i1]
            dndm_hi = dndm_all[i2]
            
            func = interp1d(mxvals[i1:i1+2], np.array([dndm_lo, dndm_hi]),
                axis=0)
            
            dndm = func(par1)
            #dndm = np.zeros_like(dndm_all[0])
            #for i, red in enumerate(zvals):   
            #    dndm[i] = np.interp(par1, mxvals[i1:i1+2], [dndm_lo[i], dndm_hi[i]])    
            #
            return dndm#_all[i]
            
        else:    
            fn = par0
    else:
        fn = 'wdm_model.txt'
    
    k, ps = get_ps(**kwargs)
    np.savetxt(fn, np.array([k, ps]).T)
        
    # Re-generate this on-the-fly?
    par.file.psfct = fn
    #"../../code/files/CDM_Planck15_pk.dat"
    #par.file.mf_table = kwargs['hmf_extra_par1']
    #"mfCDM_table.npz" 

    #par.file.psfct = "files/WDM_Planck15_m4.0_pk.dat"
    #par.file.mf_table = "mfWDM4_table.npz"

    par.code.window = kwargs['hmf_window']
    par.code.Nrbin = 100
    par.code.rmin  = 0.002
    par.code.rmax  = 25.
    
    par.cosmo.zmin = zmin
    par.cosmo.zmax = zmax
    par.cosmo.Nz   = Nz
    #par.rhoc       = 
    par.Om         = kwargs['omega_m_0']
    par.Ob         = kwargs['omega_b_0']
    par.s8         = kwargs['sigma_8']
    par.h0         = kwargs['hubble_0']
    par.ns         = kwargs['primordial_index']
    
    par.mf.q       = 0.707 if par.code.window == 'tophat' \
        else 1
    
    h = par.cosmo.h0

    m, _dndlnm = gmf.dndlnm(par)
    dndlnm = np.array(_dndlnm)
    dndm = dndlnm / m
        
    # Need to interpolate back to ARES mass range.
    new_dndm = np.zeros_like(tab_dndm)
    for i, z in enumerate(tab_z):
        new_dndm[i,:] = np.interp(np.log10(tab_M), np.log10(m / h), 
            np.log10(dndm[i,:] * h**4), left=-np.inf, right=-np.inf)
        
    return 10**new_dndm
    
    
popIII_uv = ParameterBundle('pop:fcoll') \
          + ParameterBundle('sed:uv')
popIII_uv['pop_Tmin'] = 500.     # Restrict PopIII to molecular cooling halos
popIII_uv['pop_Tmax'] = 1e4      # Restrict PopIII to molecular cooling halos
popIII_uv['pop_fstar'] = 1e-4    # Will be free parameter
popIII_uv.num = 2                # Population #2

# Add PopIII X-ray emission. Assume "MCD" spectrum and evolve the X-ray
# background properly. Attach ID number 3.
popIII_xr = ParameterBundle('pop:fcoll') \
          + ParameterBundle('sed:mcd') \
          + ParameterBundle('physics:xrb')
popIII_xr['pop_sfr_model'] = 'link:sfrd:2'
popIII_xr.num = 3

popIII_basic = popIII_uv + popIII_xr
    
    
