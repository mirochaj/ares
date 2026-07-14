"""

PreProcessing.py

Author: Jordan Mirocha
Affiliation: Caltech
Created on: Thu Mar 26 10:13:30 2026

Description:

"""

import gc
import os
import sys
import time
import pickle
import numpy as np
from ..sources import Galaxy
from . import ParameterBundle
from itertools import product
from ..simulations import Simulation
from ..physics.Constants import s_per_myr

try:
    import h5py
except ImportError:
    pass

try:
    from multiprocess import Pool, current_process
except ImportError:
    pass

try:
    from schwimmbad import MPIPool
except ImportError:
    pass

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1

class DummyPool(object):
    def __init__(self, processes=1):
        pass 
    def map(self, func, params):
        return [func(param) for param in params]
    def close(self):
        pass

sfh_options = ['exp_decl', 'exp_rise', 'const', 'exp_decl_trunc', 'fail']

def load_checkpoint(fn, x, verbose=False):

    try:
        with open(fn, 'rb') as f:
            x, sfr, (m_rec, sfr_rec), sfh, tau = pickle.load(f)
    except EOFError:
        print(f"Failed to open {fn}. Will re-generate.")
        return None
    #try:
    #    waves, lum = np.loadtxt(fn, unpack=True)
    #except ValueError:
    #    print(f'Failed to load {fn}.')
    #    raise ValueError(f'Failed to load {fn}')
#
    #f = open(fn, 'r')
    #hdr = f.readline()[1:].split(';')
    #sfh = hdr[0].strip().split('=')[-1]
    #sfh_num = sfh_options.index(sfh)
    #tau = float(hdr[1].strip().split('=')[-1])
    #sfr = float(hdr[2].strip().split('=')[-1])
    #m_rec = float(hdr[3].strip().split('=')[-1])
    #sfr_rec = float(hdr[4].strip().split('=')[-1])
    #f.close()

    if verbose:
        print(f"! Loaded {fn}.")

    return x, sfr, (m_rec, sfr_rec), sfh, tau
    
def get_checkpoint_dir(output_dir):
    return f'{output_dir}/sed_corr'

def get_checkpoint_fn(x, pop_idnum, output_dir, spec=False):
    z, _mass_ = x

    if z < 0.1:
        fn = f'corr_z_{z:.4f}_m_{_mass_:.2f}_pop_{pop_idnum}'
    else:
        fn = f'corr_z_{z:.2f}_m_{_mass_:.2f}_pop_{pop_idnum}'
    
    if spec:
        fn += '_spec'
    else:
        fn += '_info'
        
    fn += '.pkl'

    fn_out = f'{get_checkpoint_dir(output_dir)}/{fn}'

    return fn_out 

def get_sfh_params(x, pop_ms, pop, pop_small_dt, pars_g, output_dir, mtol=1e-2,
    sfh_model_override=None, tau_guess=1e3, clobber_checkpoints=0,
    debug=False):
    """
    Determine the SFH parameters for a galaxy with given properties `x`.

    Parameters
    ----------
    x : tuple, list, np.ndarray 
        For now, just two elements: redshift, stellar mass / Msun

    Note
    ----
    The input mass is assumed to be the OBSERVED mass.    

    """ 

    galaxy = Galaxy(**pars_g)

    # _mass_ is really log10(stellar mass / Msun)
    z, _mass_ = x

    fn_out = get_checkpoint_fn(x, pop.id_num, output_dir, spec=0)

    if (not clobber_checkpoints) and os.path.exists(fn_out):
        
        result = load_checkpoint(fn_out, x)

        if debug:
            print(f'loaded {fn_out}')
    
        if result is not None:
            return result
        
        
    if debug:
        print(f"Will generate {fn_out}...")

    t = pop.cosm.t_of_z(z) / s_per_myr

    mass_obs = 10**_mass_
    mass_sys = pop.get_mstell_sys(z=z, Mh=None)
    mass_true = 10**(_mass_ - mass_sys)

    # Get appropriate SFR for such an object
    #ssfr_true = pop.get_ssfr(z, mass_true)
    #ssfr_sys = pop.get_ssfr_sys(z=z)
    #ssfr_obs = 10**(np.log10(ssfr_true) + ssfr_sys)

    pop_idnum = pop.id_num

    smhm = pop.get_smhm(z=z, Mh=pop.halos.tab_M)
    
    if pop_idnum == 1:
        sfr_all_halos = pop_ms.get_sfr(z=z, Mh=pop.halos.tab_M)
        corr = pop.pf['pop_sfr_below_ms']
    else:
        sfr_all_halos = pop.get_sfr(z=z, Mh=pop.halos.tab_M)
        corr = 1.

    sfr_true = np.interp(mass_true, smhm * pop.halos.tab_M, sfr_all_halos)
    sfr_true /= corr

    sfr_sys = pop.get_sfr_sys(z=z, Mh=None)
    sfr_obs = 10**(np.log10(sfr_true) + sfr_sys)

    mass_use = mass_true
    sfr_use = sfr_true

    # [optional] assume t0 is some user-defined value
    kw_t0 = {}

    if sfh_model_override is not None:
        sfh_model = sfh_model_override
    else:
        sfh_model = 'exp_decl'

    # Do the real work: get parameters of SFH that produce stellar mass 
    # `mass_use` and SFR `sfr_use` at redshift `z` (or time `t`).
    # Note: xtol=1e-4 corresponds to century-level convergence in tau or t0
    #       ftol refers to acceptable convergence in np.log10(mass_in / mass_out)
    #       so if we've converged to 1e-4 we're well below mtol=0.01.

    sfh_kw = galaxy.get_kwargs(t, mass_use, sfr_use,
        sfh=sfh_model,
        tau_guess=tau_guess, 
        xtol=1e-4, ftol=1e-4, mtol=mtol,
        mass_return=True, disp=0, 
        **kw_t0)
            
    # Save recovered mass and SFR for print-out and convergence check.        
    m_rec = sfh_kw['mass_obs']
    sfr_rec = sfh_kw['sfr_obs']

    if debug:
        print("!"*40)
        print(f"*"*80)
        print(f"! Found tau={sfh_kw['tau']:.2e} in {t2-t1:.1f} sec")
        print(f"*"*80)
        print(f"! Check on z={z:.4f}, log10(Mstell)={np.log10(mass_use):.4f}")
        print(f"! Recovered mass is {np.log10(m_rec):.4f}")
        print(f"! Check on z={z:.4f}, log10(SFR)={np.log10(sfr_use):.4f}")
        print(f"! Recovered SFR is {np.log10(sfr_rec):.4f}")
        print("!"*40)

        print(sfh_kw)

        if (np.isinf(m_rec) or np.isnan(m_rec)) or (np.isinf(sfr_rec) or np.isnan(sfr_rec)):
            import matplotlib.pyplot as plt 

            t_hr = pop_small_dt.halos.tab_t
            sfh_hr = galaxy.get_sfr(t_hr, tobs=t, **sfh_kw)
    
            plt.plot(t_hr, sfh_hr)

            print(sfh_hr)

            input('<enter>')

    ##
    # 
    converged = (np.abs(np.log10(mass_use/m_rec)) <= mtol) \
        and (np.abs(np.log10(sfr_use/sfr_rec)) <= mtol)
    
    if not converged:
        print(f"! Not converged: dMst={np.abs(np.log10(mass_use/m_rec)):.4f}, dSFR={np.abs(np.log10(sfr_use/sfr_rec)):.4f}")
        print(f"! sfr_rec={sfr_rec}, sfr_use={sfr_use}")

    with open(fn_out, 'wb') as f:
        pickle.dump((x, sfr_use, (m_rec, sfr_rec), sfh_kw, converged), f)

    if debug:
        print(f"! Saved {fn_out}")

    return x, sfr_use, (m_rec, sfr_rec), sfh_kw, converged

def generate_sed(sfh_results, pop, pop_small_dt, pars_g, output_dir, waves):

    x, sfr_use, (m_rec, sfr_rec), sfh_kw, converged = sfh_results
    z, m = x 

    galaxy = Galaxy(**pars_g)

    fn_out_spec = get_checkpoint_fn(x, pop.id_num, output_dir, spec=1)
    if os.path.exists(fn_out_spec):
        try:
            with open(fn_out_spec, 'rb') as f:
                waves, spec = pickle.load(f)
            return x, waves, spec
        except EOFError:
            print(f"Error opening {fn_out_spec}. Will re-generate.")
            pass

    # Switch to Myr time resolution for low-mass galaxies
    t_hr = np.arange(pop_small_dt.halos.tab_t.min(), 
        pop_small_dt.halos.tab_t.max() + 1, 1)
    
    # Synthesize SFH   
    t = pop.cosm.t_of_z(z) / s_per_myr
    sfh_hr = galaxy.get_sfr(t_hr, tobs=t, **sfh_kw)
    # Get spectrum
    spec = galaxy.get_spec(z, t=t_hr, sfh=sfh_hr, waves=waves, hist={})
    # Save
    with open(fn_out_spec, 'wb') as f:
        pickle.dump((waves, spec), f)
    print(f"Wrote {fn_out_spec}.")

    del galaxy, t, sfh_hr
    gc.collect()

    return x, waves, spec
    
def generate_sed_tab(base_kwargs, output_dir, pop_idnum, 
    mtol=1e-2, mtol_num=3e-1,
    dlam=10, lam_min=900, lam_max=5e4,
    clobber_checkpoints=0, clobber_final_database=0, 
    use_multiprocess=1, nthreads=1):
    """
    Generate
    """

    fn_out_final = f'{output_dir}/sedtab_pop_{pop_idnum}.hdf5'

    if os.path.exists(fn_out_final) and (not clobber_final_database):
        print(f"Found {fn_out_final}, will load since clobber_final_database=0.")
        return fn_out_final

    if (nthreads > 1) or (size > 1):
        if use_multiprocess:
            is_root = current_process().name == 'MainProcess'
            JobPool = Pool
        else:
            JobPool = MPIPool
            is_root = rank == 0
    else:
        is_root = 1
        JobPool = DummyPool
        
    # 
    waves = np.arange(lam_min, lam_max + dlam, dlam)
    
    # Key 2-D parameter space
    mass_bins = np.arange(2, 12.6, 0.05)
    zbins = 10**np.arange(-2, 1.5, 0.05)

    num_seds = mass_bins.size * zbins.size
        
    # Setup pars
    #############################################################################
    pars = base_kwargs    
    pars['verbose'] = False
    
    # Use this to get (SFR, Mstell) relations over z
    sim_base = Simulation(**pars)
    
    # Just need this to grab high time res grid
    pars_small_dt = base_kwargs.copy()
    pars_small_dt.update(ParameterBundle('mirocha2025:slow'))
    pars_small_dt['verbose'] = False
    
    sim_small_dt = Simulation(**pars_small_dt)
        
    # This is for the ares.sources.Galaxy instance that figures out 
    # SFHs for us and does spectral synthesis
    pars_g = {}
    pars_g['source_aging'] = True
    pars_g['source_ssp'] = True
    pars_g['source_sed_degrade'] = None
    pars_g['source_sed'] = pars['pop_sed{0}']
    pars_g['source_imf'] = 'chabrier'
    pars_g['source_tracks'] = 'Padova1994'
    pars_g['source_Z'] = 0.02
    
    ##
    # How to model SFH? Generalize in future.
    pars_g['source_sfh'] = 'exp_decl'
    
    # Always start with exp_decl, then try exp_rise (if star-forming), but
    # ultimately use a constant SFH if those fail.
    pars_g['source_sfh_fallback_last_resort'] = True
    pars_g['source_sfh_fallback'] = 'const' if pop_idnum == 1 else 'exp_rise'
    pars_g['verbose'] = False
    
    # Need pop instances to retrieve target mass and SFR, time/redshift arrays, etc.
    # Don't freak out that we're not indexing with `pop_idnum`. We don't need 
    # galaxy:halo scaling relations. We're just computing stuff on a grid of stellar 
    # mass and redshift -- how that M_stell relates to galaxies happens outside this scope.
    # Would need to update this if SFGs and QGs had different obs/true systematics, 
    pop = sim_base.pops[pop_idnum]
    pop_small_dt = sim_small_dt.pops[0]

    if pop_idnum == 1:
        id_bb = pop.pf['pop_sfr_below_ms_of_pop']
        pop_ms = sim_base.pops[id_bb]
    else:
        pop_ms = pop

    assert zbins.min() >= sim_base.pops[0].halos.tab_z.min()
    
    ##
    # Setup output data structure
    corr_all = -np.inf * np.ones((len(zbins), len(mass_bins), len(waves)))
    
    lum_all = corr_all.copy()
    
    # Store best-fit pars
    pars_all = -np.inf * np.ones((len(zbins), len(mass_bins), 2))
    sfh_all = -np.inf * np.ones((len(zbins), len(mass_bins), 1))
    tau_all = -np.inf * np.ones((len(zbins), len(mass_bins), 1))
    sfr_all = -np.inf * np.ones((len(zbins), len(mass_bins)))
    
    sfh_options = ['exp_decl', 'exp_rise', 'const', 'exp_decl_trunc', 'fail']
    
    # Construct all possible combinations of (z, Ms)
    all_params = [element for element in product(zbins, mass_bins[-1::-1])]
    
    ##
    # Setup output directory
    if (not os.path.exists(get_checkpoint_dir(output_dir))) and is_root:
        os.mkdir(get_checkpoint_dir(output_dir))
    
    if is_root:
        print(f"! SED table will have {num_seds} elements.")
    
    ##
    # Run it
    if is_root:
        print(f"! Generating SFHs for pop={pop_idnum}...")
    
    t1 = time.time()

    ## 
    # Setup the appropriate pool
    if use_multiprocess and nthreads > 1:
        p = JobPool(processes=nthreads, maxtasksperchild=50)
    elif (size > 1):
        assert not use_multiprocess
        p = JobPool(use_dill=1)
        #if not p.is_master():
        #    p.wait()
        #    sys.exit(0)
            
    else:
        p = JobPool()

    def sfh_func(y):
        return get_sfh_params(y, pop_ms, pop, pop_small_dt, 
            pars_g, output_dir, mtol=mtol)
    
    all_results = list(p.map(sfh_func, all_params))
    
    t2 = time.time()
    
    if is_root:
        print(f"! Done getting SFH kwargs in {t2-t1:.2f} sec. Time to generate SEDs")

    ##
    # Generate SEDs
    if is_root:
        print(f"! Generating SEDs for pop={pop_idnum} using {size} threads...")
    
    t1 = time.time()
    
    def sed_func(y):
        return generate_sed(
            y, pop, pop_small_dt, pars_g, output_dir, waves)
    
    all_seds = list(p.map(sed_func, all_results))

    p.close()
    
    t2 = time.time()
    
    if is_root:
        print(f"! Done getting SEDs in {t2-t1:.2f} sec.")

    ##
    # Save a file with the whole parameter space
    corr_all = -np.inf * np.ones((len(zbins), len(mass_bins), len(waves)))
    lum_all = -np.inf * np.ones((len(zbins), len(mass_bins), len(waves)))

    # Store best-fit pars
    pars_all = -np.inf * np.ones((len(zbins), len(mass_bins), 2))
    sfh_all = -np.inf * np.ones((len(zbins), len(mass_bins)))
    tau_all = -np.inf * np.ones((len(zbins), len(mass_bins)))

    mrec_all = -np.inf * np.ones((len(zbins), len(mass_bins)))
    sfrrec_all = -np.inf * np.ones((len(zbins), len(mass_bins)))

    ##
    # By here, everything should have run already.
    # No need to read back in from disk -- everything we need is
    # stored in all_results and all_seds

    ##
    # The reason we do two separate loops here is because 
    # there's no guarantee that elements of all_results and
    # all_spec will match given that we're farming out the 
    # work to different threads.
    for result in all_results:
        if result is None:
            continue
        
        x = result[0]
        z, m = x 
        iz = np.argmin(np.abs(z - zbins))
        iM = np.argmin(np.abs(m - mass_bins))

        x, sfr, (m_rec, sfr_rec), sfh_kw, converged = result

        ##
        # Save stuff
        sfh_all[iz,iM] = sfh_options.index(sfh_kw['sfh'])
        sfr_all[iz,iM] = sfr
        tau_all[iz,iM] = sfh_kw['tau']
    
        mrec_all[iz,iM] = m_rec
        sfrrec_all[iz,iM] = sfr_rec
    
    
    for result in all_seds:
    
        if result is None:
            continue
        
        x = result[0]
        z, m = x 

        iz = np.argmin(np.abs(z - zbins))
        iM = np.argmin(np.abs(m - mass_bins))

        # Load
        #result = get_sfh_params(x, sim_base, pop, 
        #    pop_small_dt, galaxy, output_dir, mtol=mtol)
            
        #x, sfr, (m_rec, sfr_rec), sfh_kw, converged = result
#
        #fn_out_spec = get_checkpoint_fn(x, pop_idnum, output_dir, spec=1)
#
        ##if os.path.exists(fn_out_spec):
        #with open(fn_out_spec, 'rb') as f:
        #    waves, spec = pickle.load(f)
        #print(f"! Loaded {fn_out_spec}.")
            
        # result is (x, waves, spectrum)
        lum_all[iz,iM,:] = result[-1].copy()
    
    failed = sfh_all == 4
    
    # Save table for use in modeling
    if os.path.exists(fn_out_final) and (not clobber_final_database):
        sys.exit(0)
    
    with h5py.File(fn_out_final, 'w') as f:
        f.create_dataset('z', data=zbins)
        f.create_dataset('Ms', data=mass_bins)
        f.create_dataset('SFR', data=sfr_all)
        f.create_dataset('Ms_rec', data=mrec_all)
        f.create_dataset('SFR_rec', data=sfrrec_all)
        f.create_dataset('waves', data=waves)
        f.create_dataset('lum', data=lum_all)
        f.create_dataset('tau', data=tau_all)
        f.create_dataset('sfh', data=sfh_all)
    
    print(f"Wrote {fn_out_final}.")

    return fn_out_final
    
    