"""

test_inference_lf.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Wed 25 Mar 2020 11:01:32 EDT

Description:

"""

import os
import glob
import ares
import numpy as np
from ares.physics.Constants import rhodot_cgs

def test():
    # Will save UVLF at these redshifts and magnitudes
    redshifts = np.array([3, 3.8, 4, 4.9, 5, 5.9, 6, 6.9, 7, 7.9, 8])
    MUV = np.arange(-28, -8.8, 0.2)

    fit_z = [6]

    # blob 1: the LF. Give it a name, and the function needed to calculate it.
    blob_n1 = ['galaxy_lf']
    blob_i1 = [('z', redshifts), ('bins', MUV)]
    blob_f1 = ['get_lf']

    blob_pars = \
    {
     'blob_names': [blob_n1],
     'blob_ivars': [blob_i1],
     'blob_funcs': [blob_f1],
     'blob_kwargs': [None],
    }

    # Do a Schechter function fit just for speed
    base_pars = \
    {
     'pop_sfr_model': 'uvlf',

     # Stellar pop + fesc
     'pop_calib_wave': 1600.,
     'pop_lum_per_sfr': 0.2e28, # to avoid using synthesis models

     'pop_uvlf': 'pq',
     'pq_func': 'schechter_evol',
     'pq_func_var': 'MUV',
     'pq_func_var2': 'z',

     # Bouwens+ 2015 Table 6 for z=5.9
     #'pq_func_par0[0]': 0.39e-3,
     #'pq_func_par1[0]': -21.1,
     #'pq_func_par2[0]': -1.90,
     #
     # phi_star
     'pq_func_par0': np.log10(0.47e-3),

     # z-pivot
     'pq_func_par3': 6.,

     # Mstar
     'pq_func_par1': -20.95,

     # alpha
     'pq_func_par2': -1.87,

     'pq_func_par4': -0.27,
     'pq_func_par5': 0.01,
     'pq_func_par6': -0.1,


    }

    base_pars.update(blob_pars)

    free_pars = \
    [
     'pq_func_par0',
     'pq_func_par1',
     'pq_func_par2',
    ]

    is_log = [False, False, False]

    from distpy import DistributionSet
    from distpy import UniformDistribution

    ps = DistributionSet()
    ps.add_distribution(UniformDistribution(-5, -1),  'pq_func_par0')
    ps.add_distribution(UniformDistribution(-25, -15),'pq_func_par1')
    ps.add_distribution(UniformDistribution(-3, 0),   'pq_func_par2')

    guesses = \
    {
     'pq_func_par0': -3,
     'pq_func_par1': -22.,
     'pq_func_par2': -2.,
    }

    if len(fit_z) > 1:
        free_pars.extend(['pq_func_par4', 'pq_func_par5', 'pq_func_par6'])
        is_log.extend([False]*3)
        guesses['pq_func_par4'] = -0.3
        guesses['pq_func_par5'] = 0.01
        guesses['pq_func_par6'] = 0.

        ps.add_distribution(UniformDistribution(-2, 2), 'pq_func_par4')
        ps.add_distribution(UniformDistribution(-2, 2), 'pq_func_par5')
        ps.add_distribution(UniformDistribution(-2, 2), 'pq_func_par6')


    # Test error-handling
    for ztol in [0, 0.3]:
        # Initialize a fitter object and give it the data to be fit
        fitter_lf = ares.inference.FitGalaxyPopulation(**base_pars)

        # The data can also be provided more explicitly
        fitter_lf.ztol = ztol
        fitter_lf.redshifts = {'lf': fit_z}

        if ztol == 0:
            try:
                fitter_lf.data = 'bouwens2015'
            except ValueError:
                print("Correctly caught error! Moving on.")
                continue
        else:
            # This should would if ztol >= 0.1, so we want this to crash
            # visibly if there's a failure internally.
            fitter_lf.data = 'bouwens2015'

        fitz_s = 'z_'
        for red in np.sort(fit_z):
            fitz_s += str(int(round(red)))

        fitter = ares.inference.ModelFit(**base_pars)
        fitter.add_fitter(fitter_lf)

        # Establish the object to which we'll pass parameters
        from ares.populations.GalaxyCohort import GalaxyCohort
        fitter.simulator = GalaxyCohort

        fitter.parameters = free_pars
        fitter.is_log = is_log
        fitter.prior_set = ps

        # In general, the more the merrier (~hundreds)
        fitter.nwalkers = 2 * len(fitter.parameters)

        fitter.jitter = [0.1] * len(fitter.parameters)
        fitter.guesses = guesses

        # Run the thing
        fitter.run('test_uvlf', burn=10, steps=10, save_freq=10,
            clobber=True, restart=False)

        # Make sure things make sense
        anl = ares.analysis.ModelSet('test_uvlf')

        # Other random stuff
        all_kwargs = anl.AssembleParametersList(include_bkw=True)
        assert len(all_kwargs) == anl.chain.shape[0]

        iML = np.argmax(anl.logL)
        best_pars = anl.max_likelihood_parameters()

        for i, par in enumerate(best_pars.keys()):
            assert all_kwargs[iML][par] == best_pars[par]

    # Clean-up
    mcmc_files = glob.glob('{}/test_uvlf*'.format(os.environ.get('ARES')))

    # Iterate over the list of filepaths & remove each file.
    for fn in mcmc_files:
        try:
            os.remove(fn)
        except:
            print("Error while deleting file : ", filePath)

    assert True

if __name__ == '__main__':
    test()
