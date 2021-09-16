"""

test_inference_gs.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Wed 25 Mar 2020 09:41:20 EDT

Description:

"""

import os
import ares
import glob
import numpy as np

def test():

    # These go to every calculation
    zblobs = np.arange(6, 31)

    base_pars = \
    {
     'problem_type': 101,
     'tanh_model': True,
     'blob_names': [['tau_e', 'z_C', 'dTb_C'], ['cgm_h_2', 'igm_Tk', 'dTb']],
     'blob_ivars': [None, [('z', zblobs)]],
     'blob_funcs': None,
    }

    for i in range(3):

        fitter_gs = ares.inference.FitGlobal21cm()

        fitter_gs.checkpoint_append = min(i, 1)
        fitter_gs.frequencies = freq = np.arange(40, 200) # MHz
        fitter_gs.data = -100 * np.exp(-(80. - freq)**2 / 2. / 20.**2)

        # Set errors
        fitter_gs.error = 20. # flat 20 mK error

        fitter = ares.inference.ModelFit(**base_pars)
        fitter.add_fitter(fitter_gs)
        fitter.simulator = ares.simulations.Global21cm

        fitter.parameters = ['tanh_J0', 'tanh_Jz0', 'tanh_Jdz', 'tanh_Tz0', 'tanh_Tdz']
        fitter.is_log = [True] + [False] * 4

        from distpy import DistributionSet
        from distpy import UniformDistribution

        ps = DistributionSet()
        ps.add_distribution(UniformDistribution(-3, 3), 'tanh_J0')
        ps.add_distribution(UniformDistribution(5, 20), 'tanh_Jz0')
        ps.add_distribution(UniformDistribution(0.1, 20), 'tanh_Jdz')
        ps.add_distribution(UniformDistribution(5, 20), 'tanh_Tz0')
        ps.add_distribution(UniformDistribution(0.1, 20), 'tanh_Tdz')

        fitter.prior_set = ps
        fitter.jitter = [0.1] * len(fitter.parameters)

        fitter.nwalkers = 2 * len(fitter.parameters)

        nsteps = 5
        # Do a quick burn-in and then run for 50 steps (per walker)
        fitter.run(prefix='test_tanh', burn=nsteps, steps=nsteps, save_freq=1,
            clobber=i<2, restart=i==2)

        anl = ares.analysis.ModelSet('test_tanh')

        # Read-in some attributes
        assert anl.nwalkers == 10
        #assert anl.priors != {}
        assert anl.is_mcmc
        assert anl.Nd == len(fitter.parameters)
        assert np.isfinite(np.nanmean(anl.logL))
        assert anl.steps == nsteps
        priors = anl.priors
        assert np.all(np.array(anl.is_log) == np.array(fitter.is_log))

        logL = anl.logL

        # Be generous
        assert 0.05 <= np.mean(anl.facc) <= 0.95, np.mean(anl.facc)

        checkpts = anl.checkpoints

        unique = anl.unique_samples
        assert len(unique) == anl.chain.shape[1]
        assert np.all(np.array([len(elem) for elem in unique]) <= anl.chain.shape[0])
        # Make sure walkers are moving
        for j in range(len(anl.parameters)):
            assert np.unique(np.diff(anl.chain[:,j])).size > 1

        # Isolate walker, check shape etc.
        w0, logL, flags = anl.get_walker(0)

        assert w0.shape[0] == nsteps * (1 + int(i==2)), \
            "Shape wrong {}".format(w0.shape[0])
        assert w0.shape[1] == len(fitter.parameters)

        # Make sure skipping elements produces a dataset of the right (reduced)
        # size.
        shape_all = anl.chain.shape
        anl.skip = 2
        shape_good = (anl.chain.shape[0]-2, anl.chain.shape[1])
        assert anl.chain[anl.mask==False].size == np.product(shape_good)

        # Extract some error-bars
        mu, (lo, hi) = anl.get_1d_error(anl.parameters[0])

        # Compare mu to max likelihood value?
        best_pars = anl.max_likelihood_parameters()

        cov = anl.CovarianceMatrix(anl.parameters)

        # Grab some blobs, check shape
        data = anl.ExtractData(['dTb', 'igm_Tk', 'tau_e', anl.parameters[0]])
        assert data[anl.parameters[0]].shape == (anl.chain.shape[0],)
        assert data['dTb'].shape == (anl.chain.shape[0], zblobs.size)
        # Didn't vary any ionization parameters so tau_e shouldn't change.
        assert np.all(np.diff(data['tau_e']) == 0)

        models = anl.RetrieveModels(Nmods=1, **{'tanh_Tz0': 10.})

        assert models != []

        bad_walkers = anl.identify_bad_walkers()

        # Export to hdf5 and read back in just for fun.
        anl.export(anl.parameters, prefix='test_tanh', clobber=True)

        anl_hdf5 = ares.analysis.ModelSet('test_tanh')
        assert np.all(anl.chain == anl_hdf5.chain)
        assert np.all([anl.parameters[i] == anl_hdf5.parameters[i] \
            for i in range(len(anl.parameters))])

    # Clean-up. Assumes test suite is being run from $ARES
    mcmc_files = glob.glob('{}/test_tanh*'.format(os.environ.get('ARES')))

    # Iterate over the list of filepaths & remove each file.
    for fn in mcmc_files:
        try:
            os.remove(fn)
        except:
            print("Error while deleting file : ", filePath)

if __name__ == '__main__':
    test()
