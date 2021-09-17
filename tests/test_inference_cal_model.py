"""

test_inference_cal_model.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Fri 27 Mar 2020 09:28:21 EDT

Description:

"""

import os
import glob
import ares

def test():

    gpop = ares.analysis.GalaxyPopulation()

    pars = ares.util.ParameterBundle('mirocha2020:univ')
    pars['pop_thin_hist'] = 1 # speed-up
    pars['pop_dust_yield'] = 0
    pars.update(ares.util.ParameterBundle('testing:galaxies'))

    # Test with standard 4-parameter SFE model
    cal = ares.inference.CalibrateModel(fit_lf=[6], fit_beta=False,
        free_params_sfe=['norm'],
        zevol_sfe=None,
        include_fduty=False, include_fdtmr=False,
        save_sam=False, save_smf=False, save_lf=True, save_beta=False,
        save_sfrd=True, ztol=0.21)

    cal.base_kwargs = pars

    blobs = cal.blobs

    assert len(cal.parameters) == 1
    assert len(cal.guesses.keys()) == 1
    assert cal.free_params_dust == []

    # Run for a few steps
    cal.run(steps=1, burn=0, save_freq=1, prefix='test_lfcal', clobber=True)

    anl = ares.analysis.ModelSet('test_lfcal')

    assert anl.chain.shape == (2, 1), "Chain not the right shape."
    assert anl.logL.size == 2, "logL not the right size."

    # Add some dust
    cal = ares.inference.CalibrateModel(fit_lf=[6], fit_beta=[6],
        free_params_sfe=['norm', 'peak', 'slope-low', 'slope-high'],
        zevol_sfe=None,
        include_dust='screen',
        free_params_dust=['norm', 'slope', 'scatter'],
        zevol_dust=None,
        include_fduty=False, include_fdtmr=False,
        save_sam=True, save_smf=True, save_lf=True, save_beta=True,
        save_sfrd=True, ztol=0.21)

    cal.base_kwargs = pars

    blobs = cal.blobs

    assert len(cal.parameters) == 7
    assert len(cal.guesses.keys()) == 7
    assert len(cal.free_params_dust) == 3

    guesses = cal.get_initial_walker_position()

    assert len(guesses.keys()) == 7

    pars['pop_dust_yield'] = 0.4
    pop = ares.populations.GalaxyPopulation(**pars)

    # Clean-up
    mcmc_files = glob.glob('{}/test_lfcal*'.format(os.environ.get('ARES')))

    # Iterate over the list of filepaths & remove each file.
    for fn in mcmc_files:
        try:
            os.remove(fn)
        except:
            print("Error while deleting file : ", filePath)


if __name__ == '__main__':
    test()
