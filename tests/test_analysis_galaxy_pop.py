"""

test_analysis_galaxy_pop.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Fri 27 Mar 2020 09:42:49 EDT

Description:

"""

import ares
import numpy as np

def test():

    gpop = ares.analysis.GalaxyPopulation()

    ax_lf = gpop.Plot(z=4, round_z=0.21, quantity='lf', fig=1)
    ax_smf = gpop.Plot(z=4, round_z=0.21, quantity='smf', fig=2)

    ax_lf_multi = gpop.MultiPlot([4,6,8], ncols=3,
        round_z=0.21, quantity='lf', fig=3)

    #pars = ares.util.ParameterBundle('mirocha2017:base').pars_by_pop(0, 1)
    #pop = ares.populations.GalaxyPopulation(**pars)

    # We run this inside test_inference_cal_model.py with a population object
    # and MCMC output.
    ax_mega = gpop.PlotSummary(None, fig=4)

    pars = ares.util.ParameterBundle('mirocha2020:univ') \
         + ares.util.ParameterBundle('testing:galaxies')
    pop = ares.populations.GalaxyPopulation(**pars)

    axes5 = gpop.PlotColors(pop, fig=5)
    axes6 = gpop.PlotColors(pop, fig=6, show_Mstell=True, show_MUV=False,
        dlam_c94=10)
    gpop.PlotColorEvolution(pop, show_beta_jwst_W=False, show_beta_jwst_M=False,
        include_Mstell=False, zarr=np.array([6,7,8]), fig=7)
    #gpop.PlotColorEvolution(pop, show_beta_jwst=False, show_beta_hst=False,
    #    include_Mstell=True,
    #    zarr=np.array([4, 6]), fig=8, dlam=2)


if __name__ == '__main__':
    test()
