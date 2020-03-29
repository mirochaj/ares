"""

test_inference_grid.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Wed 25 Mar 2020 11:32:57 EDT

Description: 

"""

import os
import glob
import ares
import numpy as np

def test():
    blobs_scalar = ['z_D', 'dTb_D', 'tau_e']
    blobs_1d = ['cgm_h_2', 'igm_Tk', 'dTb']
    blobs_1d_z = np.arange(5, 21)
    
    base_pars = \
    {
     'problem_type': 101,
     'tanh_model': True,
     'blob_names': [blobs_scalar, blobs_1d],
     'blob_ivars': [None, [('z', blobs_1d_z)]],
     'blob_funcs': None,
    }
    
    mg = ares.inference.ModelGrid(**base_pars)
    
    z0 = np.arange(6, 13, 1)
    dz = np.arange(1, 8, 1)
    
    mg.axes = {'tanh_xz0': z0, 'tanh_xdz': dz}
    
    mg.run('test_grid', clobber=True, save_freq=100)
    
    anl = ares.analysis.ModelSet('test_grid')
    ax1 = anl.Scatter(anl.parameters, c='tau_e', fig=1)
    
    anl_2 = anl.Slice((0.06, 0.08), ['tau_e'])
    anl_2.Scatter(anl_2.parameters, ax=ax1, color='k')
    
    #ax2 = anl.ContourScatter(anl.parameters[0], anl.parameters[1], 'tau_e',
    #    fig=3)
    
    # Clean-up
    mcmc_files = glob.glob('{}/test_grid*'.format(os.environ.get('ARES')))
    
    # Iterate over the list of filepaths & remove each file.
    for fn in mcmc_files:
        try:
            os.remove(fn)
        except:
            print("Error while deleting file : ", filePath)

if __name__ == '__main__':
    test()

    