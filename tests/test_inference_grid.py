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
    size = z0.size * dz.size
    
    mg.axes = {'tanh_xz0': z0, 'tanh_xdz': dz}
    
    # Basic checks
    assert mg.grid.Nd == 2
    assert mg.grid.structured == True
    assert len(mg.grid.coords) == size
    assert [mg.grid.axis(i) for i in range(2)] \
        == [mg.grid.axis(par) for par in mg.grid.axes_names]
        
    assert mg.grid.meshgrid(mg.grid.axes_names[0]).size == size
    
    mg.run('test_grid', clobber=True, save_freq=100)
        
    anl = ares.analysis.ModelSet('test_grid')
    ax1 = anl.Scatter(anl.parameters, c='tau_e', fig=1)
    
    anl_2 = anl.Slice((0.06, 0.08), ['tau_e'])
    anl_2.Scatter(anl_2.parameters, ax=ax1, color='k')
    
    ax2, cb2 = anl.Contour(anl.parameters, 'tau_e', fig=2)
    ax3, cb3 = anl.Contour(anl.parameters, 'tau_e', fig=3, fill=False)
    
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

    