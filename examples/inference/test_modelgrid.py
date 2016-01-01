"""

test_modelgrid.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sun Jun 15 12:59:04 MDT 2014

Description: 

"""

import ares, time
import numpy as np

#
##
z0 = np.arange(6, 12, 0.1)
dz = np.arange(0.1, 8.1, 0.1)
grid_axes = {'tanh_xz0': z0, 'tanh_xdz': dz}
##
#

base_pars = \
{
 'problem_type': 101,
 'tanh_model': True,
 'blob_names': [['tau_e', 'z_C', 'z_D', 'igm_dTb_C', 'igm_dTb_D'], 
    ['cgm_h_2', 'igm_Tk', 'igm_dTb']],
 'blob_ivars': [None, np.arange(5, 21)],
 'blob_funcs': None,
}


mg = ares.inference.ModelGrid(**base_pars)

mg.axes = grid_axes
mg.LoadBalance(0)

t1 = time.time()
mg.run('test_grid', clobber=True, save_freq=10)
t2 = time.time()

print "Run complete in %.4g minutes.\n" % ((t2 - t1) / 60.)





