"""

generate_figures.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Thu May 31 16:02:30 PDT 2018

Description: 

"""

import os
import sys

try:
    fn = sys.argv[1]
except IndexError:
    fn = None

docs = \
[
 'example_gs_standard.rst',
 'example_gs_multipop.rst',
 'example_gs_phenomenological.rst',
 'example_pop_galaxy.rst',
 'example_crb_uv.rst',
 'example_crb_xr.rst', 
 'example_pop_galaxy.rst',
 'uth_pop_radiation.rst',
 'uth_pq.rst',
]

for fn_rst in docs:
    
    # Way to run single .rst file if we want
    if fn is not None:
        if fn_rst != fn:
            continue
        
    os.system('python $MODS/rst_to_py/rst_to_py.py {}'.format(fn_rst))
    
    fn_py = fn_rst.replace('rst', 'py')
    
    print('Running {}...'.format(fn_py))
    os.system('python {}'.format(fn_py))
    print('')
    
