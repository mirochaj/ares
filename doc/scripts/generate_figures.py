"""

generate_figures.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Jun 29 16:32:22 MDT 2015

Description: 

"""

import matplotlib.pyplot as pl

scripts = \
[
 'run_example_crb_lw.py',

]


for script in scripts:
    with open(script) as script_file:
        exec(compile(script_file, script, 'exec'))
    
