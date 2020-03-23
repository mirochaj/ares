"""

generate_inits_tables.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Mon 23 Mar 2020 13:29:43 EDT

Description: 

"""

import sys
import ares

try:
    cosmorec_path = sys.argv[1]
except IndexError:
    raise IndexError("Must supply path to CosmoRec executable!")
    
cosm = ares.physics.Cosmology(cosmology_name='planck_TTTEEE_lowl_lowE',
    cosmology_id='best', cosmorec_path=cosmorec_path)
    
cosm.get_inits()    
    