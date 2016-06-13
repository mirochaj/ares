"""

test_setup_pbundles.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Mon Jun 13 11:59:46 PDT 2016

Description: 

"""

import ares

pb_pop = ares.util.ParameterBundles.Population('sfe')
pb_sed = ares.util.ParameterBundles.Spectrum('bpass')

pf = pb_pop + pb_sed

pop = ares.populations.GalaxyPopulation(**pf)

