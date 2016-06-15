"""

test_setup_pbundles.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Mon Jun 13 11:59:46 PDT 2016

Description: 

"""

import ares

pop1 = ares.util.ParameterBundles.Population('sfe')
pop2 = ares.util.ParameterBundles.Population('sfe')

pop1.num = 0
pop2.num = 0


sed = ares.util.ParameterBundles.Spectrum('bpass')

pf = pop1 + sed

gpop = ares.populations.GalaxyPopulation(**pf)

"""
How to deal with multiple PHPs?

Institute warning if you're about to add one, that you need to assign an ID
number to the first PHP befor eyou continue?

Should make these like defaults and labels, i.e., add a config file.
"""

