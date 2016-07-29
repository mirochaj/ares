"""

test_setup_pbundles.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Mon Jun 13 11:59:46 PDT 2016

Description: 

"""

import ares

pop1 = ares.util.ParameterBundle('pop:sfe')
pop2 = ares.util.ParameterBundle('pop:fcoll')

pop1.num = 0
pop2.num = 1

sed = ares.util.ParameterBundle('sed:bpass')

pf = pop1 + sed

assert pf - pop1 == sed, 'Error in subtraction of ParameterBundle.'
