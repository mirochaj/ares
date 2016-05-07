"""

test_lf.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Thu Jan 28 12:37:32 PST 2016

Description: 

"""

import ares

lf = ares.analysis.ObservedLF()

ax = lf.Plot(6.9, sources=['bouwens2015', 'atek2015'], round_z=0.1)

mp = lf.MultiPlot([3.8, 4.9, 5.9, 6.9, 7.9, 9.0], ncols=3, round_z=0.3, fig=2)





