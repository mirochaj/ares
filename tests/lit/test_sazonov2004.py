"""

test_sazonov2004.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Jun 10 17:37:50 MDT 2015

Description: 

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

s04 = ares.util.read_lit('sazonov2004')

E = np.logspace(0., 6., 100)
F = map(s04.Spectrum, E)

pl.loglog(E, F)
pl.xlabel(ares.util.labels['E'])
pl.ylabel('Relative intensity')
