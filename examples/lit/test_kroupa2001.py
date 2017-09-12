"""

test_kroupa2001.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Jun 11 14:05:10 MDT 2015

Description: 

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

k01 = ares.util.read_lit('kroupa2001')
imf = k01.InitialMassFunction()

m = np.logspace(-2, 2)

pl.loglog(m, list(map(imf, m)))

