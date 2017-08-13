"""
Sanders et al. 2015

Table 1.
"""

import numpy as np

info = \
{
 'reference': 'Sanders et al. 2015',
 'data': 'Table 1', 
}

# u, g, and r dropouts, respectively
redshifts = [2.3]

ULIM = -1e10

data = {}
data['mzr'] = \
{
 2.3: {'M': [10**9.45, 10**9.84, 10**10.11, 10**10.56],
       'phi': [8.18, 8.30, 8.44, 8.52],
       'err': [(0.10, 0.07), (0.05, 0.04), (0.04, 0.04), (0.02, 0.02)],           
      },               
}

units = {'mzr': 1.}

