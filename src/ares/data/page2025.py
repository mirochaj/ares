"""
page2025.py

Page et al. 2025, MNRAS, 536, 518P

https://arxiv.org/abs/2501.06075
https://ui.adsabs.harvard.edu/abs/2025MNRAS.536..518P/abstract
"""

import numpy as np

redshifts = [0.5]
cosmo = \
{
 'hubble_0': 0.7,
 'omega_m_0': 0.3,
 'omega_l_0': 0.7,
}
units = {'M': 'mags_abs', 'phi': 'log10(density)'}

magbins = np.arange(-20.72, -17.42, 0.3)

# error bars are (+/-)
data = {}
data['lf'] = \
{
 0.5: {'M': magbins,
       'phi': np.array([-4.57, -4.27, -3.97, -3.66, -3.15, -2.97, -2.76, -2.61, 
            -2.42, -2.35, -2.13]),
       'err': np.array([(0.52, 0.76), (0.37, 0.45), (0.25, 0.28), (0.17, 0.18), (0.09, 0.10),
    (0.08, 0.08), (0.07, 0.07), (0.05, 0.06), (0.06, 0.06), (0.09, 0.09),
    (0.14, 0.15)]),
      },
}


