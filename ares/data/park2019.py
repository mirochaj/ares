"""

park2019.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Fri 31 Dec 2021 12:43:15 EST

Description:

"""

from .mirocha2020 import legacy

base = legacy.copy()
base['pop_sfr_model'] = '21cmfast'

_updates = \
{
 # SFE
 'pop_fstar': 'pq[0]',
 'pq_func[0]': 'pl',
 'pq_func_var[0]': 'Mh',

 'pop_tstar': 0.5,                  # 0.5 in Park et al.

 # PL parameters
 'pq_func_par0[0]': 0.05,           # Table 1 in Park et al. (2019)
 'pq_func_par1[0]': 1e10,
 'pq_func_par2[0]': 0.5,
 'pq_func_par3[0]': -0.61,

 'pop_calib_wave': 1600,
 'pop_calib_lum': None,
 'pop_lum_per_sfr': 1. / 1.15e-28,    # Park et al. (2019); Eq. 12

 # Mturn stuff
 'pop_Mmin': 1e5, # Let focc do the work.
 'pop_focc': 'pq[40]',
 "pq_func[40]": 'exp-',
 'pq_func_var[40]': 'Mh',
 'pq_func_par0[40]': 1.,
 'pq_func_par1[40]': 5e8,
 'pq_func_par2[40]': -1.,

}

base.update(_updates)
