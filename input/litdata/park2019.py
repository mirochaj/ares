"""

park2019.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Fri 31 Dec 2021 12:43:15 EST

Description:

"""

from mirocha2017 import dpl

base = dpl.copy()
base['pop_sfr_model'] = '21cmfast'

_updates = \
{
 # SFE
 'pop_fstar{0}': 'pq[0]',
 'pq_func[0]{0}': 'pl',
 'pq_func_var[0]{0}': 'Mh',

 'pop_tstar{0}': 0.5,                  # 0.5 in Park et al.

 # PL parameters
 'pq_func_par0[0]{0}': 0.05,           # Table 1 in Park et al. (2019)
 'pq_func_par1[0]{0}': 1e10,
 'pq_func_par2[0]{0}': 0.5,
 'pq_func_par3[0]{0}': -0.61,

 'pop_calib_wave{0}': 1600,
 'pop_calib_lum{0}': None,
 'pop_lum_per_sfr{0}': 1. / 1.15e-28,    # Park et al. (2019); Eq. 12

 # Should add Mturn stuff
}

base.update(_updates)
