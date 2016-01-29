"""
Oesch et al., 2013, ApJ, 773, 75

Table 6. 4 the last 5 rows.
"""

info = \
{
 'reference': 'Oesch et al., 2013, ApJ, 773, 75',
 'data': 'Table 4', 
}

redshifts = [9, 10]

ULIM = -1e10

fits = {}

fits['lf'] = {}

#fits['lf']['pars'] = \
#{
# 'Mstar': [-20.88, -21.17, -20.94, -20.87, -20.63], 
# 'pstar': [1.97e-3, 0.74e-3, 0.5e-3, 0.29e-3, 0.21e-3],
# 'alpha': [-1.64, -1.76, -1.87, -2.06, -2.02],
#}
#
#fits['lf']['err'] = \
#{
# 'Mstar': [0.08, 0.12, 0.2, 0.26, 0.36], 
# 'pstar': [0.315e-3, 0.16e-3, 0.19e-3, 0.165e-3, 0.17e-3],  # should be asymmetric!
# 'alpha': [0.04, 0.05, 0.1, 0.13, 0.23],
#}

# Table 4
# Note: not currently including any of the upper limits
data = {}
data['lf'] = \
{
 9.: {'M': [-20.66, -19.66, -18.66, -17.66],
      'phi': [0.18e-3, 0.15e-3, 0.35e-3, 1.6e-3],
      'err': [ULIM, (0.13e-3, 0.15e-3), 0.24e-3, 0.9e-3],
      },
 10.: {'M': [-20.78, -20.28, -19.78, -19.28, -18.78, -18.28, -17.78],
       'phi': [0.0077e-3, 0.013e-3, 0.027e-3, 0.083e-3, 0.17e-3, 0.34e-3, 0.58e-3],
       'err': [ULIM, ULIM, ULIM, ULIM, ULIM, ULIM, (0.5e-3, 0.58e-3)],
      },               
}

units = {'phi': 1.}

