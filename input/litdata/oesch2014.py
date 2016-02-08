"""
Oesch et al., 2014, ApJ, 786, 108

Table 6. 4 the last 5 rows.
"""

info = \
{
 'reference': 'Oesch et al., 2014, ApJ, 786, 108',
 'data': 'Table 5', 
}

redshifts = [10]

ULIM = -1e10

data = {}
data['lf'] = \
{
 10.: {'M': [-21.28, -20.78, -20.28, -19.78, -19.28, -18.78, -18.28, -17.78],
       'phi': [0.0027e-3, 0.01e-3, 0.0078e-3, 0.02e-3, 0.089e-3, 0.25e-3, 0.68e-3, 1.3e-3],
       'err': [(0.0023e-3, 0.0027e-3), (0.005e-3, 0.006e-3), ULIM, ULIM, 
               ULIM, ULIM, ULIM, (1.1e-3, 1.3e-3)],
      },               
}

units = {'phi': 1.}

