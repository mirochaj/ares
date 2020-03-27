"""
Oesch et al., 2016, arxiv

Table 2 and volume estimate from text.
"""

import numpy as np

#info = \
#{
# 'reference': 'Oesch et al., 2016, ApJ, 786, 108',
# 'data': 'Table 5', 
#}

redshifts = [11.1]

wavelength = 1600. # I think?

ULIM = -1e10

tmp_data = {}
tmp_data['lf'] = \
{
 11.1: {'M': [-22.1],
       'phi': [1. / 1.2e6],
       'err': [None],
      },               
}

units = {'lf': 1.}

data = {}
data['lf'] = {}
for key in tmp_data['lf']:
    #mask = np.array(tmp_data['lf'][key]['err']) == ULIM
    N = len(tmp_data['lf'][key]['M'])
    mask = np.array([tmp_data['lf'][key]['err'][i] == ULIM for i in range(N)])
    
    data['lf'][key] = {}
    data['lf'][key]['M'] = np.ma.array(tmp_data['lf'][key]['M'], mask=mask) 
    data['lf'][key]['phi'] = np.ma.array(tmp_data['lf'][key]['phi'], mask=mask) 
    data['lf'][key]['err'] = tmp_data['lf'][key]['err']
