"""
Bouwens et al., 2017, ApJ, 843, 129

Table 4 and volume estimate from text.
"""

info = \
{
 'reference': 'Bouwens et al., 2017, ApJ, 843, 129',
 'data': 'Table 5', 
 'label': 'Bouwens+ (2017)' 
}

import numpy as np

redshifts = [6.]

wavelength = 1600. # I think?

ULIM = -1e10

tmp_data = {}
tmp_data['lf'] = \
{
 6.0: {'M': list(np.arange(-20.75, -12.25, 0.5)),
       'phi': [0.0002, 0.0009, 0.0007, 0.0018, 0.0036,   
               0.0060, 0.0071, 0.0111, 0.0170, 0.0142,
               0.0415, 0.0599, 0.0817, 0.1052, 0.1275,
               0.1464, 0.1584],
       'err': [(0.0002, 0.0002), (0.0004, 0.0004),
               (0.0004, 0.0004), (0.0006, 0.0006),
               (0.0009, 0.0009), (0.0012, 0.0012),
               (0.0066, 0.0014), (0.0101, 0.0022),
               (0.0165, 0.0039), (0.0171, 0.0054),
               (0.0354, 0.0069), (0.0757, 0.0106),
               (0.1902, 0.0210), (0.5414, 0.0434),
               (1.6479, 0.0747), (5.4369, 0.1077),
               (19.8047, 0.1343)],
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

