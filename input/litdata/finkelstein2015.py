"""
Bouwens et al., 2015, ApJ, 803, 34

Table 6. 4 the last 5 rows.
"""

import numpy as np

info = \
{
 'reference': 'Finkelstein et al., 2015, ApJ, 810, 71',
 'data': 'Table 5', 
 'fits': 'Table 4', 
}

redshifts = [4, 5, 6, 7, 8]
wavelength = 1500.

ULIM = -1e10

fits = {}

fits['lf'] = {}

fits['lf']['pars'] = \
{
 'Mstar': [], 
 'pstar': [],
 'alpha': [],
}

fits['lf']['err'] = \
{
 'Mstar': [], 
 'pstar': [],
 'alpha': [],
}

# Table 5
tmp_data = {}
tmp_data['lf'] = \
{
 4: {  'M': list(np.arange(-23, -17, 0.5)),
       'phi': [0.0016, 0.0093, 0.0276, 0.1192, 0.2968,
               0.6491, 1.2637, 1.6645, 2.6392, 3.6169,
               5.8343, 6.4858],
       'err': [ULIM, [0.0045, 0.0033], [0.0074, 0.0062], [0.0145, 0.0132],
                     [0.0230, 0.0219], [0.0361, 0.0347], [0.0494, 0.0474],
                     [0.0630, 0.0618], [0.1192, 0.1165], [0.6799, 0.6091],
                     [0.8836, 0.8204], [1.0166, 0.9467]],
      },
 5: {  'M': list(np.arange(-23, -17, 0.5)),
        'phi': [0.0023, 0.0082, 0.0082, 0.0758, 0.2564, 0.5181, 0.9315,
                1.2086, 2.0874, 3.6886, 4.7361, 7.0842],
        'err': [ULIM, [0.0050, 0.0035], [0.0051, 0.0036], [0.0137, 0.0125],
                      [0.0255, 0.0240], [0.0365, 0.0338], [0.0477, 0.0482],
                      [0.0488, 0.0666], [0.1212, 0.1147], [0.3864, 0.3725],
                      [0.4823, 0.4413], [1.2829, 1.1364]],
       },      
 
 6: {  'M': list(np.arange(-23, -17, 0.5)),
        'phi': [0.0025, 0.0025, 0.0091, 0.0338, 0.0703, 0.1910, 
                0.3970, 0.5858, 0.8375, 2.4450, 3.6662, 5.9126],
        'err': [ULIM, ULIM, 
                [0.0057, 0.0039], [0.0105, 0.0085], [0.0148, 0.0128],
                [0.0249, 0.0229], [0.0394, 0.0357], [0.0527, 0.0437],
                [0.0916, 0.0824], [0.3887, 0.3515], [1.0076, 0.8401],
                [1.4481, 1.2338]],
       }, 
 7: {  'M': list(np.arange(-23, -17.5, 0.5)),
         'phi': [0.0029, 0.0029, 0.0046, 0.0187, 0.0690, 0.1301, 0.2742,
                 0.3848, 0.5699, 2.5650, 3.0780],
         'err': [ULIM, ULIM,
                 [0.0049, 0.0028], [0.0085, 0.0067], [0.0156, 0.0144],
                 [0.0239, 0.0200], [0.0379, 0.0329], [0.0633, 0.0586],
                 [0.2229, 0.1817], [0.8735, 0.7161], [1.0837, 0.8845]],
        },     
 8: {  'M': list(np.arange(-23, -18, 0.5)),
          'phi': [0.0035, 0.0035, 0.0035, 0.0079, 0.0150, 0.0615,
                  0.1097, 0.2174, 0.6073, 1.5110],
          'err': [ULIM, ULIM, ULIM, 
                  [0.0068, 0.0046], [0.0094, 0.0070], [0.0197, 0.0165],
                  [0.0356, 0.0309], [0.1805, 0.1250], [0.3501, 0.2616],
                  [1.0726, 0.7718]],
         },          
 
}

for redshift in tmp_data['lf'].keys():
    for i in range(len(tmp_data['lf'][redshift]['M'])):        
        tmp_data['lf'][redshift]['phi'][i] *= 1e-3
        
        if tmp_data['lf'][redshift]['err'][i] == ULIM:
            continue
            
        tmp_data['lf'][redshift]['err'][i][0] *= 1e-3
        tmp_data['lf'][redshift]['err'][i][1] *= 1e-3
        tmp_data['lf'][redshift]['err'][i] = \
            tuple(tmp_data['lf'][redshift]['err'][i])

units = {'phi': 1.}

data = {}
data['lf'] = {}
for key in tmp_data['lf']:
    mask = []
    for element in tmp_data['lf'][key]['err']:
        if element == ULIM:
            mask.append(1)
        else:
            mask.append(0)
    
    mask = np.array(mask)
    
    data['lf'][key] = {}
    data['lf'][key]['M'] = np.ma.array(tmp_data['lf'][key]['M'], mask=mask) 
    data['lf'][key]['phi'] = np.ma.array(tmp_data['lf'][key]['phi'], mask=mask) 
    data['lf'][key]['err'] = tmp_data['lf'][key]['err']








