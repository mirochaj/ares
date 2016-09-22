"""

ares/util/BlobBundles.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Tue Aug  9 14:19:32 PDT 2016

Description: 

"""

import numpy as np
from .ParameterBundles import ParameterBundle

_gs_hist = ['cgm_h_2', 'igm_h_2', 'igm_Tk', 'Ja', 'Jlw', 'Ts', 'dTb']

_gs_ext = ['z_B', 'z_C', 'z_D']
for tp in list('BCD'):
    for field in _gs_hist:
        _gs_ext.append('%s_%s' % (field, tp))

# Add the zero-crossing even though its not an extremum
_gs_ext.append('z_ZC')
_gs_ext.append('tau_e')

_def_z = np.arange(5, 41, 0.1)

_gs_shape_n = ['hwhm_diff_C', 'hwqm_diff_C', 'fwhm_C', 'fwqm_C']
_gs_shape_n.extend(['hwhm_diff_D', 'hwqm_diff_D', 'fwhm_D', 'fwqm_D'])

_gs_shape_f = ['Width(max_fraction=0.5, peak_relative=True)', 
               'Width(max_fraction=0.25, peak_relative=True)',
               'Width(max_fraction=0.5, peak_relative=False)',
               'Width(max_fraction=0.25, peak_relative=False)',
               'Width(absorption=False, max_fraction=0.5, peak_relative=True)', 
               'Width(absorption=False, max_fraction=0.25, peak_relative=True)',
               'Width(absorption=False, max_fraction=0.5, peak_relative=False)',
               'Width(absorption=False, max_fraction=0.25, peak_relative=False)'
               ]

_extrema = {'blob_names':_gs_ext, 'blob_ivars': None,  'blob_funcs': None}
_history = {'blob_names':_gs_hist,'blob_ivars': _def_z,'blob_funcs': None}
_shape = {'blob_names':_gs_shape_n,'blob_ivars': None, 'blob_funcs': _gs_shape_f}

_blobs = \
{
 'gs': {'basics': _extrema, 'history': _history, 'shape': _shape},
 'rb': None, # eventually 
}

_keys = ('blob_names', 'blob_ivars', 'blob_funcs')

class BlobBundle(ParameterBundle):
    def __init__(self, bundle=None, **kwargs):
        ParameterBundle.__init__(self, bundle=bundle, bset=_blobs, **kwargs)
                
    def __add__(self, other):
        
        tmp = self.copy()
        
        out = {key: [] for key in _keys}
        
        # Make sure to not overwrite anything here!
        for d in [self, other]:
            for key in _keys:
                out[key].append(d[key])
                
        return BlobBundle(**out)
        
        