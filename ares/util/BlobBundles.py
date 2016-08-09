"""

ares/util/BlobBundles.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Tue Aug  9 14:19:32 PDT 2016

Description: 

"""

import numpy as np
from .ParameterBundles import ParameterBundle

_gs_ext = ['z_B', 'dTb_B', 'z_C', 'dTb_C', 'z_D', 'dTb_D']

_gs_hist = ['cgm_h_2', 'igm_h_2', 'igm_Ts', 'igm_Tk', 'dTb']
_def_z = np.arange(5, 41, 0.1)

_gs_shape_n = ['hwhm_diff_C', 'hwqm_diff_C', 'fwhm_C',
             'hwhm_diff_D', 'hwqm_diff_D', 'fwhm_D']
             
_gs_shape_f = ['Width(max_fraction=0.5, peak_relative=True)', 
               'Width(max_fraction=0.25, peak_relative=True)',
               'Width(max_fraction=0.25, peak_relative=False)',
               'Width(max_fraction=0.5, peak_relative=True)', 
               'Width(max_fraction=0.25, peak_relative=True)',
               'Width(max_fraction=0.25, peak_relative=False)'
               ]

_extrema = {'blob_names':_gs_ext, 'blob_ivars': None,  'blob_funcs': None}
_history = {'blob_names':_gs_hist,'blob_ivars': _def_z,'blob_funcs': None}
_shape = {'blob_names':_gs_shape_n,'blob_ivars': None, 'blob_funcs': _gs_shape_f}

_blobs = {'gs': {'extrema': _extrema, 'history': _history, 'shape': _shape}}

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
        
        