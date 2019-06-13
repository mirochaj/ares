"""

Survey.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Sat 16 Feb 2019 10:40:18 EST

Description: 

"""

import re
import os
import copy
import numpy as np
import matplotlib.pyplot as pl
from ..physics.Constants import c
from ..physics.Cosmology import Cosmology

try:
    import ares
except ImportError:
    pass

flux_AB = 3631. * 1e-23 # 3631 * 1e-23 erg / s / cm**2 / Hz
nanoJ = 1e-23 * 1e-9

_path = os.environ.get('ARES') + '/input'

class Survey(object):
    def __init__(self, cam='nircam', mod='modA', chip=1):
        self.camera = cam
        self.chip = chip
        
        if cam == 'nircam':
            self.path = '{}/nircam/nircam_throughputs/{}/filters_only'.format(_path, mod)
        elif cam == 'wfc3':
            self.path = '{}/wfc3'.format(_path)
        elif cam == 'wfc':
            self.path = '{}/wfc'.format(_path)
        else:
            raise NotImplemented('Unrecognized camera \'{}\''.format(cam))
    @property
    def cosm(self):
        if not hasattr(self, '_cosm'):
            self._cosm = Cosmology()
        return self._cosm
    
    @property
    def src(self):
        if not hasattr(self, '_src'):
            from ares.sources import SynthesisModel
            self._src = SynthesisModel(source_sed='eldridge2009')
            print("Defaulting to BPASS v1 source model.")
        return self._src
            
    @src.setter
    def src(self, value):
        self._src = value
    
    @property
    def wavelengths(self):
        """
        Wavelength array [Angstrom] in REST frame of sources.
        """
        if not hasattr(self, '_wavelengths'):
            self._wavelengths = self.src.wavelengths
        return self._wavelengths
    
    @wavelengths.setter
    def wavelengths(self, value):
        self._wavelengths = value
        
    @property
    def frequencies(self):
        if not hasattr(self, '_frequencies'):
            self._frequencies = c / (self.wavelengths / 1e8)
        return self._frequencies    
        
    @property
    def dwdn(self):
        if not hasattr(self, '_dwdn'):
            tmp = np.abs(np.diff(self.wavelengths) / np.diff(self.frequencies))
            self._dwdn = np.concatenate((tmp, [tmp[-1]]))
        return self._dwdn
    
    def PlotFilters(self, ax=None, fig=1, filter_set='W', annotate=True):
        """
        Plot transmission curves for NIRCAM filters.
        """
        
        if ax is None:
            fig = pl.figure(fig, figsize=(6, 6))
            ax = fig.add_subplot(111)
            gotax = False
        else:
            gotax = True
        
        data = self._read_throughputs(filter_set)
        
        colors = ['k', 'b', 'c', 'm', 'y', 'r', 'orange', 'g'] * 10
        for i, filt in enumerate(data.keys()):
            
            ax.plot(data[filt][0], data[filt][1], label=filt, color=colors[i])
            
            if annotate:
                if filt.endswith('IR'):
                    _filt = filt[0:-3]
                else:    
                    _filt = filt
                    
                ax.annotate(_filt, (data[filt][2], 0.8), ha='center', va='top',
                    color=colors[i], rotation=90)
            
        ax.set_xlabel(r'Observed Wavelength $[\mu \mathrm{m}]$')    
        ax.set_ylabel('Transmission')
        ax.set_ylim(-0.05, 1.05)
        #ax.legend(loc='best', frameon=False, fontsize=10, ncol=2)
            
        return ax
    
    def _read_throughputs(self, filter_set='W', filters=None):
        if self.camera == 'nircam':
            return self._read_nircam(filter_set, filters)
        elif self.camera == 'wfc3':
            return self._read_wfc3(filter_set, filters)
        elif self.camera == 'wfc':
            return self._read_wfc(filter_set, filters)
        else:
            raise NotImplemented('help')
            
    def _read_nircam(self, filter_set='W', filters=None):

        if not hasattr(self, '_filter_cache'):
            self._filter_cache = {}
            
        get_all = False    
        if filters is not None:
            if filters == 'all':
                get_all = True
            else:
                assert type(filters) in [list, tuple]               
        else:
            filters = []
            if type(filter_set) != list:
                filter_set = [filter_set]

        data = {}
        for fn in os.listdir(self.path):
                    
            pre = fn.split('_')[0]
                
            if get_all or (pre in filters):
            
                if pre in self._filter_cache:
                    data[pre] = self._filter_cache[pre]
                    continue
                    
                if ('W2' in pre):
                    continue    
                
                num = re.findall(r'\d+', pre)[0]
                cent = float('{}.{}'.format(num[0], num[1:]))    
                
                # Wavelength [micron], transmission
                x, y = np.loadtxt('{}/{}'.format(self.path, fn), unpack=True, 
                    skiprows=1)
                
                data[pre] = self._get_filter_prop(x, y, cent)
                
                self._filter_cache[pre] = copy.deepcopy(data[pre])
            
            elif filter_set is not None:
                
                for _filters in filter_set:
                
                    if _filters in self._filter_cache:
                        data[pre] = self._filter_cache[_filters]
                        continue
                        
                    if _filters not in pre:
                        continue        
                    
                    # Need to distinguish W from W2
                    if (_filters == 'W') and ('W2' in pre):
                        continue
                                                                            
                    # Determine the center wavelength of the filter based its string
                    # identifier.    
                    k = pre.rfind(_filters)    
                    cent = float('{}.{}'.format(pre[1], pre[2:k]))
                    
                    # Wavelength [micron], transmission
                    x, y = np.loadtxt('{}/{}'.format(self.path, fn), unpack=True, 
                        skiprows=1)
                    
                    data[pre] = self._get_filter_prop(x, y, cent)
                    
                    self._filter_cache[pre] = copy.deepcopy(data[pre])
        
        return data   
        
    def _parse_filter(self, cam):    
        # Determine the center wavelength of the filter based on its 
        # string identifier.    
        k = pre.rfind(_filters)
        cent = float('{}.{}'.format(pre[1], pre[2:k]))
        
        _i, x, y = np.loadtxt('{}/IR/{}'.format(self.path, fn), 
            unpack=True, skiprows=1, delimiter=',')
        
    def _read_wfc(self, filter_set='W', filters=None):
        if not hasattr(self, '_filter_cache'):
            self._filter_cache = {}
            
        get_all = False    
        if filters is not None:
            if filters == 'all':
                get_all = True
            else:
                assert type(filters) in [list, tuple]
        else:   
            filters = []
            # Grab all 'W' or 'N' etc. filters 
            if type(filter_set) != list:
                filter_set = [filter_set]    
            
        data = {}
        for fn in os.listdir(self.path):
        
            pre = fn.split('wfc_')[1].split('.dat')[0]
                        
            if get_all or (pre in filters):
                
                if pre in self._filter_cache:
                    data[pre] = self._filter_cache[pre]
                    continue
                    
                cent = float('0.{}'.format(pre[1:4]))
                
                x, y = np.loadtxt('{}/{}'.format(self.path, fn), 
                    unpack=True, skiprows=1)
                                            
                # Convert wavelengths from nanometers to microns
                data[pre] = self._get_filter_prop(x / 1e4, y, cent)
            
                self._filter_cache[pre] = copy.deepcopy(data[pre])    
            
            elif filter_set is not None:
                for _filters in filter_set:
                
                    if _filters in self._filter_cache:
                        data[pre] = self._filter_cache[_filters]
                        continue
                        
                    if _filters not in pre:
                        continue
                    
                    # Determine the center wavelength of the filter based on its string
                    # identifier.    
                    k = pre.rfind(_filters)
                    cent = float('0.{}'.format(pre[1:k]))
                    
                    x, y = np.loadtxt('{}/{}'.format(self.path, fn), 
                        unpack=True, skiprows=1)
                                                
                    # Convert wavelengths from nanometers to microns
                    data[pre] = self._get_filter_prop(x / 1e4, y, cent)
                
                    self._filter_cache[pre] = copy.deepcopy(data[pre])
        
        return data    
                 
    def _read_wfc3(self, filter_set='W', filters=None):
        if not hasattr(self, '_filter_cache'):
            self._filter_cache = {}
            
        get_all = False    
        if filters is not None:
            if filters == 'all':
                get_all = True
            else:
                assert type(filters) in [list, tuple]
        else:    
            filters = []
            if type(filter_set) != list:
                filter_set = [filter_set]    
            
        data = {}
        for fn in os.listdir(self.path+'/IR'):
                    
            pre = fn.split('_IR_throughput')[0]        

            # Read-in no matter what
            if get_all or (pre in filters):
                
                if pre in self._filter_cache:
                    data[pre] = self._filter_cache[pre]
                    continue
                    
                cent = float('{}.{}'.format(pre[1], pre[2:-1]))    
                    
                _i, x, y = np.loadtxt('{}/IR/{}'.format(self.path, fn), 
                    unpack=True, skiprows=1, delimiter=',')
                    
                # Convert wavelengths from Angstroms to microns
                data[pre] = self._get_filter_prop(x / 1e4, y, cent)
            
                self._filter_cache[pre] = copy.deepcopy(data[pre])    
                    
                        
            elif filter_set is not None:
                
            
                for _filters in filter_set:
                
                    if _filters in self._filter_cache:
                        data[pre] = self._filter_cache[_filters]
                        continue
                        
                    if _filters not in pre:
                        continue
                
                    # Determine the center wavelength of the filter based on its 
                    # string identifier.    
                    cent = float('{}.{}'.format(pre[1], pre[2:-1]))
                    
                    _i, x, y = np.loadtxt('{}/IR/{}'.format(self.path, fn), 
                        unpack=True, skiprows=1, delimiter=',')
                                
                    # Convert wavelengths from Angstroms to microns
                    data[pre] = self._get_filter_prop(x / 1e4, y, cent)
                
                    self._filter_cache[pre] = copy.deepcopy(data[pre])

        return data

    def _get_filter_prop(self, x, y, cent):
        Tmax = max(y)
        _ok = y > 1e-3
        
        # Find non-contiguous regions (NIRCAM F090W only?)
        # This is a kludgey fix
        i = np.arange(0, x.size)
        
        bpts = np.where(np.diff(i[_ok==1]) != 1)[0]
        chunks = np.split(i[_ok==1], bpts+1)
        if len(chunks) == 1:
            ok = _ok
        else:
            # Each chunk is a list of contiguous indices
            for chunk in chunks:
                lo, hi = chunk[0], chunk[-1]
                if not (x[lo] <= cent <= hi):
                    continue
                break    
                
            ok = np.zeros_like(y)    
            ok[chunk] = 1
                
        # Compute width of filter
        hi = max(x[ok == True])
        lo = min(x[ok == True])
        
        # Average T-weighted wavelength in filter.     
        mi = np.sum(x[ok == True] * y[ok == True]) / np.sum(y[ok == True])
        
        dx = np.array([hi - mi, mi - lo])
        Tavg = np.sum(y[ok==1]) / y[ok==1].size
                
        # Get the Hz^-1 units back
        freq = c / x / 1e-4
        
        dHz = c / (lo * 1e-4) - c / (hi * 1e-4)
                
        return x, y, cent, dx, Tavg, dHz
