"""

SynthesisModelHybrid.py

Author: Henri Lamarre
Affiliation: UCLA
Created on: Tue Oct 1 16:57:45 PDT 2019

Description: 

"""

import numpy as np
from .SynthesisModel import SynthesisMaster, SynthesisModel
import scipy.interpolate as sci
from .Source import Source

class SynthesisModelHybrid(SynthesisMaster):

    def __init__(self, **kwargs):
        Source.__init__(self, **kwargs)
        self.starburst = None
        self.bpass = None
        self.interpolated_data = None

    @property
    def data(self):
        if self.bpass is None:
            self.bpass = SynthesisModel(source_sed='eldridge2009', source_Z=self.pf['source_Z'], source_ssp=False)
        if self.starburst is None:
            self.starburst = SynthesisModel(source_sed='leitherer1999',source_Z=self.pf['source_Z'], source_ssp=False)
        if self.interpolated_data is None:
            b_interp = 150
            b_data = np.array(self.bpass.data).T
            smooth_bpass = []
            x_b = self.bpass.wavelengths
            kernel_b = np.zeros_like(x_b)
            kernel_b[x_b.size//2 - b_interp//2: x_b.size//2 + b_interp//2] = np.ones(b_interp) / float(b_interp)
            smooth_bpass = np.zeros_like(b_data)
            for i in range(len(smooth_bpass)):
                smooth_bpass[i] = np.convolve(b_data[i], kernel_b, mode='same')
            
            s_interp = 12
            s_data = np.array(self.starburst.data).T
            smooth_starburst = []
            x_s = self.starburst.wavelengths
            kernel_s = np.zeros_like(x_s)
            kernel_s[x_s.size//2 - s_interp//2: x_s.size//2 + s_interp//2 ] = np.ones(s_interp) / float(s_interp)
            smooth_starburst = np.zeros_like(s_data)
            for i in range(len(smooth_starburst)):
                smooth_starburst[i] = np.convolve(s_data[i], kernel_s, mode='same')

            interpolated_b = sci.interp2d(self.bpass.wavelengths, self.bpass.times, smooth_bpass)
            interpolated_s = sci.interp2d(self.starburst.wavelengths, self.starburst.times, smooth_starburst)
            print(self.pf['source_coef'])
            self.interpolated_data =\
            (self.pf['source_coef']*interpolated_b(self.wavelengths, self.times)+(1-self.pf['source_coef'])*interpolated_s(self.wavelengths, self.times)).T
        
        return self.interpolated_data

    @property
    def wavelengths(self):
        if self.bpass is None:
            self.bpass = SynthesisModel(source_sed='eldridge2009', source_Z=self.pf['source_Z'], source_ssp=False)
        if self.starburst is None:
            self.starburst = SynthesisModel(source_sed='leitherer1999',source_Z=self.pf['source_Z'], source_ssp=False)
        w_space = np.logspace(np.log10(max(self.bpass.wavelengths[0], self.starburst.wavelengths[0])),
                                       np.log10(min(self.bpass.wavelengths[-1], self.starburst.wavelengths[-1])),
                                       min(len(self.bpass.wavelengths), len(self.starburst.wavelengths)))
        return w_space

    @property
    def times(self):
        if self.bpass is None:
            self.bpass = SynthesisModel(source_sed='eldridge2009', source_Z=self.pf['source_Z'], source_ssp=False)
        if self.starburst is None:
            self.starburst = SynthesisModel(source_sed='leitherer1999',source_Z=self.pf['source_Z'], source_ssp=False)
        t_space = np.linspace(max(self.bpass.times[0], self.starburst.times[0]),
                    min(self.bpass.times[-1], self.starburst.times[-1]),
                    min(len(self.bpass.times), len(self.starburst.times)))
        return t_space

    # @property
    # def L1600_per_sfr(self):
    #     return self.L_per_sfr()

    # def L_per_sfr(self, wave=1600., avg=1, Z=None):
    #     """
    #     Specific emissivity at provided wavelength.
        
    #     Parameters
    #     ----------
    #     wave : int, float
    #         Wavelength at which to determine emissivity.
    #     avg : int
    #         Number of wavelength bins over which to average
        
    #     Units are 
    #         erg / s / Hz / (Msun / yr)
    #     or 
    #         erg / s / Hz / Msun

    #     """
        
    #     cached = self._cache_L_per_sfr(wave, avg, Z)
        
    #     if cached is not None:
    #         return cached
        
    #     yield_UV = self.L_per_SFR_of_t(wave)
            
    #     # Interpolate in time to obtain final LUV
    #     if self.pf['source_tsf'] in self.times:
    #         result = yield_UV[np.argmin(np.abs(self.times - self.pf['source_tsf']))]
    #     else:
    #         k = np.argmin(np.abs(self.pf['source_tsf'] - self.times))
    #         if self.times[k] > self.pf['source_tsf']:
    #             k -= 1
                
    #         if not hasattr(self, '_LUV_interp'):
    #             self._LUV_interp = sci.interp1d(self.times, yield_UV, kind='linear')
            
    #         result = self._LUV_interp(self.pf['source_tsf'])
            
    #     self._cache_L_per_sfr_[(wave, avg, Z)] = result
            
    #     return result