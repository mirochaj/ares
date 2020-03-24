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
