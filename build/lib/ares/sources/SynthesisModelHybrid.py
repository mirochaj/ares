"""

SynthesisModelHybrid.py

Author: Henri Lamarre
Affiliation: UCLA
Created on: Tue Oct 1 16:57:45 PDT 2019

Description:

"""

import numpy as np
from .SynthesisModel import SynthesisModelBase, SynthesisModel
import scipy.interpolate as sci
from .Source import Source

# Not in test suite because we only pull-down BPASS to save time.
class SynthesisModelHybrid(SynthesisModelBase): # pragma: no cover

    def __init__(self, **kwargs):
        self.pf = kwargs

        Source.__init__(self, **kwargs)
        self.starburst = None
        self.bpass = None
        self.smooth_data = None

    @property
    def data(self):
        if self.pf['pop_sps_data'] is not None:
            self.bpass = self.pf['pop_sps_data'][2]
            self.starburst = self.pf['pop_sps_data'][3]
            self.smooth_data = self.pf['pop_sps_data'][4]

        if self.bpass is None:
            model = SynthesisModel(source_sed='eldridge2009', source_Z=self.pf['source_Z'], source_ssp=False)
            self.bpass = [model.data, model.wavelenghts, model.times]
        if self.starburst is None:
            model = SynthesisModel(source_sed='leitherer1999',source_Z=self.pf['source_Z'], source_ssp=False)
            self.starburst = [model.data, model.wavelenghts, model.times]
        if self.smooth_data is None:
            b_interp = 150
            b_data = np.array(self.bpass[0]).T
            smooth_bpass = []
            x_b = self.bpass[1]
            kernel_b = np.zeros_like(x_b)
            kernel_b[x_b.size//2 - b_interp//2: x_b.size//2 + b_interp//2] = np.ones(b_interp) / float(b_interp)
            smooth_bpass = np.zeros_like(b_data)
            for i in range(len(smooth_bpass)):
                smooth_bpass[i] = np.convolve(b_data[i], kernel_b, mode='same')

            s_interp = 12
            s_data = np.array(self.starburst[0]).T
            smooth_starburst = []
            x_s = self.starburst[1]
            kernel_s = np.zeros_like(x_s)
            kernel_s[x_s.size//2 - s_interp//2: x_s.size//2 + s_interp//2 ] = np.ones(s_interp) / float(s_interp)
            smooth_starburst = np.zeros_like(s_data)
            for i in range(len(smooth_starburst)):
                smooth_starburst[i] = np.convolve(s_data[i], kernel_s, mode='same')

            self.smooth_data = [smooth_bpass, smooth_starburst]

        interpolated_b = sci.interp2d(self.bpass[1], self.bpass[2], self.smooth_data[0])
        interpolated_s = sci.interp2d(self.starburst[1], self.starburst[2], self.smooth_data[1])
        data = (self.pf['source_coef']*interpolated_b(self.wavelengths, self.times)+\
            (1-self.pf['source_coef'])*interpolated_s(self.wavelengths, self.times)).T

        return data

    @property
    def wavelengths(self):
        if self.pf['pop_sps_data'] is not None:
            self.bpass = self.pf['pop_sps_data'][2]
            self.starburst = self.pf['pop_sps_data'][3]
        if self.bpass is None:
            model = SynthesisModel(source_sed='eldridge2009', source_Z=self.pf['source_Z'], source_ssp=False)
            self.bpass = [model.data, model.wavelenghts, model.times]
        if self.starburst is None:
            model = SynthesisModel(source_sed='leitherer1999',source_Z=self.pf['source_Z'], source_ssp=False)
            self.starburst = [model.data, model.wavelenghts, model.times]

        w_space = np.logspace(np.log10(max(self.bpass[1][0], self.starburst[1][0])),
                                       np.log10(min(self.bpass[1][-1], self.starburst[1][-1])),
                                       min(len(self.bpass[1]), len(self.starburst[1])))
        return w_space

    @property
    def times(self):
        if self.pf['pop_sps_data'] is not None:
            self.bpass = self.pf['pop_sps_data'][2]
            self.starburst = self.pf['pop_sps_data'][3]
        if self.bpass is None:
            model = SynthesisModel(source_sed='eldridge2009', source_Z=self.pf['source_Z'], source_ssp=False)
            self.bpass = [model.data, model.wavelenghts, model.times]
        if self.starburst is None:
            model = SynthesisModel(source_sed='leitherer1999',source_Z=self.pf['source_Z'], source_ssp=False)
            self.starburst = [model.data, model.wavelenghts, model.times]

        t_space = np.linspace(max(self.bpass[2][0], self.starburst[2][0]),
                    min(self.bpass[2][-1], self.starburst[2][-1]),
                    min(len(self.bpass[2]), len(self.starburst[2])))
        return t_space
