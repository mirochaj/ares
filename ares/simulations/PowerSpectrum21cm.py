#import os
import numpy as np
from .Global21cm import Global21cm
#from ..util.ReadData import _sort_history
from ..util import ParameterFile, ProgressBar
#from ..analysis.BlobFactory import BlobFactory
#from ..physics.Constants import nu_0_mhz, E_LyA
from ..solvers import FluctuatingBackground
from ..analysis.PowerSpectrum import PowerSpectrum as AnalyzePS

#
#try:
#    import dill as pickle
#except ImportError:
#    import pickle

defaults = \
{
 'load_ics': True,
}

class PowerSpectrum21cm(AnalyzePS):
    def __init__(self, **kwargs):
        """ Set up a power spectrum calculation. """
        
        # See if this is a tanh model calculation
        #is_phenom = self._check_if_phenom(**kwargs)

        kwargs.update(defaults)
        if 'problem_type' not in kwargs:
            kwargs['problem_type'] = 101

        self.kwargs = kwargs

    @property
    def mean_history(self):
        if not hasattr(self, '_mean_history'):
            self._mean_history = Global21cm(**self.kwargs)
        
        return self._mean_history
            
    @mean_history.setter
    def mean_history(self, value):
        self._mean_history = value
        
    @property
    def pops(self):
        return self.mean_history.medium.field.pops
    
    @property
    def grid(self):
        return self.mean_history.medium.field.grid
        
    @property
    def pf(self):
        if not hasattr(self, '_pf'):
            self._pf = ParameterFile(**self.kwargs)
        return self._pf

    @pf.setter
    def pf(self, value):
        self._pf = value

    @property
    def gs(self):
        if not hasattr(self, '_gs'):
            self._gs = Global21cm(**self.kwargs)
        return self._gs
        
    @gs.setter
    def gs(self, value):
        """ Set global 21cm instance by hand. """
        self._gs = value
    
    #@property
    #def global_history(self):
    #    if not hasattr(self, '_global_') 
    
    @property
    def k(self):
        if not hasattr(self, '_k'):
            if self.pf['output_wavenumbers'] is not None:
                self._k = self.pf['output_wavenumbers']
                self._logk = np.log10(self._k)
            else:
                lkmin = self.pf['powspec_logkmin']
                lkmax = self.pf['powspec_logkmax']
                dlk = self.pf['powspec_dlogk']
                self._logk = np.arange(lkmin, lkmax+dlk, dlk, dtype=float)
                self._k = 10.**self._logk
        return self._k
        
    @property
    def field(self):
        if not hasattr(self, '_field'):
            self._field = FluctuatingBackground(**self.kwargs)
                
        return self._field
    
    def run(self):
        """
        Run a simulation, compute power spectrum at each redshift.
        
        Returns
        -------
        Nothing: sets `history` attribute.
        
        """
        
        self.redshifts = self.z = np.sort(self.pf['powspec_redshifts'])[-1::-1]
                
        #pb = ProgressBar(self.z.size, use=self.pf['progress_bar'])
        #pb.start()
        
        all_ps = []                        
        for i, (z, data) in enumerate(self.step()):
                          
            #pb.update(i)
                    
            # Do stuff
            all_ps.append(data)

        #pb.finish()
        
        # Will be (z, k)
        self.history = np.array(all_ps)
        
    def step(self):
        """
        Generator for the power spectrum.
        """
        
        #ps = np.zeros_like(k)
        for i, z in enumerate(self.z):

            ps_xx = np.zeros_like(self.k)
            ps_xd = np.zeros_like(self.k)
            ps_dd = np.zeros_like(self.k)
            
            cf_xx = np.zeros_like(self.k)
            cf_xd = np.zeros_like(self.k)
            cf_dd = np.zeros_like(self.k)

            for j, pop in enumerate(self.pops):
                
                """
                Possible approach: loop over pops to determine *total* 
                zeta_ion, etc...will that make sense?
                """
                
                if pop.pf['pop_ion_fluct']:
                    R_b, M_b, bsd = self.field.BubbleSizeDistribution(z)
                    
                    #ps_xx[:] = self.field.PowerSpectrum(z, self.k, 
                    #    field_1='h_2', field_2='h_2', popid=j)
                    #cf_xx[:] = self.field.CorrelationFunction(z, self.k, 
                    #    field_1='h_2', field_2='h_2', popid=j)    

            # Dictionary-ify everything
            data = {'ps_xx': ps_xx, 'ps_xd': ps_xd, 'ps_dd': ps_dd,
                    'cf_xx': cf_xx, 'cf_xd': cf_xd, 'cf_dd': cf_dd}

            data.update({'R_b': R_b, 'M_b': M_b, 'bsd':bsd})

            # Global quantities
            QHII = self.field.BubbleFillingFactor(z)
            data['QHII'] = QHII

            # Here, add together the power spectra with various Beta weights
            # to get 21-cm power spectrum

            yield z, data


