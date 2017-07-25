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
            self.gs.run()
            self._mean_history = self.gs.history
        
        return self._mean_history
            
    @mean_history.setter
    def mean_history(self, value):
        self._mean_history = value
        
    @property
    def pops(self):
        return self.gs.medium.field.pops
    
    @property
    def grid(self):
        return self.gs.medium.field.grid
    
    @property
    def hydr(self):
        return self.grid.hydr
        
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
        
        #self.mean_history.run()
        
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
        
        # Setup linear grid of radii
        #R = np.linspace(0.1, 1e2, 1e3)
        #k = np.fft.fftfreq(R.size, np.diff(R)[0])
        k = self.k
        dr = 2. * np.pi / k
        
        for i, z in enumerate(self.z):
                        
            data = {}            
                        
            for j, pop in enumerate(self.pops):
                
                """
                Possible approach: loop over pops to determine *total* 
                zeta_ion, etc...will that make sense? THEN compute PS.
                """
                
                if pop.pf['pop_ion_fluct']:
                    R_b, M_b, bsd = self.field.BubbleSizeDistribution(z)
                    
                    #ps_xx = self.field.PowerSpectrum(z, 
                    #    field_1='h_2', field_2='h_2', k=self.k, popid=j)
                    cf_xx = self.field.CorrelationFunction(z,
                        field_1='h_2', field_2='h_2', dr=dr, popid=j)
                    ps_xx = np.fft.fft(cf_xx)
                    

                    data['ps_xx'] = ps_xx
                    data['cf_xx'] = cf_xx
                    data.update({'R_b': R_b, 'M_b': M_b, 'bsd':bsd})
                    data['k'] = k
                #else:
                #    data['ps_xx'] = np.zeros_like(k)
                    
                if pop.pf['pop_dens_fluct']:
                    #cf_dd = self.field.CorrelationFunction(z,
                    #    field_1='d', field_2='d', R=R, popid=j)
                    #data['cf_dd'] = cf_dd
                    
                    MF = pop.halos.MF
                    MF.update(z=z)

                    ps_mm = MF.power
                    lnk = np.log(MF.k)
                    data['ps_dd'] = np.interp(np.log(k), lnk, ps_mm)
                #else:
                #    data['ps_dd'] = np.zeros_like(k)                 
                                        
                # Cross-correlation terms...

            # Will we ever have multiple populations contributing to 
            # different fluctuations? Or, should we require that all ionizing
            # sources be parameterized in such a way that we only 
            # calculate, e.g., ps_xx once.

            # Global quantities
            QHII = self.field.BubbleFillingFactor(z)
            data['QHII'] = QHII

            # Here, add together the power spectra with various Beta weights
            # to get 21-cm power spectrum
            
            Tk = np.interp(z, self.mean_history['z'][-1::-1], 
                self.mean_history['igm_Tk'][-1::-1])
            Ja = np.interp(z, self.mean_history['z'][-1::-1], 
                self.mean_history['Ja'][-1::-1])
            xHII, ne = [0] * 2
            
            # Add beta factors to dictionary
            for f1 in ['x', 'd']:
                func = self.hydr.__getattribute__('beta_%s' % f1)
                data['beta_%s' % f1] = func(z, Tk, xHII, ne, Ja)
            
            # This is just expanding out the terms for the 
            # ensemble averaged brightness temperature fluctuation,
            # FT{<d_21(k) d_21(k')>}
            data['ps_21'] = np.zeros_like(k)
            for i, f1 in enumerate(['x', 'd']):
                for j, f2 in enumerate(['x', 'd']):
                    
                    if 'ps_%s%s' % (f1, f2) not in data:
                        continue
                    
                    # No double counting please
                    if j > i:
                        continue
                    
                    if f1 != f2 and (not self.pf['include_cross_correlations']):
                        continue
                    if f1 == f2 and (not self.pf['include_auto_correlations']):
                        continue 
                                      
                    #coeff = data['beta_%s' % f1] * data['beta_%s' % f2]    
                    coeff = 1.
                    data['ps_21'] += coeff * data['ps_%s%s' % (f1, f2)].real

            yield z, data


