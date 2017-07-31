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
        
        self.redshifts = self.z = \
            np.array(np.sort(self.pf['powspec_redshifts'])[-1::-1], dtype=np.float64)
        
           
                
        #pb = ProgressBar(self.z.size, use=self.pf['progress_bar'])
        #pb.start()
        
        #self.mean_history.run()
        
        N = self.z.size
        pb = self.pb = ProgressBar(N, use=self.pf['progress_bar'], 
            name='ps-21cm')
        pb.start()


        all_ps = []                        
        for i, (z, data) in enumerate(self.step()):

            # Do stuff
            all_ps.append(data)

            if i == 0:
                keys = data.keys()

            pb.update(i)

        pb.finish()
        
        # Re-organize to look like Global21cm outputs, i.e., dictionary
        # with one key per quantity of interest, and in this case a 2-D
        # array of shape (z, k)
        data_proc = {}
        for key in keys:
                                                
            twod = False
            if type(all_ps[0][key]) == np.ndarray:
                twod = True
                                        
            # Second dimension is usually k
            if twod:
                tmp = np.zeros((len(self.z), all_ps[0][key].shape[0]))
            else:
                tmp = np.zeros_like(self.z)
                
            for i, z in enumerate(self.z):
                tmp[i] = all_ps[i][key]
                
            data_proc[key] = tmp
        
        self.history = data_proc

    def _regrid_and_fft(self, x, y, k, N=1000000):
        # Here, x is dr, and is supplied in descending order
        x_g = np.linspace(x.min(), x.max(), N)
        y_g = np.interp(x_g, x[-1::-1], y[-1::-1].real)
                
        # y_g is gridded, corresponds to ascending dr's, i.e., descending k
        ft_g = np.fft.fft(y_g).real
        
        # Flip just so interpolation works, otherwise we're good because
        # output should corresponding to ascending k values.
        ft_o = np.interp(k, 2. * np.pi / x_g[-1::-1], ft_g[-1::-1])

        return ft_o

    def _regrid_and_ifft(self, x, y, dr, N=1000000):
        if np.allclose(np.diff(np.diff(x)), 0):
            x_g = x#
            y_g = y#
            
            ft_o = np.fft.ifft(y_g)
        else:
            # This is really k, assured to be in ascending order
            x_g = np.linspace(x.min(), x.max(), N)
            # power spectrum interpolated onto new grid of k.
            # Assumed that k is ascending
            y_g = np.interp(x_g, x, y.real)
            
            # Corresponds to ascending k-values, i.e., descending dr values          
            ft_g = np.fft.ifft(y_g).real
            
            # These dr values are descending
            dr_from_k = 2. * np.pi / x_g
                        
            # Must flip for interpolation to work, but flip back after
            # because dr values are provided in descending order
            ft_o = np.interp(dr, dr_from_k[-1::-1], ft_g[-1::-1])[-1::-1]

        return ft_o

    def step(self):
        """
        Generator for the power spectrum.
        """

        # Setup linear grid of radii
        #R = np.linspace(0.1, 1e2, 1e3)
        #k = np.fft.fftfreq(R.size, np.diff(R)[0])
        k = self.k
        dr = self.dr = 2. * np.pi / k
        
        for i, z in enumerate(self.z):
                        
            data = {}            
                
            data['k'] = k
            data['z'] = z
            data['cf_xx'] = np.zeros_like(k)
            data['cf_dd'] = np.zeros_like(k)
            data['cf_xd'] = np.zeros_like(k)            
                        
            zeta = 0.0            
            for j, pop in enumerate(self.pops):
                
                """
                Possible approach: loop over pops to determine *total* 
                zeta_ion, etc...will that make sense? THEN compute PS.
                """                
                
                # Ionization fluctuations
                if pop.pf['pop_ion_fluct'] and self.pf['include_xcorr']:
                    R_b, M_b, bsd = self.field.BubbleSizeDistribution(z)

                    cf_xx = self.field.CorrelationFunction(z,
                        field_1='x', field_2='x', dr=dr, popid=j)
                                        
                    data['cf_xx'] += cf_xx
                    data.update({'R_b': R_b, 'M_b': M_b, 'bsd':bsd})
                else:
                    pass
                    #ps_xx = xi_xx = 0.0

                # Density fluctuations
                if self.pf['include_density_fl'] and self.pf['include_acorr']:
                    # Halo model
                    #ps_dd = pop.halos.PowerSpectrum(z, k)
                    ps_dd = pop.halos.PowerSpectrum(z, k)
                    data['ps_dd'] = ps_dd
                    
                    #data['cf_dd'] = xi_dd = np.fft.fftshift(np.fft.ifft(ps_dd))
                    xi_dd = self._regrid_and_ifft(k, ps_dd, dr)
                    #xi_dd = np.fft.ifft(ps_dd)
                    data['cf_dd'] += xi_dd.real
                else:
                    pass
                    #ps_dd = 0.0

                    
                
                if self.pf['include_temp_fl'] and self.pf['include_acorr']:
                    cf_TT = self.field.CorrelationFunction(z,
                        field_1='c', field_2='c', dr=dr, popid=j)
                                        
                    data['cf_TT'] = cf_TT
                    
                    # Must convert from temperature perturbation 
                    # to contrast perturbation                
                
                if not self.pf['include_xcorr']:
                    break
                
                                        
                if self.pf['include_temp_fl'] and self.pf['include_ion_fl']:
                    cf_xT = self.field.CorrelationFunction(z,
                        field_1='x', field_2='c', dr=dr, popid=j)
                                        
                    data['cf_xT'] = cf_xT
                                        
                # Cross-correlation terms...
                # Density-ionization cross correlation
                if (self.pf['include_density_fl'] and pop.pf['pop_ion_fluct']) and \
                    self.pf['include_xcorr']:
                    pass
                else:
                    pass
                    #data['cf_xd'] += 0.0
                    #ps_xd = xi_xd = 0.0
                    
                #else:
                #    data['ps_dd'] = np.zeros_like(k)
                
                
                
                
                
                
                break

            # Will we ever have multiple populations contributing to 
            # different fluctuations? Or, should we require that all ionizing
            # sources be parameterized in such a way that we only 
            # calculate, e.g., ps_xx once.

            # Global quantities
            QHII = self.field.BubbleFillingFactor(z)
            #QHII = np.interp(z, self.mean_history['z'][-1::-1], 
            #    self.mean_history['cgm_h_2'][-1::-1])
            data['QHII'] = QHII

            # Here, add together the power spectra with various Beta weights
            # to get 21-cm power spectrum
            
            Tk = np.interp(z, self.mean_history['z'][-1::-1], 
                self.mean_history['igm_Tk'][-1::-1])
            Ja = np.interp(z, self.mean_history['z'][-1::-1],
                self.mean_history['Ja'][-1::-1])
            xHII, ne = [0] * 2
            
            # Assumes saturation
            Ts = Tk
            Tcmb = self.cosm.TCMB(z)
            data['cf_cc'] = (Ts / (Ts - Tcmb)) \
                - (Tcmb / (Ts - Tcmb)) / (1. + data['cf_TT'])
            
            # Add beta factors to dictionary
            #for f1 in ['x', 'd']:
            #    func = self.hydr.__getattribute__('beta_%s' % f1)
            #    data['beta_%s' % f1] = func(z, Tk, xHII, ne, Ja)
            
            # This is just expanding out the terms for the 
            # ensemble averaged brightness temperature fluctuation,
            # FT{<d_21(k) d_21(k')>}
            
            Tbar = np.interp(z, self.gs.history['z'][-1::-1], 
                self.gs.history['dTb'][-1::-1])
                
            xi_xx = data['cf_xx']
            xi_dd = data['cf_dd']
            xi_xd = data['cf_xd']
            data['cf_21'] = xi_xx * (1. + xi_dd) + QHII**2 * xi_dd + \
                xi_xd * (xi_xd + 2. * QHII) 
            
            data['dTb'] = Tbar
            #data['cf_21'] -= QHII**2
            #data['cf_21'] *= self.hydr.T0(z)**2
                                                              
            data['ps_21'] = self._regrid_and_fft(dr, data['cf_21'], self.k)
            
            yield z, data


