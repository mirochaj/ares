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
    
    #@property
    #def k(self):
    #    if not hasattr(self, '_k'):
    #        if self.pf['output_wavenumbers'] is not None:
    #            self._k = self.pf['output_wavenumbers']
    #            self._logk = np.log10(self._k)
    #        else:
    #            lkmin = self.pf['powspec_logkmin']
    #            lkmax = self.pf['powspec_logkmax']
    #            dlk = self.pf['powspec_dlogk']
    #            self._logk = np.arange(lkmin, lkmax+dlk, dlk, dtype=float)
    #            self._k = 10.**self._logk
    #    return self._k
        
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
        
        self._data_pp = all_ps
        
        kmi, kma = self.k[self.k>0].min(), self.k[self.k>0].max()
        dlogk = self.pf['powspec_dlogk']
        rmi, rma = self.dr.min(), self.dr.max()
        dlogr = self.pf['powspec_dlogr']

        kfin = 10**np.arange(np.log10(kmi), np.log10(kma), dlogk)
        rfin = 10**np.arange(np.log10(rmi), np.log10(rma), dlogr)

        # Re-organize to look like Global21cm outputs, i.e., dictionary
        # with one key per quantity of interest, and in this case a 2-D
        # array of shape (z, k)
        data_proc = {}
        for key in keys:
            
            if key in ['k', 'dr']:
                continue

            down_sample = False
            twod = False
            if type(all_ps[0][key]) == np.ndarray:
                twod = True

                if ('cf' in key) or ('jp' in key):
                    xfin = rfin
                    x = self.dr
                    mask = Ellipsis
                    size = rfin.size
                    down_sample = True
                elif 'ps' in key:
                    xfin = kfin
                    mask = self.k > 0
                    x = self.k
                    size = kfin.size
                    down_sample = True
                else:
                    size = all_ps[i][key].size

            # Second dimension is k or dr
            if twod:
                tmp = np.zeros((len(self.z), size))
            else:
                tmp = np.zeros_like(self.z)
                
            for i, z in enumerate(self.z):
            
                # Downsample otherwise plotting takes forever later.
                if twod and down_sample:
                    tmp[i] = np.interp(xfin, x[mask], all_ps[i][key].real[mask])
                else:
                    tmp[i] = all_ps[i][key]
                
            data_proc[key] = tmp
        
        data_proc['k'] = kfin
        data_proc['dr'] = rfin
        
        self.history = data_proc

    def _temp_to_contrast(self, z, T):
        """
        Convert a temperature to a contrast
        """
        
        # Grab mean temperature
        Tk = Ts = np.interp(z, self.mean_history['z'][-1::-1], 
            self.mean_history['igm_Tk'][-1::-1])
        Tcmb = self.cosm.TCMB(z)
        
        delta_T = Ts / T - 1.
        
        delta_C = (Ts / (Ts - Tcmb)) \
            - (Tcmb / (Ts - Tcmb)) / (1. + delta_T) - 1.

        return delta_C

    def step(self):
        """
        Generator for the power spectrum.
        """

        # Setup linear grid of radii
        #R = np.linspace(0.1, 1e2, 1e3)
        #k = np.fft.fftfreq(R.size, np.diff(R)[0])
        #k = self.k
        #dr = self.dr = 1. / k
        
        step = 1e-4
        #self.dr = dr = np.arange(1e-3, 1e2+step, step)
        self.dr = dr = self.pf['fft_scales']
        self.k = k = np.fft.fftfreq(dr.size, step)
        
        k_mi, k_ma = k.min(), k.max()
        dlogk = self.pf['powspec_dlogk']
        k_pos = 10**np.arange(np.log10(k[k>0].min()), np.log10(k_ma)+dlogk, dlogk)
        #self.k_coarse = 10**np.arange(np.log10(k_mi), np.log10(k_ma)+dlogk, dlogk)
        #self.k_coarse = np.concatenate(([0], k_pos, -1 * k_pos[-1::-1]))
        self.k_coarse = np.concatenate((-1 * k_pos[-1::-1], [0], k_pos))
        self.k_pos = k_pos
        
        r_mi, r_ma = dr.min(), dr.max()
        dlogr = self.pf['powspec_dlogr']
        self.dr_coarse = 10**np.arange(np.log10(r_mi), np.log10(r_ma)+dlogr, dlogr)

        for i, z in enumerate(self.z):

            data = {}
                
            ## 
            # First, loop over populations and determine total
            # UV and X-ray outputs. 
            ##          
            
            # Prepare for the general case of Mh-dependent things
            zeta = 0.0#np.zeros_like(self.pops[0].halos.M)
            #Tpro = None           
            for j, pop in enumerate(self.pops):
                if pop.pf['pop_ion_fl']:
                    erg_per_ph = pop.src.erg_per_phot(13.6, 24.6)
                    Nion = pop.yield_per_sfr * self.cosm.g_per_b / erg_per_ph
                    zeta += Nion * pop.pf['pop_fesc'] * pop.pf['pop_fstar']
                                       
                    print j, zeta                   

                if pop.pf['pop_temp_fl']:
                    pass
                
                if pop.pf['pop_lya_fl']:
                    pass
                                
            data['k'] = k
            data['k_cr'] = self.k_coarse
            data['z'] = z
            
            data['cf_xd'] = np.zeros_like(k)
                
            # Ionization fluctuations
            if self.pf['include_ion_fl'] and self.pf['include_acorr']:
                R_b, M_b, bsd = self.field.BubbleSizeDistribution(z, zeta)
            
                p_ii = self.field.JointProbability(z, self.dr_coarse, 
                    zeta, term='ii', Tprof=None)
                                    
                # Interpolate onto fine grid                
                data['jp_ii'] = np.interp(dr, self.dr_coarse, p_ii)
                
                # Dimensions here are set by mass-sampling in HMF table
                data.update({'R_b': R_b, 'M_b': M_b, 'bsd':bsd})
            else:
                p_ii = data['jp_ii'] = np.zeros_like(dr)
            
            # Density fluctuations
            if self.pf['include_density_fl'] and self.pf['include_acorr']:
                # Halo model
                ps_posk = self.pops[0].halos.PowerSpectrum(z, self.k_pos)
                
                #data['ps_dd_cr'] = ps_dd
                
                # Must interpolate to uniformly (in real space) sampled
                # grid points to do inverse FFT
                ps_fold = np.concatenate((ps_posk[-1::-1], [0], ps_posk))
                ps_dd = np.interp(self.k, self.k_coarse, ps_fold)
                data['ps_dd'] = ps_dd
                data['cf_dd'] = np.fft.ifft(data['ps_dd'])
            else:
                data['cf_dd'] = data['ps_dd'] = np.zeros_like(dr)
            
            # Temperature fluctuations                
            if self.pf['include_temp_fl'] and self.pf['include_acorr']:
                p_hh = self.field.JointProbability(z, self.dr_coarse, 
                    zeta, term='hh', Tprof=None)
                                    
                data['jp_hh'] = np.interp(dr, self.dr_coarse, p_hh)
                
                # Must convert from temperature perturbation 
                # to contrast perturbation                
            else:
                p_hh = data['jp_hh'] = np.zeros_like(dr)
            
            ##
            # Cross-correlations
            ##
            if self.pf['include_xcorr']:
            
                # Cross-correlation terms...
                # Density-ionization cross correlation
                if (self.pf['include_density_fl'] and self.pf['include_ion_fl']):
                    pass
                else:
                    pass
            
                if self.pf['include_temp_fl'] and self.pf['include_ion_fl']:
                    p_ih = self.field.JointProbability(z, self.dr_coarse, 
                        zeta, term='ih')
                    data['jp_ih'] = np.interp(dr, self.dr_coarse, p_ih)
                else:
                    data['jp_ih'] = np.zeros_like(k)
            else:
                p_ih = data['jp_ih'] = np.zeros_like(k)

            ##
            # Compute global quantities, 21-cm PS (from all component CFs),
            # and finish up.
            ##

            # Global quantities
            QHII = self.field.BubbleFillingFactor(z, zeta)
            #QHII = np.interp(z, self.mean_history['z'][-1::-1],
            #    self.mean_history['cgm_h_2'][-1::-1])
            data['QHII'] = QHII
            
            if self.pf['include_temp_fl']:
                Qhot = self.field.BubbleShellFillingFactor(z, zeta)
                data['Qhot'] = Qhot
                Cbar = self._temp_to_contrast(z, self.pf['bubble_shell_temp'])
                data['Tbar'] = Qhot * Cbar
            else:
                data['Qhot'] = Qhot = Cbar = 0.0

            data['cf_xx'] = data['jp_ii'] - QHII**2
            data['jp_cc'] = data['jp_hh'] * Cbar**2
            data['cf_cc'] = data['jp_cc'] - Cbar**2 * Qhot**2
            data['cf_xc'] = data['jp_ih'] - Cbar * Qhot * QHII

            # Here, add together the power spectra with various Beta weights
            # to get 21-cm power spectrum
            
            Tk = np.interp(z, self.mean_history['z'][-1::-1], 
                self.mean_history['igm_Tk'][-1::-1])
            Ja = np.interp(z, self.mean_history['z'][-1::-1],
                self.mean_history['Ja'][-1::-1])
            xHII, ne = [0] * 2
            
            # Assumes strong coupling. Mapping between temperature 
            # fluctuations and contrast fluctuations.
            Ts = Tk
            Tcmb = self.cosm.TCMB(z)
            #data['cf_cc'] = (Ts / (Ts - Tcmb)) \
            #    - (Tcmb / (Ts - Tcmb)) / (1. + data['cf_TT']) - 1.
            #
            # Add beta factors to dictionary
            #for f1 in ['x', 'd']:
            #    func = self.hydr.__getattribute__('beta_%s' % f1)
            #    data['beta_%s' % f1] = func(z, Tk, xHII, ne, Ja)
            
            # This is just expanding out the terms for the 
            # ensemble averaged brightness temperature fluctuation,
            # FT{<d_21(k) d_21(k')>}
            
            xavg = np.interp(z, self.gs.history['z'][-1::-1], 
                self.gs.history['cgm_h_2'][-1::-1])
            Tbar = np.interp(z, self.gs.history['z'][-1::-1], 
                self.gs.history['dTb'][-1::-1]) / (1. - xavg)
                
            xi_xx = data['cf_xx']
            xi_dd = data['cf_dd']
            xi_xd = data['cf_xd']
                        
            # This is Eq. 11 in FZH04
            xbar = 1. - QHII
            data['cf_21_sat'] = xi_xx * (1. + xi_dd) + xbar**2 * xi_dd + \
                xi_xd * (xi_xd + 2. * xbar) 
                
            ##
            # MODIFY CORRELATION FUNCTION depending on saturation
            ##    
            if not self.pf['include_temp_fl']:
                data['cf_21'] = data['cf_21_sat']
            else:
                # Simplified for now
                data['cf_21'] = data['cf_21_sat'] + QHII**2 + 2 * data['cf_xc']
                
            
            data['dTb0'] = Tbar
            #data['cf_21'] -= QHII**2
            #data['cf_21'] *= self.hydr.T0(z)**2
                         
            #data['ps_21_sat'] = self._regrid_and_fft(dr, data['cf_21_sat'], self.k)                                                  
            #data['ps_21'] = self._regrid_and_fft(dr, data['cf_21'], self.k)
            #
            data['ps_21'] = np.fft.fft(data['cf_21'])
            
            
            yield z, data


