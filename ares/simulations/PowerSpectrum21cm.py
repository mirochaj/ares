import os
import pickle
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

        self.z = np.array(np.sort(self.pf['powspec_redshifts'])[-1::-1], 
            dtype=np.float64)

        N = self.z.size
        pb = self.pb = ProgressBar(N, use=self.pf['progress_bar'], 
            name='ps-21cm')

        all_ps = []                        
        for i, (z, data) in enumerate(self.step()):

            # Do stuff
            all_ps.append(data)

            if i == 0:
                keys = data.keys()
                
            if not pb.has_pb:
                pb.start()

            pb.update(i)

        pb.finish()
        
        self._data_pp = all_ps
        
        kmi, kma = self.k[self.k>0].min(), self.k[self.k>0].max()
        dlogk = self.pf['powspec_dlogk']
        rmi, rma = self.R.min(), self.R.max()
        dlogr = self.pf['powspec_dlogr']

        kfin = 10**np.arange(np.log10(kmi), np.log10(kma), dlogk)
        rfin = 10**np.arange(np.log10(rmi), np.log10(rma), dlogr)

        # Re-organize to look like Global21cm outputs, i.e., dictionary
        # with one key per quantity of interest, and in this case a 2-D
        # array of shape (z, k)
        data_proc = {}
        for key in keys:
            
            if key in ['k', 'R', 'k_cr', 'R_cr']:
                continue

            down_sample = False
            twod = False
            if type(all_ps[0][key]) == np.ndarray:
                twod = True

                if ('cf' in key) or ('jp' in key) or ('ev' in key):
                    xfin = rfin
                    x = self.R
                    mask = Ellipsis
                    size = rfin.size
                    down_sample = True
                elif 'ps' in key:
                    xfin = kfin
                    mask = self.k > 0
                    x = self.k
                    size = kfin.size
                    # Already down-sampled?
                    down_sample = True
                else:
                    size = all_ps[i][key].size

            # Second dimension is k or R
            if twod:
                tmp = np.zeros((len(self.z), size))
            else:
                tmp = np.zeros_like(self.z)
                
            for i, z in enumerate(self.z):
            
                # Downsample otherwise plotting takes forever later.
                if twod and down_sample:
                    tmp[i] = np.interp(np.log(xfin), np.log(x[mask]), 
                        all_ps[i][key].real[mask])
                else:
                    tmp[i] = all_ps[i][key]
                
            data_proc[key] = tmp
        
        data_proc['k'] = kfin
        data_proc['R'] = rfin
        
        self.history = data_proc
    
    @property
    def include_con_fl(self):
        if not hasattr(self, '_include_con_fl'):
            self._include_con_fl = self.pf['include_temp_fl'] or \
                self.pf['include_lya_fl']
        return self._include_con_fl

    def _temp_to_contrast(self, z, T):
        """
        Convert a temperature to a contrast.
        
        ..note:: For coupled regions, T will be the mean *kinetic* temperature.
        
        """
        
        # Grab mean spin temperature
        Ts = np.interp(z, self.mean_history['z'][-1::-1], 
            self.mean_history['igm_Ts'][-1::-1])
        Tk = np.interp(z, self.mean_history['z'][-1::-1], 
            self.mean_history['igm_Tk'][-1::-1])    
        Tcmb = self.cosm.TCMB(z)
        
        T = max(T, Tk)
        
        # The actual spin temperature anywhere is the mean * (1 + delta_T)
        delta_T = T / Ts - 1.
                
        # Don't allow negative contrasts! That would imply that the heated
        # shell around an HII region is *colder* than the temperature of the
        # bulk IGM.
        if not self.pf['include_lya_fl']:
            # This is really an indication that the modeled temperature
            # in heated regions is unrealistic
            delta_T = max(delta_T, 0.)
        #if not self.pf['include_temp_fl']:
        #    delta_T = min(delta_T, 0.)
        
        # Convert temperature perturbation to contrast perturbation.
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
        
        step = np.diff(self.pf['fft_scales'])[0]
        #self.dr = dr = np.arange(1e-3, 1e2+step, step)
        self.R = R = self.pf['fft_scales']
        self.k = k = np.fft.fftfreq(R.size, step)
        logR = np.log(R)
        
        k_mi, k_ma = k.min(), k.max()
        dlogk = self.pf['powspec_dlogk']
        k_pos = 10**np.arange(np.log10(k[k>0].min()), np.log10(k_ma)+dlogk, dlogk)
        #self.k_coarse = 10**np.arange(np.log10(k_mi), np.log10(k_ma)+dlogk, dlogk)
        #self.k_coarse = np.concatenate(([0], k_pos, -1 * k_pos[-1::-1]))
        #self.k_pos = k_pos
        
        self.k_pos = self.pops[0].halos.k_cr_pos
        self.R_cr = self.pops[0].halos.R_cr
        self.logR_cr = np.log(self.R_cr)
        
        for i, z in enumerate(self.z):

            data = {}
                
            ## 
            # First, loop over populations and determine total
            # UV and X-ray outputs. 
            ##          
            
            # Prepare for the general case of Mh-dependent things
            Nion = np.zeros_like(self.field.halos.M)
            Nlya = np.zeros_like(self.field.halos.M)
            zeta = np.zeros_like(self.field.halos.M)
            zeta_lya = np.zeros_like(self.field.halos.M)
            #Tpro = None           
            for j, pop in enumerate(self.pops):
                pop_zeta = pop.IonizingEfficiency(z=z)

                if pop.is_src_ion_fl:

                    if type(pop_zeta) is tuple:
                        _Mh, _zeta = pop_zeta
                        zeta += np.interp(self.field.halos.M, _Mh, _zeta)
                        Nion += pop.src.Nion
                    else:
                        zeta += pop_zeta
                        Nion += pop.pf['pop_Nion']
                        Nlya += pop.pf['pop_Nlw']
                        
                    zeta = np.maximum(zeta, 1.)    
                            
                if pop.is_src_heat_fl:
                    pass

                if pop.is_src_lya_fl:
                    Nlya += pop.pf['pop_Nlw']
                    #Nlya += pop.src.Nlw

            # Only used if...powspec_lya_method==0?
            zeta_lya += zeta * (Nlya / Nion)
            
            ##
            # Make scalar if it's a simple model
            ##
            if np.all(np.diff(zeta) == 0):
                zeta = zeta[0]
            if np.all(np.diff(zeta_lya) == 0):
                zeta_lya = zeta_lya[0]

            ##
            # First: some global quantities we'll need
            ##
            avg_Tk = np.interp(z, self.mean_history['z'][-1::-1],
                self.mean_history['igm_Tk'][-1::-1])
            #xibar = np.interp(z, self.mean_history['z'][-1::-1],
            #    self.mean_history['cgm_h_2'][-1::-1])
            #Qi = xibar
            
            #if self.pf['include_ion_fl']:
            #    if self.pf['powspec_rescale_Qion']:
            #        xibar = min(np.interp(z, self.pops[0].halos.z,
            #            self.pops[0].halos.fcoll_Tmin) * zeta, 1.)
            #        Qi = xibar
            #        
            #        xibar = np.interp(z, self.mean_history['z'][-1::-1],
            #            self.mean_history['cgm_h_2'][-1::-1])
            #        
            #    else:
            #        Qi = self.field.BubbleFillingFactor(z, zeta)
            #        xibar = 1. - np.exp(-Qi)
            #else:
            #    Qi = 0.
            
            Qi = self.field.BubbleFillingFactor(z, zeta)
            xibar = np.interp(z, self.mean_history['z'][-1::-1],
                self.mean_history['cgm_h_2'][-1::-1])
                
            xbar = 1. - xibar
            data['Qi'] = Qi
            data['xibar'] = xibar
            
            data['k'] = k
            data['dr'] = self.R
            data['dr_cr'] = self.R_cr
            data['k_cr'] = self.k_pos
            data['z'] = z
            
            data['zeta'] = zeta
            
            logr = np.log(self.R)
            
            ##
            # Density fluctuations
            ##
            if self.pf['include_density_fl'] and self.pf['include_acorr']:
                ps_dd = self.pops[0].halos.PowerSpectrum(z, self.k_pos)
                
                # PS is positive so it's OK to log-ify it
                data['ps_dd'] = np.exp(np.interp(np.log(np.abs(self.k)), 
                    np.log(self.k_pos), np.log(ps_dd)))
                
                # If this isn't tabulated, need to send in full dr array
                cf_c = self.pops[0].halos.CorrelationFunction(z, self.R_cr)
                
                # Need finer-grain resolution for this. Leave correlation
                # function in linear units for interpolation because it will 
                # be negative on large scales.
                data['cf_dd'] = np.interp(logr, np.log(self.R_cr), cf_c)
            else:
                data['cf_dd'] = data['ps_dd'] = np.zeros_like(R)

            ##    
            # Ionization fluctuations
            ##
            if self.pf['include_ion_fl'] and self.pf['include_acorr']:
                R_b, M_b, bsd = self.field.BubbleSizeDistribution(z, zeta)
            
                p_ii, p_ii_1, p_ii_2 = self.field.JointProbability(z, 
                    self.R_cr, zeta, term='ii', Tprof=None, data=data)
                
                # Interpolate onto fine grid
                data['jp_ii'] = np.interp(logR, self.logR_cr, p_ii)
                data['jp_ii_1'] = np.interp(logR, self.logR_cr, p_ii_1)
                data['jp_ii_2'] = np.interp(logR, self.logR_cr, p_ii_2)

                data['ev_ii'] = data['jp_ii']
                data['ev_ii_1'] = data['jp_ii_1']
                data['ev_ii_2'] = data['jp_ii_2']

                # Dimensions here are set by mass-sampling in HMF table
                data.update({'R_b': R_b, 'M_b': M_b, 'bsd':bsd})
            else:
                p_ii = data['jp_ii'] = data['ev_ii'] = np.zeros_like(R)

            ##
            # Temperature fluctuations                
            ##
            data['ev_coco'] = np.zeros_like(R)
            data['ev_coco_1'] = np.zeros_like(R)
            data['ev_coco_2'] = np.zeros_like(R)
            if self.pf['include_temp_fl'] and self.pf['include_acorr']:
                Qh = self.field.BubbleShellFillingFactor(z, zeta, zeta_lya)
                Ch = self._temp_to_contrast(z, self.pf['bubble_shell_temp'])
                data['Ch'] = Ch
                data['Qh'] = Qh
                data['avg_Ch'] = avg_Ch = Qh * Ch

                p_hh, p_hh_1, p_hh_2 = self.field.JointProbability(z, 
                    self.R_cr, zeta, term='hh', Tprof=None, data=data)

                data['jp_hh'] = np.interp(logR, self.logR_cr, p_hh)
                #data['jp_hh'] = np.minimum(1., data['jp_hh'])
                data['jp_hh_1'] = np.interp(logR, self.logR_cr, p_hh_1)
                data['jp_hh_2'] = np.interp(logR, self.logR_cr, p_hh_2)
              
                data['ev_coco'] += Ch**2 * data['jp_hh']
                data['ev_coco_1'] += Ch**2 * data['jp_hh_1']
                data['ev_coco_2'] += Ch**2 * data['jp_hh_2']
              
                data['avg_C'] = data['avg_Ch']

            else:
                p_hh = data['jp_hh'] = np.zeros_like(R)
                data['Ch'] = Ch = 0.0
                data['Qh'] = Qh = 0.0
                data['avg_Ch'] = 0.0
                data['avg_C'] = 0.0

            ##
            # Lyman-alpha fluctuations                
            ##    
            if self.pf['include_lya_fl']:
                if self.pf['include_acorr']:
                    Qc = self.field.BubbleFillingFactor(z, zeta, zeta_lya, lya=True)
                    
                    Cc = self._temp_to_contrast(z, avg_Tk)
                    data['Cc'] = Cc
                    data['Qc'] = Qc
                    data['avg_Cc'] = avg_Cc = Qc * Cc
                    p_cc, p_cc_1, p_cc_2 = self.field.JointProbability(z, 
                        self.R_cr, zeta, term='cc', Tprof=None, data=data, 
                        zeta_lya=zeta_lya)
                    
                    data['jp_cc'] = np.interp(logR, self.logR_cr, p_cc)
                    data['jp_cc_1'] = np.interp(logR, self.logR_cr, p_cc_1)
                    data['jp_cc_2'] = np.interp(logR, self.logR_cr, p_cc_2)
                    
                    # Should this be necessary?
                    data['jp_cc'] = np.minimum(1., data['jp_cc'])
                    
                    data['ev_coco'] += Cc**2 * data['jp_cc']
                    data['avg_C'] += data['avg_Cc']

                # This should maybe be moved elsewhere.
                if self.pf['include_temp_fl']:
                    p_hc, p_hc_1, p_hc_2 = self.field.JointProbability(z, 
                        self.R_cr, zeta, term='hc', Tprof=None, data=data, 
                        zeta_lya=zeta_lya)

                    data['jp_hc'] = np.interp(logR, self.logR_cr, p_hc)
                    data['jp_hc_1'] = np.interp(logR, self.logR_cr, p_hc_1)
                    data['jp_hc_2'] = np.interp(logR, self.logR_cr, p_hc_2)
                    data['ev_coco'] += Cc * Ch * data['jp_hc'] * min(Qh + Qc, 1.)
                    
                    if (Qh + Qc) > 1:
                        print "WARNING: Qh+Qc > 1"
                    
                else:
                    data['jp_hc'] = np.zeros_like(R)
            else:
                data['jp_cc'] = data['jp_hc'] = data['ev_cc'] = \
                    np.zeros_like(R)
                data['Cc'] = data['Qc'] = Cc = Qc = 0.0
                data['avg_Cc'] = 0.0
                
            ##
            # Cross-terms between ionization and contrast
            ##
            if self.include_con_fl and self.pf['include_ion_fl']:
                if self.pf['include_temp_fl']:
                    p_ih, p_ih_1, p_ih_2 = self.field.JointProbability(z, 
                        self.R_cr, zeta, term='ih', data=data)
                    data['jp_ih'] = np.interp(logR, self.logR_cr, p_ih)
                    data['jp_ih_1'] = np.interp(logR, self.logR_cr, p_ih_1)
                    data['jp_ih_2'] = np.interp(logR, self.logR_cr, p_ih_2)
                else:
                    data['jp_ih'] = np.zeros_like(R)
                    
                if self.pf['include_lya_fl']:
                    p_ic, p_ic_1, p_ic_2 = self.field.JointProbability(z, 
                        self.R_cr, zeta, term='ic', data=data, 
                        zeta_lya=zeta_lya)
                    data['jp_ic'] = np.interp(logR, self.logR_cr, p_ic)
                    data['jp_ic_1'] = np.interp(logR, self.logR_cr, p_ic_1)
                    data['jp_ic_2'] = np.interp(logR, self.logR_cr, p_ic_2)
                else:
                    data['jp_ic'] = np.zeros_like(R)
                        
                data['ev_ico'] = data['Ch'] * data['jp_ih'] \
                               + data['Cc'] * data['jp_ic']
            else:
                data['jp_ih'] = np.zeros_like(k)
                data['jp_ic'] = np.zeros_like(k)
                data['ev_ico'] = np.zeros_like(k)    
            
            ##
            # Cross-correlations
            ##
            if self.pf['include_xcorr']:

                ##
                # Cross-terms with density and (ionization, contrast)
                ##
                if self.pf['include_xcorr_wrt'] is None:
                    do_xcorr_xd = True
                else:
                    do_xcorr_xd = (self.pf['include_xcorr_wrt'] is not None) and \
                       ('density' in self.pf['include_xcorr_wrt']) and \
                       ('ion' in self.pf['include_xcorr_wrt'])
                
                if do_xcorr_xd:

                    # Cross-correlation terms...
                    # Density-ionization cross correlation
                    if (self.pf['include_density_fl'] and self.pf['include_ion_fl']):
                        p_id, p_id_1, p_id_2 = self.field.JointProbability(z, 
                            self.R_cr, zeta, term='id', data=data)
                        data['jp_id'] = np.interp(logR, self.logR_cr, p_id)
                        data['ev_id'] = data['jp_id']
                    else:
                        data['jp_id'] = data['ev_id'] = np.zeros_like(k)
                else:
                    data['jp_id'] = data['ev_id'] = np.zeros_like(k)        

                   
                ##
                # Cross-terms between density and contrast
                ##  
                if self.include_con_fl and self.pf['include_density_fl']:
                    #if self.pf['include_temp_fl']:
                    #    p_dh = self.field.JointProbability(z, self.R_cr, 
                    #        zeta, term='dh', data=data)
                    #    data['jp_dh'] = np.interp(R, self.R_cr, p_dh)
                    #else:
                    #    data['jp_dh'] = np.zeros_like(R)
                    #
                    #if self.pf['include_lya_fl']:
                    #    p_dc = self.field.JointProbability(z, self.R_cr, 
                    #        zeta, term='dc', data=data, zeta_lya=zeta_lya)
                    #    data['jp_dc'] = np.interp(R, self.R_cr, p_dc)
                    #else:
                    #    data['jp_dc'] = np.zeros_like(R)
                    #
                    #data['ev_dco'] = data['Ch'] * data['jp_ih'] \
                    #               + data['Cc'] * data['jp_ic']
                    
                    
                    data['ev_dco'] = data['ev_ico'] * data['ev_id'] # times delta
                    
                else:
                    data['jp_dh'] = np.zeros_like(k)
                    data['jp_dc'] = np.zeros_like(k)
                    data['ev_dco'] = np.zeros_like(k)                        
                    
            else:
                p_id = data['jp_id'] = data['ev_id'] = np.zeros_like(k)
                p_ih = data['jp_ih'] = data['ev_ih'] = np.zeros_like(k)
                p_ic = data['jp_ic'] = data['ev_ic'] = np.zeros_like(k)
                p_dh = data['jp_dh'] = data['ev_dh'] = np.zeros_like(k)
                p_dc = data['jp_dc'] = data['ev_dc'] = np.zeros_like(k)
                data['ev_ico'] = np.zeros_like(k)
                data['ev_dco'] = np.zeros_like(k)

            ##
            # Construct correlation functions from expectation values
            ##

            # Correlation function of ionized fraction and neutral fraction
            # are equivalent.
            data['cf_xx']   = data['ev_ii']   - xibar**2
            data['cf_xx_1'] = data['ev_ii_1'] - xibar**2
            data['cf_xx_2'] = data['ev_ii_2'] - xibar**2
            
            # Minus sign difference for cross term with density.
            data['cf_xd'] = -data['ev_id']
            
            # Construct correlation function (just subtract off exp. value sq.)
            data['cf_coco']   = data['ev_coco']   - data['avg_C']**2
            data['cf_coco_1'] = data['ev_coco_1'] - data['avg_C']**2
            data['cf_coco_2'] = data['ev_coco_2'] - data['avg_C']**2
            
            # Correlation between neutral fraction and contrast fields
            data['cf_xco'] = data['avg_C'] - data['ev_ico']
            
            data['cf_dco'] = data['ev_dco']

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
            
            # Add beta factors to dictionary
            #for f1 in ['x', 'd']:
            #    func = self.hydr.__getattribute__('beta_%s' % f1)
            #    data['beta_%s' % f1] = func(z, Tk, xHII, ne, Ja)
            
            # This is just expanding out the terms for the 
            # ensemble averaged brightness temperature fluctuation,
            # FT{<d_21(k) d_21(k')>}

            xavg = np.interp(z, self.gs.history['z'][-1::-1], 
                self.gs.history['cgm_h_2'][-1::-1])

            # Mean brightness temperature outside bubbles    
            Tbar = np.interp(z, self.gs.history['z'][-1::-1], 
                self.gs.history['dTb'][-1::-1]) / (1. - xavg)

            # Short-hand
            xi_xx = data['cf_xx']
            xi_dd = data['cf_dd']
            xi_xd = data['cf_xd']

            # This is Eq. 11 in FZH04
            data['cf_21_s'] = xi_xx * (1. + xi_dd) + xbar**2 * xi_dd + \
                xi_xd * (xi_xd + 2. * xbar)

            ##
            # MODIFY CORRELATION FUNCTION depending on Ts fluctuations
            ##    
            
            if (self.pf['include_temp_fl'] or self.pf['include_lya_fl']):

                ##
                # Let's start with low-order terms and build up from there.
                ##
                
                avg_xC = data['avg_C'] # ??

                # The two easiest terms in the unsaturated limit are those
                # not involving the density, <x x' C'> and <x x' C C'>.
                # Under the binary field(s) approach, we can just write
                # each of these terms down
                ev_xi_cop = Ch * data['jp_ih'] + Cc * data['jp_ic']
                
                cc = data['ev_coco']
                xx = 1. - 2. * xibar + data['ev_ii']
                xc = data['avg_C'] - ev_xi_cop
                xxc = xbar * data['avg_C'] - ev_xi_cop
                xxcc = xx * cc + avg_xC**2 + xc**2
                
                phi_u = 2. * xxc + xxcc
                    
                data['cf_21'] = data['cf_21_s'] + phi_u \
                    - 2. * xbar * avg_xC
                
            else:
                data['cf_21'] = data['cf_21_s']

            data['dTb0'] = Tbar
            data['ps_21'] = np.fft.fft(data['cf_21'])   
            
            # Save 21-cm PS as one and two-halo terms also
                     
            data['ps_xx'] = np.fft.fft(data['cf_xx'])
            data['ps_xx_1'] = np.fft.fft(data['cf_xx_1'])
            data['ps_xx_2'] = np.fft.fft(data['cf_xx_2'])
            data['ps_coco'] = np.fft.fft(data['cf_coco'])
            data['ps_coco_1'] = np.fft.fft(data['cf_coco_1'])
            data['ps_coco_2'] = np.fft.fft(data['cf_coco_2'])
            
            #if z == 10:
            #    import matplotlib.pyplot as pl
            #    
            #    pl.figure(3)
            #    pl.semilogx(R, data['cf_xx_1'])
            #    pl.semilogx(R, data['cf_xx_2'], ls='--')
            #    pl.ylim(-1e-4, 4e-4)
            #    print data['cf_xx_2']
            #    
            #    pl.figure(4)
            #    pl.semilogx(k, data['ps_xx_1'])
            #    pl.semilogx(k, data['ps_xx_2'], ls='--')
            #    pl.ylim(-1, 1)
            #    raw_input('<enter>')
            
            
            # These are all going to get downsampled before the end.

            yield z, data

    def save(self, prefix, suffix='pkl', clobber=False, fields=None):
        """
        Save results of calculation. Pickle parameter file dict.
    
        Notes
        -----
        1) will save files as prefix.history.suffix and prefix.parameters.pkl.
        2) ASCII files will fail if simulation had multiple populations.
    
        Parameters
        ----------
        prefix : str
            Prefix of save filename
        suffix : str
            Suffix of save filename. Can be hdf5 (or h5), pkl, or npz. 
            Anything else will be assumed to be ASCII format (e.g., .txt).
        clobber : bool
            Overwrite pre-existing files of same name?

        """

        self.gs.save(prefix, clobber=clobber, fields=fields)
    
        fn = '%s.fluctuations.%s' % (prefix, suffix)
    
        if os.path.exists(fn):
            if clobber:
                os.remove(fn)
            else: 
                raise IOError('%s exists! Set clobber=True to overwrite.' % fn)
    
        if suffix == 'pkl':         
            f = open(fn, 'wb')
            pickle.dump(self.history._data, f)
            f.close()
    
            try:
                f = open('%s.blobs.%s' % (prefix, suffix), 'wb')
                pickle.dump(self.blobs, f)
                f.close()
    
                if self.pf['verbose']:
                    print 'Wrote %s.blobs.%s' % (prefix, suffix)
            except AttributeError:
                print 'Error writing %s.blobs.%s' % (prefix, suffix)
    
        elif suffix in ['hdf5', 'h5']:
            import h5py
    
            f = h5py.File(fn, 'w')
            for key in self.history:
                if fields is not None:
                    if key not in fields:
                        continue
                f.create_dataset(key, data=np.array(self.history[key]))
            f.close()
    
        elif suffix == 'npz':
            f = open(fn, 'w')
            np.savez(f, **self.history._data)
            f.close()
    
            if self.blobs:
                f = open('%s.blobs.%s' % (prefix, suffix), 'wb')
                np.savez(f, self.blobs)
                f.close()
    
        # ASCII format
        else:            
            f = open(fn, 'w')
            print >> f, "#",
    
            for key in self.history:
                if fields is not None:
                    if key not in fields:
                        continue
                print >> f, '%-18s' % key,
    
            print >> f, ''
    
            # Now, the data
            for i in xrange(len(self.history[key])):
                s = ''
    
                for key in self.history:
                    if fields is not None:
                        if key not in fields:
                            continue
    
                    s += '%-20.8e' % (self.history[key][i])
    
                if not s.strip():
                    continue
    
                print >> f, s
    
            f.close()
    
        if self.pf['verbose']:
            print 'Wrote %s.fluctuations.%s' % (prefix, suffix)
        
        #write_pf = True
        #if os.path.exists('%s.parameters.pkl' % prefix):
        #    if clobber:
        #        os.remove('%s.parameters.pkl' % prefix)
        #    else: 
        #        write_pf = False
        #        print 'WARNING: %s.parameters.pkl exists! Set clobber=True to overwrite.' % prefix
    
        #if write_pf:
        #
        #    #pf = {}
        #    #for key in self.pf:
        #    #    if key in self.carryover_kwargs():
        #    #        continue
        #    #    pf[key] = self.pf[key]
        #
        #    if 'revision' not in self.pf:
        #        self.pf['revision'] = get_hg_rev()
        #
        #    # Save parameter file
        #    f = open('%s.parameters.pkl' % prefix, 'wb')
        #    pickle.dump(self.pf, f, -1)
        #    f.close()
        #
        #    if self.pf['verbose']:
        #        print 'Wrote %s.parameters.pkl' % prefix
        #