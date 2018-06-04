import os
import pickle
import numpy as np
from .Global21cm import Global21cm
from ..solvers import FluctuatingBackground
from ..util import ParameterFile, ProgressBar
#from ..analysis.BlobFactory import BlobFactory
from ..physics.Constants import cm_per_mpc, c, s_per_yr
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

        # Aren't these just self.k_cr, self.R_cr?
        kfin = np.exp(np.arange(np.log(kmi), np.log(kma)+dlogk, dlogk))
        rfin = np.exp(np.arange(np.log(rmi), np.log(rma)+dlogr, dlogr))

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

    @property
    def include_fluctuations(self):
        if not hasattr(self, '_include_fl'):
            self._include_fl = self.pf['include_ion_fl'] or \
                self.pf['include_temp_fl'] or \
                self.pf['include_density_fl'] or \
                self.pf['include_lya_fl']
            
        return self._include_fl

    def step(self):
        """
        Generator for the power spectrum.
        """

        step = np.diff(self.pf['fft_scales'])[0]

        self.R = R = self.pf['fft_scales']
        self.k = k = np.fft.fftfreq(R.size, step)
        logR = np.log(R)

        k_mi, k_ma = k[k>0].min(), k.max()
        dlogk = self.pf['powspec_dlogk']
        k_pos = np.exp(np.arange(np.log(k_mi), np.log(k_ma)+dlogk, dlogk))

        if self.include_fluctuations:
            self.k_pos = self.pops[0].halos.k_cr_pos
            self.R_cr = self.pops[0].halos.R_cr
            self.logR_cr = np.log(self.R_cr)
        else:
            self.k_pos = self.R_cr = self.logR_cr = np.zeros(10)

        for i, z in enumerate(self.z):

            data = {}
                
            ## 
            # First, loop over populations and determine total
            # UV and X-ray outputs. 
            ##          
            
            # Prepare for the general case of Mh-dependent things
            Nion = np.zeros_like(self.field.halos.M)
            Nlya = np.zeros_like(self.field.halos.M)
            fXcX = np.zeros_like(self.field.halos.M)
            zeta_ion = zeta = np.zeros_like(self.field.halos.M)
            zeta_lya = np.zeros_like(self.field.halos.M)
            zeta_X = np.zeros_like(self.field.halos.M)
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
            Tcmb = self.cosm.TCMB(z)
            Tk = np.interp(z, self.mean_history['z'][-1::-1],
                self.mean_history['igm_Tk'][-1::-1])
            Ts = np.interp(z, self.mean_history['z'][-1::-1],
                self.mean_history['Ts'][-1::-1])
            Ja = np.interp(z, self.mean_history['z'][-1::-1],
                self.mean_history['Ja'][-1::-1])
            xHII, ne = [0] * 2
            
            xa = self.hydr.RadiativeCouplingCoefficient(z, Ja, Tk)
            xc = self.hydr.CollisionalCouplingCoefficient(z, Tk)
            xt = xa + xc
            
            # Assumes strong coupling. Mapping between temperature 
            # fluctuations and contrast fluctuations.
            #Ts = Tk
            
            
            # Add beta factors to dictionary
            for f1 in ['x', 'd', 'a']:
                func = self.hydr.__getattribute__('beta_%s' % f1)
                data['beta_%s' % f1] = func(z, Tk, xHII, ne, Ja)
            
            QHII_gs = np.interp(z, self.gs.history['z'][-1::-1], 
                self.gs.history['cgm_h_2'][-1::-1])
            
            QHII_ff = self.field.BubbleFillingFactor(z, zeta,
                rescale=self.pf['powspec_volfix'])
            
            # Mean brightness temperature outside bubbles  
            # Currently not including xe effects  
            Tbar = np.interp(z, self.gs.history['z'][-1::-1], 
                self.gs.history['dTb'][-1::-1]) / (1. - QHII_gs)


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
            
            if self.pf['include_ion_fl']:
                if self.pf['powspec_force_Qi_gs']:
                    Qi = QHII_gs
                else:    
                    Qi = self.field.BubbleFillingFactor(z, zeta, 
                        rescale=self.pf['powspec_rescale_Qion'])
            else:
                Qi = QHII_gs
                
            #Qi = np.mean([QHII_gs, self.field.BubbleFillingFactor(z, zeta)])    
                                                                
            #xibar = np.interp(z, self.mean_history['z'][-1::-1],
            #    self.mean_history['cgm_h_2'][-1::-1])
                
            xibar = Qi    
                
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
                
                # k may be different for 21-cm and matter power spectra!
                
                ps_dd = self.pops[0].halos.PowerSpectrum(z, self.k_pos)
                
                # PS is positive so it's OK to log-ify it
                data['ps_dd'] = np.exp(np.interp(np.log(np.abs(self.k)),
                    np.log(self.k_pos), np.log(ps_dd)))
                
                # If this isn't tabulated, need to send in full dr array
                cf_c = self.pops[0].halos.CorrelationFunction(z, self.R_cr)
                
                # Need finer-grain resolution for this. Leave correlation
                # function in linear units for interpolation because it will 
                # be negative on large scales. Also, need a factor of two
                # to account for the dispersal of power into negative 
                # frequencies. We should do this inside PS-tabulation in the
                # future.
                data['cf_dd'] = np.interp(logr, np.log(self.R_cr), cf_c)
                
                #data['cf_dd'][0:10] = 0
                #data['cf_dd'][-10:] = 0
                
            else:
                data['cf_dd'] = data['ps_dd'] = np.zeros_like(R)

            ##    
            # Ionization fluctuations
            ##
            if self.pf['include_ion_fl'] and self.pf['include_acorr']:
                R_b, M_b, bsd = self.field.BubbleSizeDistribution(z, zeta)
            
                p_ii, p_ii_1, p_ii_2 = self.field.JointProbability(z, 
                    self.R_cr, zeta, term='ii', Tprof=None, data=data,
                    zeta_lya=zeta_lya)
                
                # Interpolate onto fine grid
                data['jp_ii'] = np.interp(logR, self.logR_cr, p_ii)
                data['jp_ii_1'] = np.interp(logR, self.logR_cr, p_ii_1)
                data['jp_ii_2'] = np.interp(logR, self.logR_cr, p_ii_2)

                data['ev_ii'] = data['jp_ii']
                data['ev_ii_1'] = data['jp_ii_1']
                data['ev_ii_2'] = data['jp_ii_2']

                # Dimensions here are set by mass-sampling in HMF table
                data.update({'R_b': R_b, 'M_b': M_b, 'bsd':bsd})
                data['delta_B'] = self.field._B(z, zeta, zeta)
            else:
                p_ii = data['jp_ii'] = data['ev_ii'] = np.zeros_like(R)
                data['ev_ii_1'] = data['ev_ii_2'] = np.zeros_like(R)

            ##
            # Temperature fluctuations                
            ##
            
            
            
            data['ev_coco'] = np.zeros_like(R)
            data['ev_coco_1'] = np.zeros_like(R)
            data['ev_coco_2'] = np.zeros_like(R)
            if self.pf['include_temp_fl']:
                
                zeta_X = 40.
                
                if self.pf['powspec_temp_method'] == 'shell':
                    assert self.pf['include_ion_fl'], \
                        "Can only do temp_method='shell' if include_ion_fl=1!"
                
                    Q = self.field.BubbleShellFillingFactor(z, zeta)
                    
                elif self.pf['powspec_temp_method'] == 'xset':
                    R_b, M_b, bsd = self.field.BubbleSizeDistribution(z, zeta_X)
                    data.update({'R_h': R_b, 'M_h': M_b, 'bsd_h':bsd})
                    data['delta_B_h'] = self.field._B(z, zeta_X, zeta_X)
                    
                    Q = [self.field.BubbleFillingFactor(z, zeta_X), 0.0, 0.0]
                else:
                    raise NotImplemented('help')
                
                data['avg_C'] = 0.0
                data['jp_hc'] = np.zeros_like(R)
                data['jp_hc_1'] = np.zeros_like(R)
                data['jp_hc_2'] = np.zeros_like(R)

                data['Qc'] = 0.0
                data['Cc'] = 0.0
                                
                tem = []
                delta_T = []
                suffixes = 'h', 'c'
                for ii in range(3):

                    if self.pf['powspec_temp_method'] == 'xset' and ii > 0:
                        continue       

                    ztemp = self.pf['bubble_shell_ktemp_zone_{}'.format(ii)]

                    if ztemp == 'mean':
                        ztemp = Tk
                    elif ztemp == 'cold':
                        ztemp = self.cosm.Tgas(z)
                    elif ztemp is None:
                        if self.pf['bubble_shell_tpert_zone_{}'.format(ii)] is None:
                            ztemp = None
                        else:
                            ztemp = Tk * (1. + self.pf['bubble_shell_tpert_zone_{}'.format(ii)])
                    
                    tem.append(ztemp)
                                                            
                    if ztemp is None:
                        continue
                                                     
                    if ii <= 1:
                        if ztemp is None:
                            continue
                            
                        s = suffixes[ii]
                        ss = suffixes[ii] + suffixes[ii]
                        data['Q{}'.format(s)] = Q[ii]
                        
                        delta_T.append(ztemp / Tk - 1.)
                        
                    else:
                        s = 'hc'
                        ss = 'hc'

                        if tem[1] is None:
                            continue

                    # Compute the joint probability.
                    # Either hh, hc, or cc
                    if self.pf['powspec_temp_method'] == 'xset':
                        zeta_ = zeta_X
                    else:
                        zeta_ = zeta

                    p_tot, p_1h, p_2h = self.field.JointProbability(z, 
                        self.R_cr, zeta_, term=ss, Tprof=None, data=data,
                        zeta_lya=zeta_lya)
                    
                    data['jp_{}'.format(ss)] = \
                        np.interp(logR, self.logR_cr, p_tot)
                    data['jp_{}_1'.format(ss)] = \
                        np.interp(logR, self.logR_cr, p_1h)
                    data['jp_{}_2'.format(ss)] = \
                        np.interp(logR, self.logR_cr, p_2h)
                                        
                    if self.pf['powspec_temp_method'] == 'lpt' or ii == 1:
                        C = (Tcmb / (Ts - Tcmb)) * delta_T[ii]
                        
                        data['ev_coco'] += C**2 * data['jp_{}'.format(ss)]
                        #data['ev_coco_1'] += C**2 * data['jp_{}_1'.format(ss)]
                        #data['ev_coco_2'] += C**2 * data['jp_{}_2'.format(ss)]
                    else:
                        
                        # Tk is really Ts, but we're assuming for now
                        # that coupling is strong. Might overestimate 
                        # fluctuations slightly at early times.
                        if ii <= 1:
                            C = (Tcmb / (Tk - Tcmb)) * delta_T[ii] \
                                / (1. + delta_T[ii])
                            Csq = C**2
                        else:
                            # Remember, this is for the hot/cold term
                            Csq = (Tcmb / (Tk - Tcmb)) * delta_T[0] * delta_T[1] \
                                / (1. + delta_T[0]) / (1. + delta_T[1])
                            C = np.sqrt(C)
    
                        data['ev_coco'] += Csq * data['jp_{}'.format(ss)]
                        data['ev_coco_1'] += Csq * data['jp_{}_1'.format(ss)]
                        data['ev_coco_2'] += Csq * data['jp_{}_2'.format(ss)]
                                                                                
                    data['C{}'.format(s)] = C
                    #data['Q{}'.format(s)] = Q[ii]
                    if ii <= 1:
                        data['avg_C{}'.format(s)] = Q[ii] * C
                    data['avg_C'] += Q[ii] * C
                
                    #if not self.pf['bubble_shell_include_xcorr']:
                    #    continue
                
                    #data['jp_hc']
            
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


                Jc = self.hydr.Ja_c(z)
                                
                #data['Ca'] = C
                data['Qa'] = 0.
                
                #data['avg_C'] += C * Qa

                Mmin = lambda zz: self.pops[0].Mmin(zz)

                # Horizon set by distance photon can travel between n=3 and n=2
                zmax = self.hydr.zmax(z, 3)
                rmax = self.cosm.ComovingRadialDistance(z, zmax) / cm_per_mpc
                
                if self.pf['include_lya_lc']:
                    
                    # Use specific mass accretion rate of Mmin halo
                    # to get characteristic halo growth time. This is basically
                    # independent of mass so it should be OK to just pick Mmin.
                    
                    if type(self.pf['include_lya_lc']) is float:
                        a = lambda zz: self.pf['include_lya_lc']
                    else:
                    
                        #oot = lambda zz: self.pops[0].dfcolldt(z) / self.pops[0].halos.fcoll_2d(zz, np.log10(Mmin(zz)))
                        #a = lambda zz: (1. / oot(zz)) / pop.cosm.HubbleTime(zz)                        
                        oot = lambda zz: self.pops[0].halos.MAR_func(zz, Mmin(zz)) / Mmin(zz) / s_per_yr
                        a = lambda zz: (1. / oot(zz)) / pop.cosm.HubbleTime(zz)
                    
                    tstar = lambda zz: a(zz) * self.cosm.HubbleTime(zz)
                    rstar = c * tstar(z) * (1. + z) / cm_per_mpc
                    uisl = lambda kk, mm, zz: self.pops[0].halos.u_isl_exp(kk, mm, zz, rmax, rstar)
                else:
                    uisl = lambda kk, mm, zz: self.pops[0].halos.u_isl(kk, mm, zz, rmax)
                
                #uisl = self.field.halos.FluxProfileFT
                
                unfw = lambda kk, mm, zz: self.pops[0].halos.u_nfw(kk, mm, zz) 

                #ps_aa = self.pops[0].halos.PowerSpectrum(z, self.k_pos, uisl)
                #ps_aa = np.array([self.pops[0].halos.PowerSpectrum(z, kpos, uisl, Mmin(z), unfw, Mmin(z)) \
                #    for kpos in self.k_pos])
                #ps_ad_1 = np.array([self.pops[0].halos.PS_OneHalo(z, kpos, uisl, Mmin, unfw, Mmin) \
                #    for kpos in self.k_pos])
                #ps_ad_2 = np.array([self.pops[0].halos.PS_TwoHalo(z, kpos, uisl, Mmin, unfw, Mmin) \
                #    for kpos in self.k_pos])
                 
                #ps_aa = ps_ad   
                ps_aa = np.array([self.pops[0].halos.PowerSpectrum(z, kpos, uisl, Mmin(z)) \
                    for kpos in self.k_pos])
                #ps_aa_1 = np.array([self.pops[0].halos.PS_OneHalo(z, kpos, uisl, Mmin) \
                #    for kpos in self.k_pos])
                #ps_aa_2 = np.array([self.pops[0].halos.PS_TwoHalo(z, kpos, uisl, Mmin) \
                #    for kpos in self.k_pos])    

                # Interpolate back to fine grid before FFTing
                data['ps_aa'] = np.exp(np.interp(np.log(np.abs(self.k)), 
                    np.log(self.k_pos), np.log(ps_aa)))
                #data['ps_aa_1'] = np.exp(np.interp(np.log(np.abs(self.k)), 
                #    np.log(self.k_pos), np.log(ps_aa_1)))
                #data['ps_aa_2'] = np.exp(np.interp(np.log(np.abs(self.k)), 
                #    np.log(self.k_pos), np.log(ps_aa_2)))

                data['cf_aa'] = np.fft.ifft(data['ps_aa']).real

                data['ev_aa'] = data['cf_aa']

                if self.pf['powspec_lya_method'] == 'lpt':
                    #C = (Tcmb * (Tk - Ts)) / (Tk * (Ts - Tcmb)) #+ 1.
                    C = (Ts / (Ts - Tcmb)) * ((Tk - Tcmb) / Tk)
                    data['ev_coco'] += (C - 1)**2 * xa**2 * (1. + data['ev_aa'])
                    data['ev_coco'] -= (C - 1) * xa
                    #data['ev_coco'] += 1.
            
                    data['Ca'] = C

                    # Really average contrast perturbation
                    data['avg_C'] = 0.0 # we don't know this

                else:

                    #data['ev_coco'] += xa**2 * (Ts / (Ts - Tcmb))**2 \
                    #    * (1. - Tcmb / Tk)**2 * (1. + data['ev_aa']) 

                    data['ev_coco'] += (1. / (1. + xa))**2 *  data['ev_aa']

                    #data['ev_coco'] += data['beta_a']**2 * data['ev_aa'] + 1.
                    
                    #if xa < 1:
                    #    data['ev_coco'] += xa**2 * (Ts / (Ts - Tcmb))**2 \
                    #        * (1. - Tcmb / Tk)**2 * (1. + data['ev_aa'])
                    #else:    
                    #    data['ev_coco'] += (Ts / (Ts - Tcmb))**2 \
                    #        * (1. - Tcmb / Tk)**2 #* (1. + data['ev_aa'])    
                            
                    #data['ev_coco'] -= (C - 1) * xa
                    
                    
                    #raise NotImplemented('help')

                    #data['ev_coco'] += data['jp_aa'] * C**2 + 1.
                                
                #data['jp_{}'.format(ss)] = \
                #    np.interp(logR, self.logR_cr, p_tot)
                
            #else:
            #    data['jp_cc'] = data['jp_hc'] = data['ev_cc'] = \
            #        np.zeros_like(R)
            #    data['Cc'] = data['Qc'] = Cc = Qc = 0.0
            #    data['avg_Cc'] = 0.0
                
            ##
            # Cross-terms between ionization and contrast.
            # Should be under xcorr
            ##
            if self.include_con_fl and self.pf['include_ion_fl']:
                if self.pf['include_temp_fl']:
                    p_ih, p_ih_1, p_ih_2 = self.field.JointProbability(z,
                        self.R_cr, zeta, term='ih', data=data, zeta_lya=zeta_lya)
                    data['jp_ih'] = np.interp(logR, self.logR_cr, p_ih)
                    data['jp_ih_1'] = np.interp(logR, self.logR_cr, p_ih_1)
                    data['jp_ih_2'] = np.interp(logR, self.logR_cr, p_ih_2)
                else:
                    data['jp_ih'] = np.zeros_like(R)
                    
                #if self.pf['include_lya_fl']:
                #    p_ic, p_ic_1, p_ic_2 = self.field.JointProbability(z, 
                #        self.R_cr, zeta, term='ic', data=data, 
                #        zeta_lya=zeta_lya)
                #    data['jp_ic'] = np.interp(logR, self.logR_cr, p_ic)
                #    data['jp_ic_1'] = np.interp(logR, self.logR_cr, p_ic_1)
                #    data['jp_ic_2'] = np.interp(logR, self.logR_cr, p_ic_2)
                #else:
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
                    do_xcorr_cd = True
                else:
                    do_xcorr_xd = (self.pf['include_xcorr_wrt'] is not None) and \
                       ('density' in self.pf['include_xcorr_wrt']) and \
                       ('ionization' in self.pf['include_xcorr_wrt'])
                
                    do_xcorr_cd = (self.pf['include_xcorr_wrt'] is not None) and \
                       ('density' in self.pf['include_xcorr_wrt']) and \
                       ('contrast' in self.pf['include_xcorr_wrt'])
                    
                
                if do_xcorr_xd:

                    # Cross-correlation terms...
                    # Density-ionization cross correlation
                    if (self.pf['include_density_fl'] and self.pf['include_ion_fl']):
                        p_id, p_id_1, p_id_2 = self.field.JointProbability(z, 
                            self.R_cr, zeta, term='id', data=data,
                            zeta_lya=zeta_lya)
                        data['jp_id'] = np.interp(logR, self.logR_cr, p_id)
                        #data['jp_id'] = data['jp_ii'] \
                        #    * self.field._B(z, zeta, zeta)
                        data['ev_id'] = data['jp_id']
                         
                        #p_ii, p_ii_1, p_ii_2 = self.field.JointProbability(z, 
                        #    self.R_cr, zeta, term='ii', data=data,
                        #    zeta_lya=zeta_lya)
                        #data['jp_in'] = np.interp(logR, self.logR_cr, p_in)
                        
                        
                        
                        
                        #p_in, p_in_1, p_in_2 = self.field.JointProbability(z, 
                        #    self.R_cr, zeta, term='in', data=data,
                        #    zeta_lya=zeta_lya)
                        #data['jp_in'] = np.interp(logR, self.logR_cr, p_in)
                        #
                        #delta_i = self.field._B0(z, zeta)
                        #
                        #b = self.field._B(z, zeta, zeta)
                        #delta_i = np.trapz()
                        #
                        #data['ev_id'] = data['jp_ii'] * delta_i
                        #    #data['jp_in'] * delta_i
                        
                        
                    else:
                        data['jp_id'] = data['ev_id'] = np.zeros_like(k)
                else:
                    data['jp_id'] = data['ev_id'] = np.zeros_like(k)        

                if do_xcorr_cd:
                    # Cross-correlation terms...
                    # Density-contrast cross correlation
                    if self.pf['include_density_fl'] and self.include_con_fl:
                        p_cd, p_cd_1, p_cd_2 = self.field.JointProbability(z, 
                            self.R_cr, zeta, term='hd', data=data,
                            zeta_lya=zeta_lya)
                        data['jp_dco'] = data['Ch'] \
                            * np.interp(logR, self.logR_cr, p_cd)
                        
                        #data['jp_dco'] = data['jp_hh'] * data['Ch'] \
                        #    * self.field._B(z, zeta, zeta)
                        data['ev_dco'] = data['jp_dco']
                    else:
                        data['jp_dco'] = data['ev_dco'] = np.zeros_like(k)
                else:
                    data['jp_dco'] = data['ev_dco'] = np.zeros_like(k)   

                   
                ##
                # Cross-terms between density and contrast
                ##  
                #if self.include_con_fl and self.pf['include_density_fl']:
                #    #if self.pf['include_temp_fl']:
                #    #    p_dh = self.field.JointProbability(z, self.R_cr, 
                #    #        zeta, term='dh', data=data)
                #    #    data['jp_dh'] = np.interp(R, self.R_cr, p_dh)
                #    #else:
                #    #    data['jp_dh'] = np.zeros_like(R)
                #    #
                #    #if self.pf['include_lya_fl']:
                #    #    p_dc = self.field.JointProbability(z, self.R_cr, 
                #    #        zeta, term='dc', data=data, zeta_lya=zeta_lya)
                #    #    data['jp_dc'] = np.interp(R, self.R_cr, p_dc)
                #    #else:
                #    #    data['jp_dc'] = np.zeros_like(R)
                #    #
                #    #data['ev_dco'] = data['Ch'] * data['jp_ih'] \
                #    #               + data['Cc'] * data['jp_ic']
                #    
                #    data['ev_dco'] = data['ev_ico'] * data['ev_id'] # times delta
                #    
                #else:
                #    data['jp_dh'] = np.zeros_like(k)
                #    data['jp_dc'] = np.zeros_like(k)
                #    data['ev_dco'] = np.zeros_like(k)                        
                    
            else:
                # Density cross-terms
                p_id = data['jp_id'] = data['ev_id'] = np.zeros_like(k)
                p_dh = data['jp_dh'] = data['ev_dh'] = np.zeros_like(k)
                p_dc = data['jp_dc'] = data['ev_dc'] = np.zeros_like(k)
                data['jp_dco'] = data['ev_dco'] = np.zeros_like(k)
                
                #p_ih = data['jp_ih'] = data['ev_ih'] = np.zeros_like(k)
                #p_hc = data['jp_hc'] = data['ev_hc'] = np.zeros_like(k)
                #p_ic = data['jp_ic'] = data['ev_ic'] = np.zeros_like(k)
                #
                #data['ev_ico'] = np.zeros_like(k)
                #

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

            # Short-hand
            xi_xx = data['cf_xx']
            xi_dd = data['cf_dd'] * (1. + xc / (xt * (1. + xt)))**2
            xi_xd = data['cf_xd']
            xi_CC = data['cf_coco']

            # This is Eq. 11 in FZH04
            data['cf_21_s'] = xi_xx * (1. + xi_dd) + xbar**2 * xi_dd + \
                xi_xd * (xi_xd + 2. * xbar)
                
            # The temperature fluctuations just aren't beating the density...

            ##
            # MODIFY CORRELATION FUNCTION depending on Ts fluctuations
            ##    
            
            if (self.pf['include_temp_fl'] or self.pf['include_lya_fl']):

                ##
                # Let's start with low-order terms and build up from there.
                ##
                
                avg_xC = 0.0#data['avg_C'] # ??

                # The two easiest terms in the unsaturated limit are those
                # not involving the density, <x x' C'> and <x x' C C'>.
                # Under the binary field(s) approach, we can just write
                # each of these terms down
                ev_xi_cop = data['Ch'] * data['jp_ih'] + data['Ch'] * data['jp_ic']
                
                ev_cd = data['ev_dco']
                ev_cc = data['ev_coco']
                ev_xx = 1. - 2. * xibar + data['ev_ii']
                ev_xc = data['avg_C'] - ev_xi_cop
                #ev_xxc = xbar * data['avg_C'] - ev_xi_cop
                #ev_xxcc = ev_xx * ev_cc + avg_xC**2 + ev_xc**2
                ev_xxc = 0.0
                ev_xxcc = ev_cc * (1. + xi_dd)
                ev_xxcd = ev_cd
                
                # <x x'>
                data['ev_xx'] = ev_xx
                
                # <x C'>
                data['ev_xc'] = ev_xc
                
                # <x x' C>
                data['ev_xxc'] = ev_xxc

                # <x x' C C'>
                data['ev_xxcc'] = ev_xxcc
                
                # <x x' C d'>
                data['ev_xxcd'] = ev_xxcd
                
                # Eq. 33 in write-up
                phi_u = 2. * ev_xxc + ev_xxcc + 2. * ev_xxcd
                                
                # Need to make sure this doesn't get saved at native resolution!
                #data['phi_u'] = phi_u
                    
                data['cf_21'] = data['cf_21_s'] + phi_u #\
                    #- 2. * xbar * avg_xC
                    
                Cbar = 1.#data['avg_C']
                data['cf_21'] = xi_CC * (1. + xi_dd) + Cbar**2 * xi_dd #+ \
                    #xi_Cd* (xi_Cd + 2. * Cbar)    
                    
                    
                    
            else:
                data['cf_21'] = data['cf_21_s']

            #data['cf_21'][0:10] = 0
            #data['cf_21'][-10:] = 0
            
            
            data['dTb0'] = Tbar
            data['ps_21'] = np.fft.fft(data['cf_21'])
            data['ps_21_s'] = np.fft.fft(data['cf_21_s'])
            
            # Save 21-cm PS as one and two-halo terms also

            # These correlation functions are in order of ascending 
            # (real-space) scale.
            data['ps_xx'] = np.fft.fft(data['cf_xx'])
            data['ps_xx_1'] = np.fft.fft(data['cf_xx_1'])
            data['ps_xx_2'] = np.fft.fft(data['cf_xx_2'])
            data['ps_coco'] = np.fft.fft(data['cf_coco'])
            data['ps_coco_1'] = np.fft.fft(data['cf_coco_1'])
            data['ps_coco_2'] = np.fft.fft(data['cf_coco_2'])
            
            # These are all going to get downsampled before the end.
            
            # Might need to downsample in real-time to limit memory
            # consumption.
            

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