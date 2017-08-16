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
        rmi, rma = self.dr.min(), self.dr.max()
        dlogr = self.pf['powspec_dlogr']

        kfin = 10**np.arange(np.log10(kmi), np.log10(kma), dlogk)
        rfin = 10**np.arange(np.log10(rmi), np.log10(rma), dlogr)

        # Re-organize to look like Global21cm outputs, i.e., dictionary
        # with one key per quantity of interest, and in this case a 2-D
        # array of shape (z, k)
        data_proc = {}
        for key in keys:
            
            if key in ['k', 'dr', 'k_cr', 'dr_cr']:
                continue

            down_sample = False
            twod = False
            if type(all_ps[0][key]) == np.ndarray:
                twod = True

                if ('cf' in key) or ('jp' in key) or ('ev' in key):
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
    
    @property
    def include_con_fl(self):
        if not hasattr(self, '_include_con_fl'):
            self._include_con_fl = self.pf['include_temp_fl'] or \
                self.pf['include_lya_fl']
        return self._include_con_fl

    def _temp_to_contrast(self, z, T):
        """
        Convert a temperature to a contrast
        """
        
        # Grab mean spin temperature
        Ts = np.interp(z, self.mean_history['z'][-1::-1], 
            self.mean_history['igm_Ts'][-1::-1])
        Tcmb = self.cosm.TCMB(z)
        
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

            zeta_lya += zeta * (Nlya / Nion) * self.pf['bubble_shell_Nsc']
            
            ##
            # First: some global quantities we'll need
            ##
            avg_Tk = np.interp(z, self.mean_history['z'][-1::-1],
                self.mean_history['igm_Tk'][-1::-1])    
            #xibar = np.interp(z, self.mean_history['z'][-1::-1],
            #    self.mean_history['cgm_h_2'][-1::-1])
            #Qi = xibar
            
            if self.pf['include_ion_fl']:
                Qi = self.field.BubbleFillingFactor(z, zeta)
            else:
                Qi = 0.
                
            xibar = 1. - np.exp(-Qi)
                
            xbar = 1. - xibar
            data['Qi'] = Qi
            data['xibar'] = xibar
            

            data['k'] = k
            data['dr'] = dr
            data['dr_cr'] = self.dr_coarse
            data['k_cr'] = self.k_coarse
            data['z'] = z
            
            data['zeta'] = zeta
            
            ##
            # Density fluctuations
            ##
            if self.pf['include_density_fl'] and self.pf['include_acorr']:
                # Halo model
                ps_posk = self.pops[0].halos.PowerSpectrum(z, self.k_pos)                
                
                # Must interpolate to uniformly (in real space) sampled
                # grid points to do inverse FFT
                ps_fold = np.concatenate((ps_posk[-1::-1], [0], ps_posk))
                #ps_fold = np.concatenate(([0], ps_posk, ps_posk[-1::-1]))
                #ps_dd = np.interp(self.k, self.k_coarse, ps_fold)
                ps_dd = np.interp(np.abs(self.k), self.k_pos, ps_posk)
                #ps_dd = self.field.halos.PowerSpectrum(z, np.abs(self.k))
                data['ps_dd'] = ps_dd
                data['cf_dd'] = np.fft.ifft(data['ps_dd'])
                
                # Interpolate onto coarser grid
                data['xi_dd_c'] = np.interp(self.dr_coarse, dr, 
                    data['cf_dd'].real)
                
            else:
                data['cf_dd'] = data['ps_dd'] = np.zeros_like(dr)
                
            ##    
            # Ionization fluctuations
            ##
            if self.pf['include_ion_fl'] and self.pf['include_acorr']:
                R_b, M_b, bsd = self.field.BubbleSizeDistribution(z, zeta)
            
                p_ii = self.field.JointProbability(z, self.dr_coarse, 
                    zeta, term='ii', Tprof=None, data=data)
                
                # Interpolate onto fine grid
                data['jp_ii'] = np.interp(dr, self.dr_coarse, p_ii)

                data['ev_ii'] = data['jp_ii']

                # Dimensions here are set by mass-sampling in HMF table
                data.update({'R_b': R_b, 'M_b': M_b, 'bsd':bsd})
            else:
                p_ii = data['jp_ii'] = data['ev_ii'] = np.zeros_like(dr)
                        
            ##
            # Temperature fluctuations                
            ##
            data['ev_coco'] = np.zeros_like(dr)
            if self.pf['include_temp_fl'] and self.pf['include_acorr']:
                
                Qh = self.field.BubbleShellFillingFactor(z, zeta)
                Ch = self._temp_to_contrast(z, self.pf['bubble_shell_temp'])
                data['Ch'] = Ch
                data['Qh'] = Qh
                data['avg_Ch'] = avg_Ch = Qh * Ch
                
                p_hh = self.field.JointProbability(z, self.dr_coarse, 
                    zeta, term='hh', Tprof=None, data=data)
                                    
                data['jp_hh'] = np.interp(dr, self.dr_coarse, p_hh)
              
                data['ev_coco'] += Ch**2 * data['jp_hh']
              
                data['avg_C'] = data['avg_Ch']
              
            else:
                p_hh = data['jp_hh'] = np.zeros_like(dr)
                data['Ch'] = data['Qh'] = Ch = Qh = data['avg_Ch'] \
                    = data['avg_C'] = 0.0
                
            ##
            # Lyman-alpha fluctuations                
            ##    
            if self.pf['include_lya_fl'] and self.pf['include_acorr']:
                Qc = self.field.BubbleShellFillingFactor(z, zeta_lya)
                Cc = self._temp_to_contrast(z, avg_Tk)
                data['Cc'] = Cc
                data['Qc'] = Qc
                data['avg_Cc'] = avg_Cc = Qc * Cc
                p_cc = self.field.JointProbability(z, self.dr_coarse, 
                    zeta, term='cc', Tprof=None, data=data, zeta_lya=zeta_lya)

                data['jp_cc'] = np.interp(dr, self.dr_coarse, p_cc)
                data['ev_coco'] += Cc**2 * data['jp_cc']
                data['avg_C'] += data['avg_Cc']

                if self.pf['include_temp_fl']:
                    p_hc = self.field.JointProbability(z, self.dr_coarse, 
                        zeta, term='hc', Tprof=None, data=data, zeta_lya=zeta_lya)
                    
                    data['jp_hc'] = np.interp(dr, self.dr_coarse, p_hc)
                    data['ev_coco'] += Cc * Ch * data['jp_hc'] * (Qh + Qc)
                else:
                    data['jp_hc'] = np.zeros_like(dr)
            else:
                data['jp_cc'] = data['jp_hc'] = data['ev_cc'] = \
                    np.zeros_like(dr)
                data['Cc'] = data['Qc'] = Cc = Qc = np.zeros_like(dr)
                    

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
                        p_id = self.field.JointProbability(z, self.dr_coarse, 
                            zeta, term='id', data=data)
                        data['jp_id'] = np.interp(dr, self.dr_coarse, p_id)
                        data['ev_id'] = data['jp_id']
                    else:
                        data['jp_id'] = data['ev_id'] = np.zeros_like(k)
                else:
                    data['jp_id'] = data['ev_id'] = np.zeros_like(k)        

                ##
                # Cross-terms between ionization and contrast
                ##
                if self.pf['include_ion_fl'] and self.include_con_fl:
                    if self.pf['include_temp_fl']:
                        p_ih = self.field.JointProbability(z, self.dr_coarse, 
                            zeta, term='ih', data=data)
                        data['jp_ih'] = np.interp(dr, self.dr_coarse, p_ih)
                    else:
                        data['jp_ih'] = np.zeros_like(dr)
                        
                    if self.pf['include_lya_fl']:
                        p_ic = self.field.JointProbability(z, self.dr_coarse, 
                            zeta, term='ic', data=data, zeta_lya=zeta_lya)
                        data['jp_ic'] = np.interp(dr, self.dr_coarse, p_ic)
                    else:
                        data['jp_ic'] = np.zeros_like(dr)
                            
                    data['ev_ico'] = data['Ch'] * data['jp_ih'] \
                                   + data['Cc'] * data['jp_ic']
                else:
                    data['jp_ih'] = np.zeros_like(k)
                    data['jp_ic'] = np.zeros_like(k)
                    data['ev_ico'] = np.zeros_like(k)
                    
            else:
                p_id = data['jp_id'] = data['ev_id'] = np.zeros_like(k)
                p_ih = data['jp_ih'] = data['ev_ih'] = np.zeros_like(k)
                p_ih = data['jp_ic'] = data['ev_ic'] = np.zeros_like(k)
                data['ev_ico'] = np.zeros_like(k)
                
            ##
            # Construct correlation functions from expectation values
            ##

            # Correlation function of ionized fraction and neutral fraction
            # are equivalent.
            data['cf_xx'] = data['ev_ii'] - xibar**2
            
            # Minus sign difference for cross term with density.
            data['cf_xd'] = -data['ev_id']
            
            # Contrast terms
            data['jp_cc'] = np.zeros_like(dr)
            data['cf_coco'] = np.zeros_like(dr)
            
            # Construct correlation function (just subtract off exp. value sq.)
            data['cf_coco'] = data['ev_coco'] - data['avg_C']**2
            
            # Correlation between neutral fraction and contrast fields
            data['cf_xco'] = data['avg_C'] - data['ev_ico']

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
                xi_xd**2 + 2. * xi_xd * xbar

            ##
            # MODIFY CORRELATION FUNCTION depending on Ts fluctuations
            ##    
            
            if (self.pf['include_temp_fl'] or self.pf['include_lya_fl']):
                # First term: just <psi psi'> in the saturated limit
                psi_s_psi_sp = data['cf_21_s'] + xbar**2

                ##
                # Let's start with low-order terms and build up from there.
                ##
                
                # The two easiest terms in the unsaturated limit are those
                # not involving the density, <x x' C'> and <x x' C C'>.
                # Under the binary field(s) approach, we can just write
                # each of these terms down
                ev_xi_cop = Ch * data['jp_ih'] + Cc * data['jp_ic']
                xxc = data['avg_C'] - ev_xi_cop
                xxcc = data['ev_coco']
                
                data['cf_21'] = psi_s_psi_sp + 2. * xxc + xxcc

                #P_id = self.field.JointProbability(z, np.zeros(1),
                #    zeta, term='id', data=data)[0]
                #print z, P_id
                data['cf_21'] -= (xbar + data['avg_C'])**2

            else:
                data['cf_21'] = data['cf_21_s']

            data['dTb0'] = Tbar
            #data['cf_21'] -= QHII**2
            #data['cf_21'] *= self.hydr.T0(z)**2
                         
            #data['ps_21_sat'] = self._regrid_and_fft(dr, data['cf_21_sat'], self.k)                                                  
            #data['ps_21'] = self._regrid_and_fft(dr, data['cf_21'], self.k)
            #
            data['ps_21'] = np.fft.fft(data['cf_21'])
            
            
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