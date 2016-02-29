"""

AbundanceMatching.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sun Nov 22 12:01:50 PST 2015

Description: 

"""

import numpy as np
from ..util import read_lit
from types import FunctionType
from ..populations import GalaxyPopulation
from scipy.optimize import fsolve, curve_fit
from scipy.integrate import quad, simps, cumtrapz, ode
from scipy.interpolate import interp1d, RectBivariateSpline
from ..util import ParameterFile, MagnitudeSystem, ProgressBar
from ..physics.Constants import s_per_yr, g_per_msun, cm_per_mpc

try:
    import mpmath
except ImportError:
    pass

z0 = 9. # arbitrary

try:
    from scipy.misc import derivative
except ImportError:
    pass

class AbundanceMatching(GalaxyPopulation):

    @property
    def mags(self):
        if not hasattr(self, '_mags'):
            self._mags = self.constraints['mags']

        return self._mags
        
    @mags.setter
    def mags(self, value):
        assert len(value) == len(self.redshifts), \
            "Need magnitudes for each redshift bin!"
        self._mags = value

    @property
    def redshifts(self):
        if not hasattr(self, '_redshifts'):
            raise AttributeError('Must set redshifts by hand or through constraints!')
    
        return self._redshifts
    
    @redshifts.setter
    def redshifts(self, value):
        if type(value) not in [list, np.ndarray, tuple]:
            self._redshifts = [value]
        else:
            self._redshifts = value
    
    @property
    def constraints(self):
        return self._constraints
    
    @constraints.setter
    def constraints(self, value):
        """
        Schechter parameters assumed for abundance match.
        """
    
        self._constraints = {}
        
        # Read constraints from litdata
        if type(value) == str:                
            self.constraints_source = value
            data = read_lit(value)
            fits = data.fits['lf']['pars']
        
        # Can optionally use a subset of redshift constraints provided 
        try:
            self._constraints['z'] = self.redshifts
        except AttributeError:
            self._constraints['z'] = data.redshifts
            self.redshifts = data.redshifts
        
        self._constraints['Mstar'] = []
        self._constraints['pstar'] = []
        self._constraints['alpha'] = []
        self._constraints['mags'] = []
        
        for i, z in enumerate(self.redshifts):
            # If we read in fits from literature, have to be sure to
            # get the redshifts correctly (since we're allowed to only
            # use a subset of them)
            
            # Also, must override litdata if appropriate pars are passed.
            
            Mname = 'pop_lf_Mstar[%g]' % z
            pname = 'pop_lf_pstar[%g]' % z
            aname = 'pop_lf_alpha[%g]' % z
            if Mname in self.pf:
                self._constraints['Mstar'].append(self.pf[Mname])
            elif type(value) == str:
                j = data.redshifts.index(z)
                self._constraints['Mstar'].append(fits['Mstar'][j])
            if pname in self.pf:   
                self._constraints['pstar'].append(self.pf[pname])
            elif type(value) == str:
                j = data.redshifts.index(z)
                self._constraints['pstar'].append(fits['pstar'][j])
            if aname in self.pf:
                self._constraints['alpha'].append(self.pf[aname])
            elif type(value) == str:
                j = data.redshifts.index(z)
                self._constraints['alpha'].append(fits['alpha'][j])
        
            self._constraints['mags'].append(np.array(data.data['lf'][z]['M']))
        
        # Parameter file will have LF in Magnitudes...argh
        redshifts = self._constraints['z']
        self._constraints['Lstar'] = []
        
        # Correct magnitudes for dust extinction, convert to luminosity
        for i, z in enumerate(redshifts):
            M = self._constraints['Mstar'][i]
            #Mdc = M - self.AUV(z, M)
            Mdc = M - self.AUV(z, M)
            L = self.magsys.MAB_to_L(mag=Mdc, z=z)
            self._constraints['Lstar'].append(L)
        
        return self._constraints 

    def fit_fstar(self):
        M = []; fstar = []; z = []
        for i, element in enumerate(self.MofL_tab):
            z.extend([self.redshifts[i]] * len(element))
            M.extend(element)
            fstar.extend(self.fstar_tab[i]) 

        x = [np.array(M), np.array(z)]
        y = np.array(fstar)

        guess = [0.2, 1e12, 0.5, 0.5]

        def to_fit(Mz, *coeff):
            M, z = Mz
            #for i in range(4):
            #    self.pf['sfe_Mfun_par%i' % i] = coeff[i]
                
            return self.fstar._call(z, M, coeff).flatten()

        coeff, cov = curve_fit(to_fit, x, y, p0=guess, maxfev=100000)

        return coeff       
    
    @property
    def fit_Lh(self):
        if not hasattr(self, '_fit_Lh'):
            self._fit_Lh = not self.fit_fstar
    
        return self._fit_Lh

    @property
    def fstar_tab(self):
        """
        These are the star-formation efficiencies derived from abundance
        matching.
        """

        if hasattr(self, '_fstar_tab'):
            return self._fstar_tab
    
        Nm = 0
        for i, z in enumerate(self.redshifts):
            Nm += len(self.mags[i])
    
        Nz = len(self.constraints['z'])
    
        self._fstar_tab = [[] for i in range(len(self.redshifts))]
        pb = ProgressBar(Nz * Nm, name='ham', use=self.pf['progress_bar'])
        pb.start()
    
        self._MofL_tab = [[] for i in range(len(self.redshifts))]
        self._LofM_tab = [[] for i in range(len(self.redshifts))]
    
        # Do it already    
        for i, z in enumerate(self.redshifts):
    
            mags = []
            for mag in self.mags[i]:
                mags.append(mag-self.AUV(z, mag))
    
            # Read in constraints for this redshift
            alpha = self.constraints['alpha'][i]
            L_star = self.constraints['Lstar'][i]    # dust corrected
            phi_star = self.constraints['pstar'][i]
    
            i_z = np.argmin(np.abs(z - self.halos.z))
            eta = self.eta[i_z]
            ngtm = self.halos.ngtm[i_z]
            log_ngtm = np.log(ngtm)
    
            # Use dust-corrected magnitudes here
            LUV_dc = [self.magsys.MAB_to_L(mag, z=z) for mag in mags]
    
            # No dust correction
            LUV_no_dc = [self.magsys.MAB_to_L(mag, z=z) \
                for mag in self.mags[i]]

            # Loop over luminosities and perform abundance match
            for j, Lmin in enumerate(LUV_dc):

                # Integral of schecter function at L > Lmin                
                xmin = Lmin / L_star
                int_phiL = mpmath.gammainc(alpha + 1., xmin)
                int_phiL *= phi_star
    
                # Number density of halos at masses > M
                ngtM_spl = interp1d(self.halos.lnM, np.log(ngtm), 
                    kind='linear', bounds_error=False)
                self.ngtM_spl = ngtM_spl
    
                def to_min(logMh):
                    int_nMh = np.exp(ngtM_spl(logMh))[0]
    
                    return abs(int_phiL - int_nMh)
    
                Mmin = np.exp(fsolve(to_min, 10., factor=0.01, 
                    maxfev=10000)[0])
    
                self._MofL_tab[i].append(Mmin)
                self._LofM_tab[i].append(LUV_dc[j])
                self._fstar_tab[i].append(Lmin / self.L1500_per_SFR(None, Mmin) \
                    / self.pSFR(z, Mmin))

                pb.update(i * Nm + j + 1)
    
        pb.finish()    
    
        return self._fstar_tab
    
    def L1600_limit(self, z):
        eta = np.interp(z, self.halos.z, self.eta)
        Mmin = np.interp(z, self.halos.z, self.Mmin)

        #sfr_M_z = RectBivariateSpline(self.halos.z, self.halos.lnM, 
        #    np.log(self.sfr_tab))

        #Lh_Mmin = np.exp(sfr_M_z(z, np.log(Mmin))[0][0]) / self.kappa_UV   

        return self.cosm.fbaryon * self.Macc(z, Mmin) \
            * eta * self.SFE(z, Mmin) / self.L1500_per_SFR(None, Mmin)
            
    def MAB_limit(self, z):
        """
        Magnitude corresponding to minimum halo mass in which stars form.
        """
        
        Lh_Mmin = self.L1600_limit(z)
        
        return self.magsys.L_to_MAB(Lh_Mmin, z=z)

    @property
    def LofM_tab(self):
        """
        Intrinsic luminosities corresponding to the supplied magnitudes.
        """
        if not hasattr(self, '_LofM_tab'):
            tab = self.fstar_tab

        return self._LofM_tab            

    @property
    def MofL_tab(self):
        """
        These are the halo masses determined via abundance matching that
        correspond to the M_UV's provided.
        """
        if not hasattr(self, '_MofL_tab'):
            tab = self.fstar_tab
    
        return self._MofL_tab

    def Mh_of_z(self, zarr):
        """
        Given a redshift, evolve a halo from its initial mass (Mmin(z)) onward.
        """
        
        # If in ascending order, flip
        if np.all(np.diff(zarr) > 0):
            zarr = zarr[-1::-1]
            
        zmin, zmax = zarr[-1], zarr[0]
        dz = np.diff(zarr)
        #dt = dz * self.cosm.dtdz(zarr[0:-1])
        
        # Initial mass of halo
        M0 = np.interp(zmax, self.halos.z, self.Mmin)
                
        # dM/dt = rhs        
        
        eta = interp1d(self.halos.z, self.eta, kind='cubic')
        
        # Minus sign because M increases as z decreases (not dtdz conversion)
        rhs = lambda z, M: -self.Macc(z, M) * eta(z) * self.cosm.dtdz(z) / s_per_yr
 
        solver = ode(rhs).set_integrator('vode', method='bdf')
        
        Mh = [M0]
                
        z = zmax
        solver.set_initial_value(M0, z)                
                
        i = 0
        while z > zarr.min():
            
            solver.integrate(z+dz[i])
            Mh_of_z = solver.y
            
            Mh.append(Mh_of_z[0])
            z += dz[i]
            i += 1
              
        return zarr, np.array(Mh)
        
    def Mstar(self, M):
        """
        Stellar mass as a function of halo mass and time.
        """
    
        
        dtdz = self.cosm.dtdz(self.halos.z)
        
        zarr, Mh_of_z = self.Mh_of_z(M)
                                
        
        # M is the initial mass of a halo
        # Macc_of_z is its MAR as a function of redshift
        Macc_of_z = self.Macc(self.halos.z, M)
        fstar_of_z = map(lambda z: self.SFE(z, Macc_of_z), self.halos.z)
        
        dtdz = self.cosm.dtdz(self.halos.z)
        Mh_of_z = cumtrapz(Macc_of_z[-1::-1] * dtdz / s_per_yr,
            x=self.halos.z[-1::-1], initial=M)
            
        Mh_of_z_all.append(Mh_of_z[-1::-1])    
                
    #@property
    #def coeff(self):
    #    if not hasattr(self, '_coeff'):
    #        if self.fit_fstar:
    #            self._coeff = self.coeff_fstar
    #        else:
    #            self._coeff = self.coeff_mtl
    #            
    #    return self._coeff
                
    @property
    def coeff_fstar(self):
        if not hasattr(self, '_coeff_fstar'):
            M = []; fstar = []; z = []
            for i, element in enumerate(self.MofL_tab):
                z.extend([self.redshifts[i]] * len(element))
                M.extend(element)
                fstar.extend(self.fstar_tab[i]) 

            x = [np.array(M), np.array(z)]
            y = np.log10(fstar)

            guess = self.guesses

            def to_fit(Mz, *coeff):
                M, z = Mz
                return self._log_fstar(z, M, *coeff).flatten()
    
            try:
                self._coeff_fstar, self._cov = \
                    curve_fit(to_fit, x, y, p0=guess, maxfev=100000)
            except RuntimeError:
                self._coeff_fstar, self._cov = guess, np.diag(guess)
    
        return self._coeff_fstar
        
    @property
    def coeff_mtl(self):
        if not hasattr(self, '_coeff_mtl'):
            M = []; Lh = []; z = []
            for i, element in enumerate(self.MofL_tab):
                z.extend([self.redshifts[i]] * len(element))
                M.extend(element)
                Lh.extend(self.LofM_tab[i])

            x = [np.array(M), np.array(z)]
            y = np.log10(Lh)

            guess = self.guesses

            def to_fit(Mz, *coeff):
                M, z = Mz
                return self._log_Lh(z, M, *coeff).flatten()
    
            try:
                self._coeff_mtl, self._cov = \
                    curve_fit(to_fit, x, y, p0=guess, maxfev=100000)
            except RuntimeError:
                self._coeff_mtl, self._cov = guess, np.diag(guess)
    
        return self._coeff_mtl
    
    def _fstar_cap(self, fstar):
        if type(fstar) in [float, np.float64]:
            if fstar > self.pf['pop_fstar_ceil']:
                return self.pf['pop_fstar_ceil']
            else:
                return fstar
        elif type(fstar) == np.ndarray:
            ceil = self.pf['pop_fstar_ceil'] * np.ones_like(fstar)
            return np.minimum(fstar, ceil)
        else:
            raise TypeError('Unrecognized type: %s' % type(fstar))

    @property
    def fstar_zhi(self):
        if not hasattr(self, '_fstar_zhi'):
            self._fstar_zhi = lambda MM, *coeff: \
                10**self._log_fstar(self.zext[2], MM, *coeff)
                
        return self._fstar_zhi

    @property
    def fstar_zlo(self):
        if not hasattr(self, '_fstar_zhi'):
            self._fstar_zlo = lambda MM, *coeff: \
                10**self._log_fstar(self.zext[1], MM, *coeff)
    
        return self._fstar_zlo

    @property
    def fstar_Mhi(self):
        if not hasattr(self, '_fstar_zhi'):
            self._fstar_Mhi = lambda MM, *coeff: \
                10**self._log_fstar(self.zext[2], MM, *coeff)

        return self._fstar_zhi

    @property
    def fstar_Mlo(self):
        if not hasattr(self, '_fstar_Mlo'):
            Mmin_of_z = [min(self.MofL_tab[i]) \
                for i, red in enumerate(self.redshifts)]
        
            fMlos = []
            alphas = []
            for i, redshift in enumerate(self.redshifts):
        
                Mlo = Mmin_of_z[i]
        
                M1 = 10**(np.log10(Mlo) - 0.01)
                M2 = 10**(np.log10(Mlo) + 0.01)
        
                f1 = 10**self._log_fstar(redshift, M1, *self.coeff)
                f2 = 10**self._log_fstar(redshift, M2, *self.coeff)
        
                fMlo = 10**self._log_fstar(redshift, Mlo, *self.coeff)
        
                alpha = (np.log10(f2) - np.log10(f1)) \
                    / (np.log10(M2) - np.log10(M1))
        
                fMlos.append(fMlo)
                alphas.append(alpha)
                
            has_cutoff = type(self.Mext) in [list, tuple]
        
            def tmp(zz, MM, *coeff):
        
        
                # If this redshift is within the range of the data, interpolate
                # between previously determined slopes
                
                if (not has_cutoff) and \
                   self.redshifts[0] <= zz <= self.redshifts[-1]:
                    Mlo = 10**np.interp(zz, self.redshifts, 
                        np.log10(Mmin_of_z))
                    fMlo = 10**np.interp(zz, self.redshifts, 
                        np.log10(fMlos))                    
        
                else:
        
                    # Assume Mlo follows shape of function at fixed
                    # logM distance from peak
        
                    Mlo = 10**(self._Mpeak_of_z(zz) \
                        - self.Mext[1] * self._sigma_of_z(zz))
        
                    fMlo = 10**self._log_fstar(zz, Mlo, *coeff)                     
        
                if self.Mext[0] == 'pl':
                    M1 = 10**(np.log10(Mlo) - 0.01)
                    M2 = 10**(np.log10(Mlo) + 0.01)
                    
                    f1 = 10**self._log_fstar(zz, M1, *coeff)
                    f2 = 10**self._log_fstar(zz, M2, *coeff)
                    
                    alpha = (np.log10(f2) - np.log10(f1)) \
                        / (np.log10(M2) - np.log10(M1))
                elif self.Mext[0] == 'const':
                    alpha = 0.
                else:
                    pass
        
                if type(MM) is np.ndarray:
                    Mbelow = MM[np.argwhere(MM < Mlo)]
                    Mabove = MM[np.argwhere(MM >= Mlo)]
                    
                    if self.Mext[0] == 'exp':
                        fst_lo = fMlo * np.exp(-Mlo / Mbelow) / np.exp(-1.)
                    else:
                        fst_lo = fMlo * (Mbelow / Mlo)**alpha
                    
                    fst_hi = 10**self._log_fstar(zz, Mabove, *coeff)
        
                    return np.concatenate((fst_lo, fst_hi)).squeeze()
                elif MM > Mlo:
                    return 10**self._log_fstar(zz, MM, *coeff)
                else:
                    if self.Mext[0] == 'exp':
                        return fMlo * np.exp(-Mlo / MM) / np.exp(-1.)
                    else:    
                        return 10**self._log_fstar(zz, MM, *coeff) #\
                            #+ fMlo * (MM / Mlo)**alpha
        
            self._fstar_Mlo = tmp
    
        return self._fstar_Mlo    
    
    @property
    def _apply_floor(self):
        if not hasattr(self, '_apply_floor_'):
            self._apply_floor_ = 1
        return self._apply_floor_
    
    @_apply_floor.setter
    def _apply_floor(self, value):
        self._apply_floor_ = value        
            
    def _log_fstar(self, z, M, *coeff):    
        
        if self.Mfunc == self.zfunc == 'poly':            
            logf = coeff[0] + coeff[1] * np.log10(M / 1e10) \
                 + coeff[2] * ((1. + z) / 8.) \
                 + coeff[3] * ((1. + z) / 8.) * np.log10(M / 1e10) \
                 + coeff[4] * (np.log10(M / 1e10))**2.
                 
            f = 10**logf
            
        elif (self.Mfunc == 'poly') and (self.zfunc == 'const'):
            logf = coeff[0] + coeff[1] * np.log10(M / 1e10) \
                 + coeff[2] * (np.log10(M / 1e10))**2.
        
            f = 10**logf    
        elif (self.Mfunc == 'lognormal') and (self.zfunc == 'linear_z'):
            logM = np.log10(M)
            
            self._fstar_of_z = lambda zz: coeff[0] + coeff[1] * (1. + zz) / z0 
            self._Mpeak_of_z = lambda zz: coeff[2] + coeff[3] * (1. + zz) / z0  # log10
            self._sigma_of_z = lambda zz: coeff[4] + coeff[5] * (1. + zz) / z0 
            
            f = self._fstar_of_z(z) \
                * np.exp(-(logM - self._Mpeak_of_z(z))**2 / 2. / 
                self._sigma_of_z(z)**2)

        elif (self.Mfunc == 'lognormal') and (self.zfunc == 'pl'):
            logM = np.log10(M)
        
            self._fstar_of_z = lambda zz: coeff[0] \
                - coeff[1] * np.log10((1. + zz) / (1. + z0))
            # logM
            self._Mpeak_of_z = lambda zz: coeff[2] \
                - coeff[3] * np.log10((1. + zz) / (1. + z0))
            self._sigma_of_z = lambda zz: coeff[4] \
                    - coeff[5] * np.log10((1. + zz) / (1. + z0))
        
            f = self._fstar_of_z(z) \
                * np.exp(-(logM - self._Mpeak_of_z(z))**2 / 2. / 
                self._sigma_of_z(z)**2)
        
        elif (self.Mfunc == 'lognormal') and (self.zfunc == 'const'):
            logM = np.log10(M)

            f = coeff[0] * np.exp(-(logM - coeff[1])**2 / 2. / coeff[2]**2)

        elif (self.Mfunc == 'lognormal') and (self.zfunc == 'linear_t'):
            logM = np.log10(M)

            self._fstar_of_z = lambda zz: coeff[0] \
                -1.5 * np.log10((1. + zz) / (1. + z0))
            # logM
            self._Mpeak_of_z = lambda zz: coeff[1] \
                -1.5 * np.log10((1. + zz) / (1. + z0))
            self._sigma_of_z = lambda zz: coeff[2]

            f = 10**self._fstar_of_z(z) \
                * np.exp(-(logM - self._Mpeak_of_z(z))**2 / 2. / 
                self._sigma_of_z(z)**2)
                
        # Nothing stopping some of the above treatments from negative fstar 
        f = np.maximum(f, 0.0)
                
        ##
        # HANDLE LOW-MASS END
        ##        
        if (self.Mext == 'floor'):
            f += self.Mext[1]
        elif self.Mext == 'pl_floor' and self._apply_floor:
            self._apply_floor = 0
            to_add = self.Mext_pars[0] * 10**self._log_fstar(z, 1e10, *coeff)
            to_add *= (M / 1e10)**self.Mext_pars[1] * np.exp(-M / 1e11)
            f += to_add
            self._apply_floor = 1
            
        # Apply ceiling
        f = np.minimum(f, self.pf['pop_fstar_ceil'])
                
        logf = np.log10(f)
            
        return logf
        
    def Lh(self, z, M):
        return 10**self._log_Lh(z, M, *self.coeff) 
           
    def _log_Lh(self, z, M, *coeff): 
        if self.Mfunc == 'pl':
            return coeff[0] + coeff[1] * np.log10(M / 1e12)
        elif self.Mfunc == 'schechter':
            return coeff[0] + coeff[1] * np.log10(M / coeff[2]) - M / coeff[2]
        elif self.Mfunc == 'poly':
            return coeff[0] + coeff[1] * np.log10(M / 1e10) \
                + coeff[2] * (np.log10(M / 1e10))**2 
    
    @property
    def Npops(self):
        return self.pf.Npops
        
    #@property
    #def pop_id(self):
    #    # Pop ID number for HAM population
    #    if not hasattr(self, '_pop_id'):
    #        for i, pf in enumerate(self.pf.pfs):
    #            if pf['pop_model'] == 'ham':
    #                break
    #        
    #        self._pop_id = i
    #    
    #    return self._pop_id

    @property
    def Ncoeff(self):
        if not hasattr(self, '_Ncoeff'): 
            self._Ncoeff = len(self.guesses)
        return self._Ncoeff

    @property
    def guesses(self):
        if not hasattr(self, '_guesses'):  
            if self.fit_fstar:
                if self.Mfunc == self.zfunc == 'poly':            
                    self._guesses = -1. * np.ones(5) 
                elif (self.Mfunc == 'poly') and (self.zfunc == 'const'):
                    self._guesses = -1. * np.ones(3)
                elif (self.Mfunc == 'lognormal') and (self.zfunc == 'linear_z'):
                    self._guesses = np.array([0.25, 0.05, 12., 0.05, 0.5, 0.05])
                elif (self.Mfunc == 'lognormal') and (self.zfunc == 'const'):
                    self._guesses = np.array([0.25, 11., 0.5])
                elif (self.Mfunc == 'lognormal') and (self.zfunc == 'pl'):
                    self._guesses = np.array([0.25, 0.05, 11., 0.05, 0.5, 0.05])    
                elif (self.Mfunc == 'lognormal') and (self.zfunc == 'linear_t'):
                    self._guesses = np.array([0.25, 12., 0.5])
                else:
                    raise NotImplemented('help')
            elif self.fit_Lh:
                if self.Mfunc == 'pl':
                    self._guesses = np.array([26., 1.]) 
                elif self.Mfunc == 'poly':
                    self._guesses = np.array([28., 1., 1., 1.])
                elif self.Mfunc == 'schechter':
                    self._guesses = np.array([28., 1., 11.5])    
                else:
                    raise NotImplemented('help')
            else:
                raise NotImplemented('Unrecognized option!')

        return self._guesses

    #def Mpeak(self, z):
    #    """
    #    The mass at which the star formation efficiency peaks.
    #    """
    #    
    #    alpha = lambda MM: self.gamma_sfe(z, MM)
    #        
    #    i = np.argmin(np.abs(z - np.array(self.redshifts)))
    #    guess = self.MofL_tab[i][np.argmax(self.fstar_tab[i])]
    #
    #    return fsolve(alpha, x0=guess, maxfev=10000, full_output=False,
    #        xtol=1e-3)[0]
    #        
    #def fpeak(self, z):
    #    return self.SFE(z, self.Mpeak(z))
    #
    #def gamma_sfe(self, z, M):
    #    """
    #    This is a power-law index describing the relationship between the
    #    SFE and and halo mass.
    #    
    #    Parameters
    #    ----------
    #    z : int, float
    #        Redshift
    #    M : int, float
    #        Halo mass in [Msun]
    #        
    #    """
    #    
    #    fst = lambda MM: self.SFE(z, MM)
    #    
    #    return derivative(fst, M, dx=1e6) * M / fst(M)
    #        
    #def alpha_lf(self, z, mag):
    #    """
    #    Slope in the luminosity function
    #    """
    #    
    #    logphi = lambda MM: np.log10(self.LuminosityFunction(z, MM, mags=True))
    #    
    #    return -(derivative(logphi, mag, dx=0.1) + 1.)
        

            
            
            
            
            
