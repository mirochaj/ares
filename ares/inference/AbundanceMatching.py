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
from scipy.optimize import fsolve, curve_fit
from ..populations.GalaxyCohort import GalaxyCohort
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

class AbundanceMatching(GalaxyCohort):

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
            
            Mname = 'pop_lf_Mstar[{0:g}]'.format(z)
            pname = 'pop_lf_pstar[{0:g}]'.format(z)
            aname = 'pop_lf_alpha[{0:g}]'.format(z)
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
            return self.fstar_no_boost(z, M, [coeff, None]).flatten()

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
    
            # Mass function
            i_z = np.argmin(np.abs(z - self.halos.z))
            ngtm = self.halos.ngtm[i_z]
    
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
        fstar_of_z = [self.SFE(z, Macc_of_z) for z in self.halos.z]
        
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
    
    @property
    def _apply_floor(self):
        if not hasattr(self, '_apply_floor_'):
            self._apply_floor_ = 1
        return self._apply_floor_
    
    @_apply_floor.setter
    def _apply_floor(self, value):
        self._apply_floor_ = value        
                      
    @property
    def Ncoeff(self):
        if not hasattr(self, '_Ncoeff'): 
            self._Ncoeff = len(self.guesses)
        return self._Ncoeff

    @property
    def guesses(self):
        if not hasattr(self, '_guesses'):  
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

        return self._guesses

