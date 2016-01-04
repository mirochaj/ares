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
from scipy.optimize import fsolve, curve_fit, fmin
from scipy.integrate import quad, simps, cumtrapz, ode
from scipy.interpolate import interp1d, RectBivariateSpline
from ..util import ParameterFile, MagnitudeSystem, ProgressBar
from ..physics.Constants import s_per_yr, g_per_msun, cm_per_mpc

try:
    import mpmath
except ImportError:
    pass

z0 = 8. # arbitrary

try:
    from scipy.misc import derivative
except ImportError:
    pass

class HAM(object):
    def __init__(self, galaxy=None, **kwargs):
        if galaxy is not None:
            self.galaxy = galaxy
            self.pf = galaxy.pf
            self.halos = galaxy.halos
            self.cosm = self.halos.cosm
            self.dfcolldt = galaxy.dfcolldt
        else:
            self.pf = ParameterFile(**kwargs)
        
    @property
    def magsys(self):
        if not hasattr(self, '_magsys'):
            self._magsys = MagnitudeSystem(**self.pf)
    
        return self._magsys
        
    @property
    def mags(self):
        if not hasattr(self, '_mags'):
            self._mags = []
    
            if type(self.pf['pop_constraints']) == str:
                data = read_lit(self.pf['pop_constraints']).data        
                                    
                for redshift in self.redshifts:
                    self._mags.append(np.array(data['lf'][redshift]['M']))
            else:
                assert len(self.pf['pop_lf_mags']) == len(self.redshifts), \
                    "Need magnitudes for each redshift bin!"
    
                self._mags = self.pf['pop_lf_mags']
    
        return self._mags    
        
    @property
    def redshifts(self):
        return self.constraints['z']
        
    @property   
    def SFRD(self):
        """
        Compute star-formation rate density (SFRD).
        
        """
        
        if not hasattr(self, '_SFRD'):
            self._SFRD = interp1d(self.halos.z, self.sfrd_tab,
                kind='cubic')
                
        return self._SFRD
    
    @property
    def constraints(self):
        """
        Schechter parameters assumed for abundance match.
        """
        if not hasattr(self, '_constraints'):
    
            self._constraints = {}
    
            # Read constraints from litdata
            if type(self.pf['pop_constraints']) == str:                
                data = read_lit(self.pf['pop_constraints'])
                fits = data.fits['lf']['pars']
    
            # Can optionally use a subset of redshift constraints provided 
            if self.pf['pop_lf_z'] is not None:
                self._constraints['z'] = self.pf['pop_lf_z']
            else:
                self._constraints['z'] = data.redshifts
    
            self._constraints['Mstar'] = []
            self._constraints['pstar'] = []
            self._constraints['alpha'] = []
    
            for i, z in enumerate(self._constraints['z']):
                # If we read in fits from literature, have to be sure to
                # get the redshifts correctly (since we're allowed to only
                # use a subset of them)
                
                # Also, must override litdata if appropriate pars are passed.
                
                Mname = 'pop_lf_Mstar[%g]' % z
                pname = 'pop_lf_pstar[%g]' % z
                aname = 'pop_lf_alpha[%g]' % z
                if Mname in self.pf:
                    self._constraints['Mstar'].append(self.pf[Mname])
                elif type(self.pf['pop_constraints']) == str:
                    j = data.redshifts.index(z)
                    self._constraints['Mstar'].append(fits['Mstar'][j])
                if pname in self.pf:   
                    self._constraints['pstar'].append(self.pf[pname])
                elif type(self.pf['pop_constraints']) == str:
                    j = data.redshifts.index(z)
                    self._constraints['pstar'].append(fits['pstar'][j])
                if aname in self.pf:
                    self._constraints['alpha'].append(self.pf[aname])
                elif type(self.pf['pop_constraints']) == str:
                    j = data.redshifts.index(z)
                    self._constraints['alpha'].append(fits['alpha'][j])

            # Parameter file will have LF in Magnitudes...argh
            redshifts = self._constraints['z']
            self._constraints['Lstar'] = []
    
            # Correct magnitudes for dust extinction, convert to luminosity
            for i, z in enumerate(redshifts):
                M = self._constraints['Mstar'][i]
                Mdc = M - self.A1600(z, M)
                L = self.magsys.MAB_to_L(mag=Mdc, z=z)
                self._constraints['Lstar'].append(L)
    
        return self._constraints 
    
    def A1600(self, z, mag):
        """
        Determine infrared excess using Meurer et al. 1999 approach.
        """
    
        if not self.pf['pop_lf_dustcorr']:
            return 0.0    
    
        if type(self.pf['pop_lf_dustcorr']) == str:
            pass    
    
        # Could be constant, but redshift dependent
        if 'pop_lf_beta[%g]' % z in self.pf:
            beta = self.pf['pop_lf_beta[%g]'] 
    
        # Could depend on redshift AND magnitude
        elif 'pop_lf_beta_slope[%g]' % z in self.pf:
            if self.pf['pop_lf_beta_slope[%g]' % z] is not None:
                beta = self.pf['pop_lf_beta_slope[%g]' % z] \
                    * (mag + 19.5) + self.pf['pop_lf_beta_pivot[%g]' % z]
            else:
                beta = self.pf['pop_lf_beta']        
        # Could just be constant
        else:
            beta = self.pf['pop_lf_beta']
    
        return 4.43 + 1.99 * beta
        
    @property
    def Macc(self):
        """
        Mass accretion rate onto halos of mass M at redshift z.
    
        ..note:: This is the *matter* accretion rate. To obtain the baryonic 
            accretion rate, multiply by Cosmology.fbaryon.
            
        """
        if not hasattr(self, '_Macc'):
            if self.pf['pop_Macc'] is None:
                self._Macc = None
            elif type(self.pf['pop_Macc']) is FunctionType:
                self._Macc = self.pf['pop_Macc']
            elif self.pf['pop_Macc'] == 'pl':
                raise NotImplemented('do this')
            else:
                self._Macc = read_lit(self.pf['pop_Macc']).Macc

        return self._Macc

    @property
    def zlim(self):
        if not hasattr(self, '_zlim'):
            self._zlim = [min(self.redshifts), max(self.redshifts)]    
        return self._zlim
        
    @property
    def Mlim(self):
        if not hasattr(self, '_Mlim'):
            self._Mlim = [[min(self.MofL_tab[i]), max(self.MofL_tab[i])] \
                for i in range(len(self.redshifts))]
        return self._Mlim
        
    @property
    def fit_fstar(self):
        if not hasattr(self, '_fit_fstar'):
            if self.pf['pop_ham_fit'] == 'fstar':
                self._fit_fstar = True
            else:
                self._fit_fstar = False
    
        return self._fit_fstar       
    
    @property
    def fit_Lh(self):
        if not hasattr(self, '_fit_Lh'):
            self._fit_Lh = not self.fit_fstar
    
        return self._fit_Lh

    def fstar(self, z, M, *coeff):
        #zok = np.all(self.zlim[0] <= z <= self.zlim[1])
                        
        # If we're extrapolating no matter what or we're within the
        # allowed redshift and mass range, we're done                
        #if (zok or (self.zext[0] is None)) and (self.Mext is None):
        return 10**self._log_fstar(z, M, *coeff)
        
        ##
        # OTHERWISE, WE ARE DOING SOME EXTRAPOLATING
        ##
        
        #elif (self.Mext is None) and (self.zext[0] == 'const'):
        #
        #    if z > self.zext[2]:
        #        fst = self.fstar_zhi(M, *coeff)
        #    elif z < self.zext[1]:
        #        fst = self.fstar_zlo(M, *coeff)
        #    else:
        #        fst = 10**self._log_fstar(z, M, *coeff)
        #
        ## Fit a power-law to the last few points (in mass).
        ## Allow the redshift evolution to do its thing
        #elif self.Mext in ['pl', 'exp']:
        #    fst = self.fstar_Mlo(z, M, *coeff)
        #  
        #elif (self.Mext in ['floor', 'pl_floor']):
        #    fst = 10**self._log_fstar(z, M, *coeff)
        #elif (self.Mext == 'const') or (self.Mext == 'const'):
        #
        ## Set a constant upper limit for fstar below given mass limit     
        ## Linearly interpolate in redshift to find this limit?
        #    #fstar_Mmin = self.fstar_Mlo[0]
        #    def tmp(zz, MM):
        #        if type(MM) is np.ndarray:
        #            Mlo = np.ones_like(MM[np.argwhere(MM < fstar_Mmin)]) \
        #                * fstar_Mmin
        #            Mhi = MM[np.argwhere(MM >= fstar_Mmin)]
        #                                
        #            fst_lo = self._fstar_func(zz, Mlo, *self._fstar_coeff)
        #            fst_hi = self._fstar_func(zz, Mhi, *self._fstar_coeff)
        #                                
        #            fst = np.concatenate((fst_lo, fst_hi)).squeeze()
        #            
        #        elif MM > 10**self.pf['pop_logM'][0]:
        #            fst = self._fstar_func(zz, MM, *self._fstar_coeff)
        #        else:
        #            fst = self._fstar_func(zz, fstar_Mmin, *self._fstar_coeff)
        #            
        #else:
        #    raise ValueError('Unknown pop extrap option!')
        #
        #return fst

    @property
    def kappa_UV(self):
        if not hasattr(self, '_kappa_UV'):
            if self.galaxy.sed_tab:
                self._kappa_UV = self.galaxy.src.pop.kappa_UV()
            else:
                self._kappa_UV = self.pf['pop_kappa_UV']
            
        return self._kappa_UV    
    
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
    
        self._MofL_tab = [[] for i in range(len(self.constraints['z']))]
        self._LofM_tab = [[] for i in range(len(self.constraints['z']))]
    
        # Do it already    
        for i, z in enumerate(self.redshifts):
    
            mags = []
            for mag in self.mags[i]:
                mags.append(mag-self.A1600(z, mag))
    
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
    
                Mmin = np.exp(fsolve(to_min, 10., factor=0.01, maxfev=1000)[0])
    
                self._MofL_tab[i].append(Mmin)
                self._LofM_tab[i].append(LUV_dc[j])
                self._fstar_tab[i].append(Lmin * self.kappa_UV \
                    / eta / self.cosm.fbaryon / self.Macc(z, Mmin))
    
                pb.update(i * Nm + j + 1)
    
        pb.finish()    
    
        return self._fstar_tab
    
    @property
    def eta(self):
        """
        Correction factor for Macc.
    
        \eta(z) \int_{M_{\min}}^{\infty} \dot{M}_{\mathrm{acc}}(z,M) n(z,M) dM
            = \bar{\rho}_m^0 \frac{df_{\mathrm{coll}}}{dt}|_{M_{\min}}
    
        """

        # Prepare to compute eta
        if not hasattr(self, '_eta'):        
    
            self._eta = np.zeros_like(self.halos.z)
    
            for i, z in enumerate(self.halos.z):
    
                # eta = rhs / lhs
    
                Mmin = self.Mmin[i]
    
                rhs = self.cosm.rho_m_z0 * self.dfcolldt(z)
                rhs *= (s_per_yr / g_per_msun) * cm_per_mpc**3
    
                # Accretion onto all halos (of mass M) at this redshift
                # This is *matter*, not *baryons*
                Macc = self.Macc(z, self.halos.M)
    
                # Find Mmin in self.halos.M
                j1 = np.argmin(np.abs(Mmin - self.halos.M))
                if Mmin > self.halos.M[j1]:
                    j1 -= 1
    
                integ = self.halos.dndlnm[i] * Macc
    
                p0 = simps(integ[j1-1:], x=self.halos.lnM[j1-1:])
                p1 = simps(integ[j1:], x=self.halos.lnM[j1:])
                p2 = simps(integ[j1+1:], x=self.halos.lnM[j1+1:])
                p3 = simps(integ[j1+2:], x=self.halos.lnM[j1+2:])
    
                interp = interp1d(self.halos.lnM[j1-1:j1+3], [p0,p1,p2,p3])
    
                lhs = interp(np.log(Mmin))
    
                self._eta[i] = rhs / lhs
    
        return self._eta
        
    def Lh_of_M(self, z):    
        eta = np.interp(z, self.halos.z, self.eta)
        
        Lh = self.cosm.fbaryon * self.Macc(z, self.halos.M) \
            * eta * self.SFE(z, self.halos.M) / self.kappa_UV
            
        return self.halos.M, Lh
        
    def SFR(self, z, M):
        eta = np.interp(z, self.halos.z, self.eta)
        return self.cosm.fbaryon * self.Macc(z, M) * eta * self.SFE(z, M)
        
    def LuminosityFunction(self, z, mags=False, undo_dc=False):
        """
        Reconstructed luminosity function.
        
        ..note:: This is number per [abcissa]
                
        Parameters
        ----------
        z : int, float
            Redshift. Will interpolate between values in halos.z if necessary.
        mags : bool
            If True, x-values will be in absolute (AB) magnitudes
        dc : bool
            If True, magnitudes will be corrected for dust attenuation.
            
        Returns
        -------
        Magnitudes (or luminosities) and number density.

        """
                
        Lh = self.SFR(z, self.halos.M) / self.kappa_UV
        
        dMh_dLh = np.diff(self.halos.M) / np.diff(Lh)
        
        dndm = interp1d(self.halos.z, self.halos.dndm[:,:-1], axis=0)
        
        # Only return stuff above Mmin
        Mmin = np.interp(z, self.halos.z, self.Mmin)
        
        above_Mmin = self.halos.M >= Mmin
        
        phi_of_L = dndm(z) * dMh_dLh
        
        if mags:
            MAB = self.magsys.L_to_MAB(Lh, z=z)
            if undo_dc:
                MAB += self.A1600(z, MAB)
            phi_of_L *= np.abs(np.diff(Lh) / np.diff(MAB))
            return MAB[:-1] * above_Mmin[0:-1], phi_of_L * above_Mmin[0:-1]
        else:
            return Lh[:-1] * above_Mmin[0:-1], phi_of_L * above_Mmin[0:-1] 

    def L1600_limit(self, z):
        eta = np.interp(z, self.halos.z, self.eta)
        Mmin = np.interp(z, self.halos.z, self.Mmin)
        
        #sfr_M_z = RectBivariateSpline(self.halos.z, self.halos.lnM, 
        #    np.log(self.sfr_tab))
            
        #Lh_Mmin = np.exp(sfr_M_z(z, np.log(Mmin))[0][0]) / self.kappa_UV   
        
        return self.cosm.fbaryon * self.Macc(z, Mmin) \
            * eta * self.SFE(z, Mmin) / self.kappa_UV
            
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

    @property
    def Mmin(self):
        if not hasattr(self, '_Mmin'):
            # First, compute threshold mass vs. redshift
            if self.pf['pop_Mmin'] is not None:
                self._Mmin = self.pf['pop_Mmin'] * np.ones(self.halos.Nz)
            else:
                Mvir = lambda z: self.halos.VirialMass(self.pf['pop_Tmin'], 
                    z, mu=self.pf['mu'])
                self._Mmin = np.array(map(Mvir, self.halos.z))

        return self._Mmin    

    @property
    def sfr_tab(self):
        """
        SFR as a function of redshift and halo mass yielded by abundance match.

            ..note:: Units are Msun/yr.
    
        """
        if not hasattr(self, '_sfr_tab'):
            self._sfr_tab = np.zeros([self.halos.Nz, self.halos.Nm])
            for i, z in enumerate(self.halos.z):
                self._sfr_tab[i] = self.eta[i] * self.Macc(z, self.halos.M) \
                    * self.cosm.fbaryon * self.SFE(z, self.halos.M)
    
                mask = self.halos.M >= self.Mmin[i]
                self._sfr_tab[i] *= mask
    
        return self._sfr_tab
                
    @property
    def sfrd_tab(self):
        """
        SFRD as a function of redshift yielded by abundance match.
    
            ..note:: Units are g/s/cm^3 (comoving).
    
        """
        if not hasattr(self, '_sfrd_tab'):
            self._sfrd_tab = np.zeros(self.halos.Nz)
            
            for i, z in enumerate(self.halos.z):
                integrand = self.sfr_tab[i] * self.halos.dndlnm[i]
 
                tot = np.trapz(integrand, x=self.halos.lnM)
                cumtot = cumtrapz(integrand, x=self.halos.lnM, initial=0.0)
                
                self._sfrd_tab[i] = tot - \
                    np.interp(np.log(self.Mmin[i]), self.halos.lnM, cumtot)
                
            self._sfrd_tab *= g_per_msun / s_per_yr / cm_per_mpc**3

        return self._sfrd_tab
        
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
                
    @property
    def coeff(self):
        if not hasattr(self, '_coeff'):
            if self.fit_fstar:
                self._coeff = self.coeff_fstar
            else:
                self._coeff = self.coeff_mtl
                
        return self._coeff
                
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
    
    def SFE(self, z, M):
        """
        Compute the star-formation efficiency.
        
        If outside the bounds, must extrapolate.
        """
        
        if self.fit_fstar:
            return self.fstar(z, M, *self.coeff)

        # Otherwise, we fit the mass-to-light ratio
        eta = np.interp(z, self.halos.z, self.eta)
        
        return self.Lh(z, M) * self.kappa_UV \
            / (self.cosm.fbaryon * self.Macc(z, M) * eta)        
            
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
                
        if (self.Mext == 'floor'):
            f += self.Mext[1]
        elif self.Mext == 'pl_floor' and self._apply_floor:
            self._apply_floor = 0
            to_add = self.Mext_pars[0] * 10**self._log_fstar(z, 1e10, *coeff)
            to_add *= (M / 1e10)**self.Mext_pars[1] * np.exp(-M / 1e11)
            f += to_add
            self._apply_floor = 1
            
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
        
    @property
    def pop_id(self):
        # Pop ID number for HAM population
        if not hasattr(self, '_pop_id'):
            for i, pf in enumerate(self.pf.pfs):
                if pf['pop_model'] == 'ham':
                    break
            
            self._pop_id = i
        
        return self._pop_id

    @property
    def irrelevant(self):
        if not hasattr(self, '_irrelevant'):
            if self.pf.pfs[self.pop_id]['pop_model'] != 'ham':
                self._irrelevant = True
            else:
                self._irrelevant = False

        return self._irrelevant

    @property
    def Mfunc(self):
        return self.pf.pfs[self.pop_id]['pop_ham_Mfun']
    
    @property
    def zfunc(self):
        return self.pf.pfs[self.pop_id]['pop_ham_zfun']
    
    @property
    def Mext(self):
        return self.pf.pfs[self.pop_id]['pop_ham_Mext']
    
    @property
    def Mext_pars(self):
        return self.pf.pfs[self.pop_id]['pop_ham_Mext_par1'], \
            self.pf.pfs[self.pop_id]['pop_ham_Mext_par2']

    @property
    def zext(self):
        return self.pf.pfs[self.pop_id]['pop_ham_zext'], \
              self.pf.pfs[self.pop_id]['pop_ham_zext_par']

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

    def Mpeak(self, z):
        """
        The mass at which the star formation efficiency peaks.
        """
        
        alpha = lambda MM: self.gamma_sfe(z, MM)
            
        i = np.argmin(np.abs(z - np.array(self.redshifts)))
        guess = self.MofL_tab[i][np.argmax(self.fstar_tab[i])]

        return fsolve(alpha, x0=guess)#, maxiter=1e4, full_output=False,
            #xtol=1e-3, ftol=1e-5, disp=False)[0]
            
    def fpeak(self, z):
        return self.SFE(z, self.Mpeak(z))
    
    def gamma_sfe(self, z, M):
        """
        This is a power-law index describing the relationship between the
        SFE and and halo mass.
        
        Parameters
        ----------
        z : int, float
            Redshift
        M : int, float
            Halo mass in [Msun]
            
        """
        
        fst = lambda MM: self.SFE(z, MM)
        
        return derivative(fst, M, dx=1e6) * M / fst(M)
            
    def gamma_LhMh(self, z, M):
        """
        Slope in Lh(Mh).
        """
        pass
        

            
            
            
            
            
