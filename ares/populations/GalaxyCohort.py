"""

GalaxyCohort.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Wed Jan 13 09:49:00 PST 2016

Description: 

"""

import re
import numpy as np
from ..util import read_lit
from types import FunctionType
from .GalaxyAggregate import GalaxyAggregate
from scipy.optimize import fsolve, curve_fit
from ..util import MagnitudeSystem, ProgressBar
from ..phenom.DustCorrection import DustCorrection
from scipy.integrate import quad, simps, cumtrapz, ode
from ..util.ParameterFile import par_info, get_pq_pars
from ..physics.RateCoefficients import RateCoefficients
from scipy.interpolate import interp1d, RectBivariateSpline
from ..util.Math import central_difference, interp1d_wrapper
from ..phenom.ParameterizedQuantity import ParameterizedQuantity
from ..physics.Constants import s_per_yr, g_per_msun, cm_per_mpc, G, m_p, \
    k_B, h_p, erg_per_ev, ev_per_hz

try:
    from scipy.misc import derivative
except ImportError:
    pass
    
z0 = 9. # arbitrary
tiny_phi = 1e-18
_sed_tab_attributes = ['Nion', 'Nlw', 'rad_yield', 'L1600_per_sfr']
    
class GalaxyCohort(GalaxyAggregate):
    
    @property
    def dust(self):
        if not hasattr(self, '_dust'):
            self._dust = DustCorrection(**self.pf)
        return self._dust
    
    @property
    def magsys(self):
        if not hasattr(self, '_magsys'):
            self._magsys = MagnitudeSystem(**self.pf)
    
        return self._magsys
        
    def _update_pq_registry(self, name, obj):
        if not hasattr(self, '_pq_registry'):
            self._pq_registry = {}
        
        if name in self._pq_registry:
            raise KeyError('%s already in registry!' % name)
        
        self._pq_registry[name] = obj
        
    def __getattr__(self, name):
        """
        This gets called anytime we try to fetch an attribute that doesn't
        exist (yet). The only special case is really L1600_per_sfr, since 
        that requires accessing a SynthesisModel.
        """
            
        # Indicates that this attribute is being accessed from within a 
        # property. Don't want to override that behavior!
        if (name[0] == '_'):
            raise AttributeError('This will get caught. Don\'t worry!')
            
        # This is the name of the thing as it appears in the parameter file.
        full_name = 'pop_' + name
                
        # Now, possibly make an attribute
        if not hasattr(self, name):
            
            try:
                is_php = self.pf[full_name][0:2] == 'pq'
            except (IndexError, TypeError):
                is_php = False
                
            # A few special cases    
            if self.sed_tab and (name in _sed_tab_attributes):
                if self.pf['pop_Z'] == 'sam':
                    tmp = []
                    Zarr = np.sort(self.src.metallicities.values())
                    for Z in Zarr:
                        kw = self.src_kwargs.copy()
                        kw['pop_Z'] = Z
                        src = self._Source(**kw)
                        
                        att = src.__getattribute__(name)
                        
                        # Must specify band
                        if name == 'rad_yield':
                            val = att(self.pf['pop_EminNorm'], self.pf['pop_EmaxNorm'])
                        else:
                            val = att
                            
                        tmp.append(val)

                    # Interpolant
                    interp = interp1d_wrapper(np.log10(Zarr), tmp, 
                        self.pf['interp_Z'])

                    result = lambda **kwargs: interp(np.log10(self.Zgas(kwargs['z'], kwargs['Mh'])))
                else:
                    att = self.src.__getattribute__(name)
                                        
                    if name == 'rad_yield':
                        val = att(self.src.Emin, self.src.Emax)
                    else:
                        val = att
                                                
                    result = lambda **kwargs: val

            elif is_php:
                tmp = get_pq_pars(self.pf[full_name], self.pf) 
                
                # Correct values that are strings:
                if self.sed_tab:
                    pars = {}
                    for par in tmp:
                        if tmp[par] == 'from_sed':
                            pars[par] = self.src.__getattribute__(name)
                        else:
                            pars[par] = tmp[par]  
                else:
                    pars = tmp            
                    
                result = ParameterizedQuantity(**pars)
                
                self._update_pq_registry(name, result)
            
            elif type(self.pf[full_name]) in [float, np.float64]:
                result = lambda **kwargs: self.pf[full_name]

            else:
                raise TypeError('dunno how to handle: %s' % name)

            # Check to see if Z?

            self.__setattr__(name, result)

            #self.__dict__[name] = result

        return getattr(self, name)
        #return self.__dict__[name]

    def Zgas(self, z, Mh):
        if not hasattr(self, '_sam_data'):
            self._sam_z, self._sam_data = self.scaling_relations

        flip = self._sam_data['Mh'][0] > self._sam_data['Mh'][-1]
        slc = slice(-1,0,-1) if flip else None
        
        if self.constant_SFE:
            _Mh = self._sam_data['Mh'][slc]
            _Z = self._sam_data['Z'][slc]
        else:
            # guaranteed to be a grid point?    
            k = np.argmin(np.abs(self.halos.z - z))

            _smhm = self._sam_data['Ms'][slc,k]  / self._sam_data['Mh'][slc,k]
            _mask = np.isfinite(smhm)
            _Mh = self._sam_data['Mh'][slc,k][_mask]
            _Z = self._sam_data['Z'][slc,k][_mask]

        return np.interp(Mh, _Mh, _Z)

    def N_per_Msun(self, Emin, Emax):
        """
        Compute photon luminosity in band of interest per unit SFR for 
        all halos.
        
        Returns
        -------
        In units of photons/Msun.
        
        """
        if not hasattr(self, '_N_per_Msun'):
            self._N_per_Msun = {}

        # If we've already figured it out, just return    
        if (Emin, Emax) in self._N_per_Msun:    
            return self._N_per_Msun[(Emin, Emax)]

        # Otherwise, calculate what it should be
        if (Emin, Emax) == (13.6, 24.6):
            # Should be based on energy at this point, not photon number
            self._N_per_Msun[(Emin, Emax)] = self.Nion(Mh=self.halos.M) \
                * self.cosm.b_per_msun
        elif (Emin, Emax) == (10.2, 13.6):
            self._N_per_Msun[(Emin, Emax)] = self.Nlw(Mh=self.halos.M) \
                * self.cosm.b_per_msun
        else:
            s = 'Unrecognized band: (%.3g, %.3g)' % (Emin, Emax)
            return 0.0
            #raise NotImplementedError(s)
            
        return self._N_per_Msun[(Emin, Emax)]

    def rho_L(self, Emin=None, Emax=None):
        """
        Compute the luminosity density in some bandpass for all redshifts.
        
        This is the most general way of computing the luminosity density as it
        takes into account all (existing) Mh- and z-dependent quantities.

        Returns
        -------
        Interpolant for luminosity density in units of erg / s / (comoving cm)**3.
        """

        if not hasattr(self, '_rho_L'):
            self._rho_L = {}
    
        # If we've already figured it out, just return    
        if (Emin, Emax) in self._rho_L:
            return self._rho_L[(Emin, Emax)]        
        
        # If nothing is supplied, compute the "full" luminosity density
        if (Emin is None) and (Emax is None):
            Emin = self.pf['pop_EminNorm']
            Emax = self.pf['pop_EmaxNorm']
        # Make sure we don't emit and bands...where we shouldn't be emitting
        elif (Emin is not None) and (Emax is not None):
            if (Emin > self.pf['pop_Emax']):
                self._rho_L[(Emin, Emax)] = lambda z: 0.0
                return self._rho_L[(Emin, Emax)]
            if (Emax < self.pf['pop_Emin']):
                self._rho_L[(Emin, Emax)] = lambda z: 0.0
                return self._rho_L[(Emin, Emax)]
        else:
            raise ValueError('help!')

        need_sam = False

        # For all halos. Reduce to a function of redshift only by passing
        # in the array of halo masses stored in 'halos' attribute.
        if Emax <= 24.6:
            N_per_Msun = self.N_per_Msun(Emin=Emin, Emax=Emax)

            # Also need energy per photon in this case
            erg_per_phot = self.src.erg_per_phot(Emin, Emax)

            # Get an array for fesc
            if (Emin, Emax) == (13.6, 24.6):
                fesc = lambda **kwargs: self.fesc(**kwargs)
            elif (Emin, Emax) == (10.2, 13.6):
                fesc = lambda **kwargs: self.fesc_LW(**kwargs)
            else:
                return None
  
            yield_per_sfr = lambda **kwargs: fesc(**kwargs) \
                * N_per_Msun * erg_per_phot            

        else:
            # X-rays separate because we never have lookup table.
            # could change in the future.

            try:
                if self.rad_yield.func_var not in ['z', 'Mh']:
                    need_sam = True
            except AttributeError:
                pass

            if need_sam:
                sam_z, sam_data = self.scaling_relations
            else:
                pass

            yield_per_sfr = lambda **kwargs: self.rad_yield(**kwargs) \
                * s_per_yr

        tab = np.zeros(self.halos.Nz)
        for i, z in enumerate(self.halos.z):

            if z > self.zform:
                continue      

            # Must grab stuff vs. Mh and interpolate to self.halos.M
            # They are guaranteed to have the same redshifts.
            if need_sam:

                kw = {'z': z, 'Mh': self.halos.M}                
                if self.constant_SFE:
                    for key in sam_data.keys():
                        if key == 'Mh':
                            continue

                        kw[key] = np.interp(self.halos.M,
                            sam_data['Mh'][-1::-1], sam_data[key][-1::-1])
                else:
                    raise NotImplemented('help')

            else:
                kw = {'z': z, 'Mh': self.halos.M}

            integrand = self.sfr_tab[i] * self.halos.dndlnm[i] \
                * yield_per_sfr(**kw)

            _tot = np.trapz(integrand, x=self.halos.lnM)
            _cumtot = cumtrapz(integrand, x=self.halos.lnM, initial=0.0)

            _tmp = _tot - \
                np.interp(np.log(self.Mmin[i]), self.halos.lnM, _cumtot)
               
            tab[i] = _tmp
                
        tab *= 1. / s_per_yr / cm_per_mpc**3
        
        self._rho_L[(Emin, Emax)] = interp1d(self.halos.z, tab, kind='cubic')
    
        return self._rho_L[(Emin, Emax)]
    
    def rho_N(self, z, Emin, Emax):
        """
        Compute the photon luminosity density in some bandpass at some redshift.
        
        Returns
        -------
        Luminosity density in units of photons / s / (comoving cm)**3.
        """
        
        if not hasattr(self, '_rho_N'):
            self._rho_N = {}
        
        # If we've already figured it out, just return    
        if (Emin, Emax) in self._rho_N:    
            return self._rho_N[(Emin, Emax)](z)
            
        tab = np.zeros(self.halos.Nz)
        
        # For all halos
        N_per_Msun = self.N_per_Msun(Emin=Emin, Emax=Emax)
        
        if (Emin, Emax) == (13.6, 24.6):
            fesc = self.fesc(z=z, Mh=self.halos.M)
        elif (Emin, Emax) == (10.2, 13.6):
            fesc = self.fesc_LW(z=z, Mh=self.halos.M)
        else:
            raise NotImplemented('help!')
    
        for i, z in enumerate(self.halos.z):
            integrand = self.sfr_tab[i] * self.halos.dndlnm[i] \
                * N_per_Msun * fesc
    
            tot = np.trapz(integrand, x=self.halos.lnM)
            cumtot = cumtrapz(integrand, x=self.halos.lnM, initial=0.0)
            
            tab[i] = tot - \
                np.interp(np.log(self.Mmin[i]), self.halos.lnM, cumtot)
            
        tab *= 1. / s_per_yr / cm_per_mpc**3
        
        self._rho_N[(Emin, Emax)] = interp1d(self.halos.z, tab, kind='cubic')
    
        return self._rho_N[(Emin, Emax)](z)
        
    def _sfrd_func(self, z):
        # This is a cheat so that the SFRD spline isn't constructed
        # until CALLED. Used only for tunneling (see `pop_tunnel` parameter). 
        return self.SFRD(z)
        
    @property   
    def SFRD(self):
        """
        Compute star-formation rate density (SFRD).
        """
        
        if not hasattr(self, '_SFRD'):
            self._SFRD = interp1d(self.halos.z, self.sfrd_tab, kind='cubic')

        return self._SFRD
        
    @SFRD.setter
    def SFRD(self, value):
        self._SFRD = value    
    
    @property   
    def SMD(self):
        """
        Compute stellar mass density (SMD).
        """
    
        if not hasattr(self, '_SMD'):
            dtdz = np.array(map(self.cosm.dtdz, self.halos.z))
            self._smd_tab = cumtrapz(self.sfrd_tab[-1::-1] * dtdz[-1::-1], 
                dx=np.abs(np.diff(self.halos.z[-1::-1])), initial=0.)[-1::-1]
            self._SMD = interp1d(self.halos.z, self._smd_tab, kind='cubic')
    
        return self._SMD
    
    @property
    def MAR(self):
        """
        Mass accretion rate onto halos of mass M at redshift z.
    
        ..note:: This is the *DM* mass accretion rate. To obtain the baryonic 
            accretion rate, multiply by Cosmology.fbaryon.
            
        """
        if not hasattr(self, '_MAR'):
            if self.pf['pop_MAR'] is None:
                self._MAR = None
            elif type(self.pf['pop_MAR']) is FunctionType:
                self._MAR = self.pf['pop_MAR']
            elif self.pf['pop_MAR'] == 'pl':
                raise NotImplemented('do this')
            elif self.pf['pop_MAR'] == 'hmf':
                self._MAR = self.halos.MAR_func
            else:
                self._MAR = read_lit(self.pf['pop_MAR'], verbose=self.pf['verbose']).MAR

        return self._MAR
        
    def iMAR(self, z, source=None):
        """
        The integrated mass (*matter*, i.e., baryons + CDM) accretion rate.
        
        Parameters
        ----------
        z : int, float
            Redshift
        source : str
            Can be a litdata module, e.g., 'mcbride2009'.
            If None, will compute from HMF.
            
        Returns
        -------
        Integrated DM mass accretion rate in units of Msun/yr/cMpc**3.
            
        """    

        if source is not None:
            src = read_lit(source, verbose=self.pf['verbose'])

            i = np.argmin(np.abs(z - self.halos.z))

            # Integrand: convert MAR from DM MAR to total matter MAR
            integ = self.halos.dndlnm[i] \
                * src.MAR(z, self.halos.M) / self.cosm.fcdm

            Mmin = np.interp(z, self.halos.z, self.Mmin)
            j1 = np.argmin(np.abs(Mmin - self.halos.M))
            if Mmin > self.halos.M[j1]:
                j1 -= 1

            p0 = simps(integ[j1-1:], x=self.halos.lnM[j1-1:])
            p1 = simps(integ[j1:], x=self.halos.lnM[j1:])
            p2 = simps(integ[j1+1:], x=self.halos.lnM[j1+1:])
            p3 = simps(integ[j1+2:], x=self.halos.lnM[j1+2:])

            interp = interp1d(self.halos.lnM[j1-1:j1+3], [p0,p1,p2,p3])

            return interp(np.log(Mmin))
        else:
            return super(GalaxyCohort, self).iMAR(z)

    def cMAR(self, z, source=None):
        """
        Compute cumulative mass accretion rate, i.e., integrated MAR in 
        halos Mmin<M'<M.
        
        Parameters
        ----------
        z : int, float
        """
        
        if source is not None:        
            src = read_lit(source, verbose=self.pf['verbose'])
            MAR = src.MAR(z, self.halos.M)    
        else:
            MAR = super(GalaxyCohort, self).MAR_via_AM(z)
                    
        # Grab redshift
        k = np.argmin(np.abs(z - self.halos.z))

        integ = self.halos.dndlnm[k] * MAR / self.cosm.fcdm

        Mmin = np.interp(z, self.halos.z, self.Mmin)
        j1 = np.argmin(np.abs(Mmin - self.halos.M))
        if Mmin > self.halos.M[j1]:
            j1 -= 1    

        incremental_Macc = cumtrapz(integ[j1:], x=self.halos.lnM[j1:],
            initial=0.0)

        return self.halos.M[j1:], incremental_Macc

    @property
    def eta(self):
        """
        Correction factor for MAR.
    
        \eta(z) \int_{M_{\min}}^{\infty} \dot{M}_{\mathrm{acc}}(z,M) n(z,M) dM
            = \bar{\rho}_m^0 \frac{df_{\mathrm{coll}}}{dt}|_{M_{\min}}

        """

        # Prepare to compute eta
        if not hasattr(self, '_eta'):
        
            if self.pf['pop_MAR_conserve_norm']:
                
                _rhs = np.zeros_like(self.halos.z)
                _lhs = np.zeros_like(self.halos.z)
                self._eta = np.ones_like(self.halos.z)

                for i, z in enumerate(self.halos.z):

                    # eta = rhs / lhs

                    Mmin = self.Mmin[i]

                    # My Eq. 3
                    rhs = self.cosm.rho_cdm_z0 * self.dfcolldt(z)
                    rhs *= (s_per_yr / g_per_msun) * cm_per_mpc**3
                
                    # Accretion onto all halos (of mass M) at this redshift
                    # This is *matter*, not *baryons*
                    MAR = self.MAR(z, self.halos.M)
                
                    # Find Mmin in self.halos.M
                    j1 = np.argmin(np.abs(Mmin - self.halos.M))
                    if Mmin > self.halos.M[j1]:
                        j1 -= 1
                
                    integ = self.halos.dndlnm[i] * MAR
                        
                    p0 = simps(integ[j1-1:], x=self.halos.lnM[j1-1:])
                    p1 = simps(integ[j1:], x=self.halos.lnM[j1:])
                    p2 = simps(integ[j1+1:], x=self.halos.lnM[j1+1:])
                    p3 = simps(integ[j1+2:], x=self.halos.lnM[j1+2:])
                
                    interp = interp1d(self.halos.lnM[j1-1:j1+3], [p0,p1,p2,p3])

                    lhs = interp(np.log(Mmin))
                    
                    _lhs[i] = lhs
                    _rhs[i] = rhs

                    self._eta[i] = rhs / lhs
                        
            else:
                self._eta = np.ones_like(self.halos.z)
    
        return self._eta
                
    def metallicity_in_PR(self, z, M):
        return 1e-2 * (M / 1e11)**0.48 #* 10**(-0.15 * z)
                
    @property
    def cooling_function(self):
        if not hasattr(self, '_Lambda'):
            #rc = RateCoefficients()
            #cool_ci = lambda T: rc.CollisionalIonizationCoolingRate(0, T)
            #cool_re = lambda T: rc.RadiativeRecombinationRate(0, T)
            #cool_ex = lambda T: rc.CollisionalExcitationCoolingRate(0, T)
            #self._Lambda = lambda T: cool_ci(T) + cool_re(T) + cool_ex(T)
            #M = lambda z, T: self.halos.VirialMass(T, z)
            Z = lambda z, M: 1e-2#self.metallicity_in_PR(z, M)
            self._Lambda = lambda T, z: 1.8e-22 * (1e6 / T) * 1e-2#Z(z, M(z, T))
            
        return self._Lambda
                
    def SFR(self, z, M, mu=0.6):
        """
        Star formation rate at redshift z in a halo of mass M.
        
        ..note:: Units should be solar masses per year at this point.
        """
        
        return self.pSFR(z, M, mu) * self.SFE(z=z, Mh=M)

    def pSFR(self, z, M, mu=0.6):
        """
        The product of this number and the SFE gives you the SFR.

        pre-SFR factor, hence, "pSFR"        
        """

        if z > self.zform:
            return 0.0
        
        #if self.model == 'sfe':
        eta = np.interp(z, self.halos.z, self.eta)
        return self.cosm.fbar_over_fcdm * self.MAR(z, M) * eta
        #elif self.model == 'tdyn':
        #    return self.cosm.fbaryon * M / self.tdyn(z, M)    
        #elif self.model == 'precip':
        #    T = self.halos.VirialTemperature(M, z, mu)
        #    cool = self.cooling_function(T, z)
        #    pre_factor = 3. * np.pi * G * mu * m_p * k_B * T / 50. / cool                        
        #    return pre_factor * M * s_per_yr
        #else:
        #    raise NotImplemented('Unrecognized model: %s' % self.model)
    
    def Emissivity(self, z, E=None, Emin=None, Emax=None):
        """
        Compute the emissivity of this population as a function of redshift
        and rest-frame photon energy [eV].

        Parameters
        ----------
        z : int, float

        Returns
        -------
        Emissivity in units of erg / s / c-cm**3 [/ eV]

        """

        if z > self.zform:
            return 0.0

        # Use GalaxyAggregate's Emissivity function
        if self.is_emissivity_scalable:
            # The advantage here is that the SFRD only has to be calculated
            # once, and the radiation field strength can just be determined
            # by scaling the SFRD.
            rhoL = super(GalaxyCohort, self).Emissivity(z, E=E, 
                Emin=Emin, Emax=Emax)
        else:
            # Here, the radiation backgrounds cannot just be scaled. 
            # Note that this method can always be used, it's just less 
            # efficient because you're basically calculating the SFRD again
            # and again.
            rhoL = self.rho_L(Emin, Emax)(z)

        if E is not None:
            return rhoL * self.src.Spectrum(E)
        else:
            return rhoL

    def StellarMassFunction(self, z):
        if not hasattr(self, '_phi_of_Mst'):
            self._phi_of_Mst = {}
        else:
            if z in self._phi_of_Mst:
                return self._phi_of_Mst[z]

        zform, data = self.scaling_relations

        sorter = np.argsort(data['Mh'])
        
        if self.constant_SFE:
            sorter = np.argsort(data['Mh'])
            Mh = data['Mh'][sorter]
            Ms = data['Ms'][sorter]
        else:
            k = np.argmin(np.abs(z - self.halos.z))
            Mh = data['Mh'][:,k]
            Ms = data['Ms'][:,k]
            
            sorter = np.argsort(Mh)
            Mh = Mh[sorter]
            Ms = Ms[sorter]
        
        dndm_func = interp1d(self.halos.z, self.halos.dndm[:,:-1], axis=0)
        dndm_z = dndm_func(z)

        # Interpolate dndm to same Mh grid as SAM
        dndm_sam = np.interp(Mh, self.halos.M[0:-1], dndm_z)

        dndm = dndm_sam * self.focc(z=z, Mh=Mh)
        dMh_dMs = np.diff(Mh) / np.diff(Ms)
                
        dMh_dlogMs = dMh_dMs * Ms[0:-1]
        
        # Only return stuff above Mmin
        Mmin = np.interp(z, self.halos.z, self.Mmin)
        Mmax = self.pf['pop_lf_Mmax']
        
        i_min = np.argmin(np.abs(Mmin - self.halos.M))
        i_max = np.argmin(np.abs(Mmax - self.halos.M))

        phi_of_Ms = dndm[0:-1] * dMh_dlogMs

        above_Mmin = Mh >= Mmin
        below_Mmax = Mh <= Mmax
        ok = np.logical_and(above_Mmin, below_Mmax)[0:-1]
        mask = self.mask = np.logical_not(ok)

        mass = np.ma.array(Ms[:-1], mask=mask)
        phi = np.ma.array(phi_of_Ms, mask=mask)

        phi[mask == True] = tiny_phi

        self._phi_of_Mst[z] = mass, phi

        return self._phi_of_Mst[z]
        
    
    def LuminosityFunction(self, z, x, mags=True):
        """
        Reconstructed luminosity function.
        
        ..note:: This is number per [abcissa]. No dust correction has
            been applied.
                
        Parameters
        ----------
        z : int, float
            Redshift. Will interpolate between values in halos.z if necessary.
        mags : bool
            If True, x-values will be in absolute (AB) magnitudes
        Returns
        -------
        Number density.

        """

        if mags:
            x_phi, phi = self.phi_of_M(z)

            # Setup interpolant
            interp = interp1d(x_phi, np.log10(phi), kind='linear',
                bounds_error=False, fill_value=-np.inf)

            phi_of_x = 10**interp(x)
        else:

            x_phi, phi = self.phi_of_L(z)

            # Setup interpolant
            interp = interp1d(np.log10(x_phi), np.log10(phi), kind='linear',
                bounds_error=False, fill_value=-np.inf)
            
            phi_of_x = 10**interp(np.log10(x))
                                                                
        return phi_of_x

    def Lh(self, z):
        """
        This is the rest-frame UV band in which the LF is measured.
        
        NOT generally use-able!!!
        
        """
        return self.SFR(z, self.halos.M) * self.L1600_per_sfr(z=z, Mh=self.halos.M)

    def phi_of_L(self, z):
        """
        Compute the luminosity function at redshift z.
        
        Returns
        -------
        Number of galaxies per unit luminosity per unit volume.
        
        """
        if not hasattr(self, '_phi_of_L'):
            self._phi_of_L = {}
        else:
            if z in self._phi_of_L:
                return self._phi_of_L[z]

        Lh = self.Lh(z)
        logL_Lh = np.log(Lh)
        
        dndm_func = interp1d(self.halos.z, self.halos.dndm[:,:-1], axis=0)

        dndm = dndm_func(z) * self.focc(z=z, Mh=self.halos.M[0:-1])
        dMh_dLh = np.diff(self.halos.M) / np.diff(Lh)
                
        dMh_dlogLh = dMh_dLh * Lh[0:-1]
        
        # Only return stuff above Mmin
        Mmin = np.interp(z, self.halos.z, self.Mmin)
        Mmax = self.pf['pop_lf_Mmax']
        
        i_min = np.argmin(np.abs(Mmin - self.halos.M))
        i_max = np.argmin(np.abs(Mmax - self.halos.M))

        if self.pf['pop_Lh_scatter'] > 0:
            sigma = self.pf['pop_Lh_scatter']
            norm = np.sqrt(2. * np.pi) / sigma / np.log(10.)

            gauss = lambda x, mu: np.exp(-(x - mu)**2 / 2. / sigma**2) / norm

            phi_of_L = np.zeros_like(Lh[0:-1])
            for k, logL in enumerate(logL_Lh[0:-1]):

                # Actually a range of halo masses that can produce galaxy
                # of luminosity Lh
                pdf = gauss(logL_Lh[0:-1], logL_Lh[k])

                integ = dndm[i_min:i_max] * pdf[i_min:i_max] * dMh_dlogLh[i_min:i_max]

                phi_of_L[k] = np.trapz(integ, x=logL_Lh[i_min:i_max])

            # This needs extra term now?
            phi_of_L /= Lh[0:-1]

        else:
            phi_of_L = dndm * dMh_dLh

        above_Mmin = self.halos.M >= Mmin
        below_Mmax = self.halos.M <= Mmax
        ok = np.logical_and(above_Mmin, below_Mmax)[0:-1]
        mask = self.mask = np.logical_not(ok)
        
        lum = np.ma.array(Lh[:-1], mask=mask)
        phi = np.ma.array(phi_of_L, mask=mask)

        phi[mask == True] = tiny_phi

        self._phi_of_L[z] = lum, phi

        return self._phi_of_L[z]

    def phi_of_M(self, z):
        if not hasattr(self, '_phi_of_M'):
            self._phi_of_M = {}
        else:
            if z in self._phi_of_M:
                return self._phi_of_M[z]

        Lh, phi_of_L = self.phi_of_L(z)

        MAB = self.magsys.L_to_MAB(Lh, z=z)

        phi_of_M = phi_of_L[0:-1] * np.abs(np.diff(Lh) / np.diff(MAB))

        self._phi_of_M[z] = MAB[0:-1], phi_of_M

        return self._phi_of_M[z]

    def MUV_max(self, z):
        """
        Compute the magnitude corresponding to the Tmin threshold.
        """   

        i_z = np.argmin(np.abs(z - self.halos.z))

        Mmin = np.interp(z, self.halos.z, self.Mmin)
        Lmin = np.interp(Mmin, self.halos.M, self.Lh(z))

        MAB = self.magsys.L_to_MAB(Lmin, z=z)

        return MAB

    def Mh_of_MUV(self, z, MUV):
        
        # MAB corresponds to self.halos.M
        MAB, phi = self.phi_of_M(z)

        return np.interp(MUV, MAB[-1::-1], self.halos.M[-1:1:-1])

    @property
    def Mmin(self):
        if not hasattr(self, '_Mmin'):
            # First, compute threshold mass vs. redshift
            if self.pf['pop_Mmin'] is not None:
                if type(self.pf['pop_Mmin']) is FunctionType:
                    self._Mmin = np.array(map(self.pf['pop_Mmin'], self.halos.z))
                else:    
                    self._Mmin = self.pf['pop_Mmin'] * np.ones(self.halos.Nz)
            else:
                Mvir = lambda z: self.halos.VirialMass(self.pf['pop_Tmin'], 
                    z, mu=self.pf['mu'])
                self._Mmin = np.array(map(Mvir, self.halos.z))

        return self._Mmin    

    @property
    def sfr_tab(self):
        """
        SFR as a function of redshift and halo mass.

            ..note:: Units are Msun/yr.
    
        """
        if not hasattr(self, '_sfr_tab'):
            self._sfr_tab = np.zeros([self.halos.Nz, self.halos.Nm])
            for i, z in enumerate(self.halos.z):
                
                if z > self.zform:
                    continue
                
                self._sfr_tab[i] = self.eta[i] * self.MAR(z, self.halos.M) \
                    * self.cosm.fbar_over_fcdm * self.SFE(z=z, Mh=self.halos.M)
                
                mask = self.halos.M >= self.Mmin[i]
                self._sfr_tab[i] *= mask
    
        return self._sfr_tab
                
    @property
    def sfrd_tab(self):
        """
        SFRD as a function of redshift.
    
            ..note:: Units are g/s/cm^3 (comoving).
    
        """
                
        if not hasattr(self, '_sfrd_tab'):
            self._sfrd_tab = np.zeros(self.halos.Nz)
            
            for i, z in enumerate(self.halos.z):
                
                if z > self.zform:
                    continue
                
                integrand = self.sfr_tab[i] * self.halos.dndlnm[i] \
                    * self.focc(z=z, Mh=self.halos.M)
 
                tot = np.trapz(integrand, x=self.halos.lnM)
                cumtot = cumtrapz(integrand, x=self.halos.lnM, initial=0.0)
                
                self._sfrd_tab[i] = tot - \
                    np.interp(np.log(self.Mmin[i]), self.halos.lnM, cumtot)
                                
            self._sfrd_tab *= g_per_msun / s_per_yr / cm_per_mpc**3

        return self._sfrd_tab
    
    @property
    def LLyC_tab(self):
        """
        Number of LyC photons emitted per unit SFR in halos of mass M.
        """
        if not hasattr(self, '_LLyC_tab'):
            M = self.halos.M
            fesc = self.fesc(None, M)
            
            dnu = (24.6 - 13.6) / ev_per_hz
            #nrg_per_phot = 25. * erg_per_ev

            Nion_per_L1600 = self.Nion(None, M) / (1. / dnu)
            
            self._LLyC_tab = np.zeros([self.halos.Nz, self.halos.Nm])
            
            for i, z in enumerate(self.halos.z):
                self._LLyC_tab[i] = self.L1600_tab[i] * Nion_per_L1600 \
                    * fesc
            
                mask = self.halos.M >= self.Mmin[i]
                self._LLyC_tab[i] *= mask
            
        return self._LLyC_tab
                
    @property
    def LLW_tab(self):
        if not hasattr(self, '_LLW_tab'):
            M = self.halos.M
    
            dnu = (13.6 - 10.2) / ev_per_hz
            #nrg_per_phot = 25. * erg_per_ev
    
            Nlw_per_L1600 = self.Nlw(None, M) / (1. / dnu)
    
            self._LLW_tab = np.zeros([self.halos.Nz, self.halos.Nm])
    
            for i, z in enumerate(self.halos.z):
                self._LLW_tab[i] = self.L1600_tab[i] * Nlw_per_L1600
    
                mask = self.halos.M >= self.Mmin[i]
                self._LLW_tab[i] *= mask

        return self._LLW_tab

    def SFE(self, **kwargs):
        """
        Just a wrapper around self.fstar.
        """
        return self.fstar(**kwargs)

    @property
    def yield_per_sfr(self):
        # Need this to avoid inheritance issue with GalaxyAggregate
        if not hasattr(self, '_yield_per_sfr'):
            if type(self.rad_yield) is FunctionType:
                self._yield_per_sfr = self.rad_yield()
            else:
                self._yield_per_sfr = self.rad_yield
            
        return self._yield_per_sfr

    @property
    def fstar(self):
        if not hasattr(self, '_fstar'):

            if self.pf['pop_calib_L1600'] is not None:
                boost = self.pf['pop_calib_L1600'] / self.L1600_per_sfr()
            else:
                boost = 1.

            if self.pf['pop_mlf'] is not None:
                self._fstar = lambda **kwargs: boost * self.fshock(**kwargs) \
                    / ((1. / self.pf['pop_fstar_max']) + self.mlf(**kwargs))
            elif type(self.pf['pop_fstar']) in [float, np.float64]:
                self._fstar = lambda **kwargs: self.pf['pop_fstar'] * boost
            elif self.pf['pop_fstar'][0:2] == 'pq':
                pars = get_pq_pars(self.pf['pop_fstar'], self.pf)
                self._fstar_inst = ParameterizedQuantity(**pars)
                
                self._fstar = \
                    lambda **kwargs: self._fstar_inst.__call__(**kwargs) \
                    * boost
            else:
                raise ValueError('Unrecognized data type for pop_fstar!')  

        return self._fstar
        
    @fstar.setter
    def fstar(self, value):
        self._fstar = value  
                
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
            
    def alpha_lf(self, z, mag):
        """
        Slope in the luminosity function
        """

        logphi = lambda logL: np.log10(self.LuminosityFunction(z, 10**logL, mags=False))

        Mdc = mag - self.dust.AUV(z, mag)
        L = self.magsys.MAB_to_L(mag=Mdc, z=z)

        return derivative(logphi, np.log10(L), dx=0.1)
    
    @property
    def fstar_tab(self):
        if not hasattr(self, '_fstar_tab'):
            self._fstar_tab = np.zeros_like(self.halos.dndm)
    
            for i, z in enumerate(self.halos.z):    
                self._fstar_tab[i,:] = self.SFE(z, self.halos.M)
    
        return self._fstar_tab
    
    def _SAM(self, z, y):
        if self.pf['pop_sam_nz'] == 1:
            return self._SAM_1z(z, y)
        elif self.pf['pop_sam_nz'] == 2:
            return self._SAM_2z(z, y)
        else:
            raise NotImplemented('No SAM with nz=%i' % self.pf['pop_sam_nz'])
            
    def _SAM_1z(self, z, y):
        """
        Simple semi-analytic model for the components of galaxies.

        Really just the right-hand sides of a set of ODEs describing the
        rate of change in the halo mass, stellar mass, and metal mass.
        Other elements can be added quite easily.

        Parameters
        ----------
        z : int, float
            current redshift
        y : array
            halo mass, gas mass, stellar mass, gas-phase metallicity

        Returns
        -------
        An updated array of y-values.

        """

        Mh, Mg, Mst, MZ = y
        
        kw = {'z':z, 'Mh': Mh, 'Ms': Mst, 'Mg': Mg}
        
        fstar = self.SFE(**kw)

        # Eq. 1: halo mass.
        y1p = -1. * self.MAR(z, Mh) * self.cosm.dtdz(z) / s_per_yr

        # Eq. 2: gas mass
        y2p = self.cosm.fbar_over_fcdm * y1p * (1. - fstar)

        #_sfr_res = Mg * self.pf['pop_fstar_res'] / 1e7
        #
        #y2p -= min(m_sfr_res, y2p)

        # Add option of parameterized stifling of gas supply, and
        # ejection of gas.

        # Eq. 3: stellar mass
        Mmin = np.interp(z, self.halos.z, self.Mmin)
        if Mh < Mmin:
            y3p = 0.
        else:
            y3p = fstar * self.cosm.fbar_over_fcdm * y1p \
                * (1. - self.pf['pop_mass_yield'])

        # Eq. 4: metal mass -- constant return per unit star formation for now
        # Could make a PHP pretty easily.
        y4p = self.pf['pop_mass_yield'] * self.pf['pop_metal_yield'] * y3p \
            * (1. - self.pf['pop_mass_escape'])

        # Stuff to add: parameterize metal yield, metal escape, star formation
        # from reservoir? How to deal with Mmin(z)? Initial conditions (from PopIII)?

        results = [y1p, y2p, y3p, y4p]
                
        return np.array(results)    
        
    def _SAM_2z(self, z, y):
        raise NotImplemented('Super not done with this!')
        
        Mh, Mg_cgm, Mg_ism_c, Mg_ism_h, Mst, MZ_ism, MZ_cgm = y
        
        kw = {'z':z, 'Mh': Mh, 'Ms': Mst, 'Mg': Mg}
        
        #
        fstar = self.SFE(**kw)
        tstar = 1e7 * s_per_yr
        
        Mdot_h = -1. * self.MAR(z, Mh) * self.cosm.dtdz(z) / s_per_yr

        # Need cooling curve here eventually.
        Z_cgm = MZ_cgm / Mg_cgm
        Mdot_precip = 0. # Mg_cgm

        # Eq. 1: halo mass.
        y1p = Mh_dot

        # Eq. 2: CGM mass
        # Pristine inflow + gas injected from galaxy winds / SN 
        # - precipitation rate - losses from halo
        y2p = y1p * self.pf['pop_fstall'] 

        # Eq. 3: Hot ISM gas mass
        # Winds and SN, gas precipitated from CGM?
        y3p = y1p * (1. - self.pf['pop_fstall'])
        
        # Eq. 4: Cold ISM gas mass
        # Pristine inflow + hot ISM cooled off - star formation
        y4p = y1p * (1. - self.pf['pop_fstall'])
        
        # Eq. 5: Star formation
        Mmin = np.interp(z, self.halos.z, self.Mmin)
        if Mh < Mmin:
            y5p = 0.
        else:
            y5p = fstar * (self.cosm.fbar_over_fcdm * y1p * (1. - self.pf['pop_fstall']) \
                + (self.pf['pop_fstar_rec'] / fstar) * Mg_ism_c / tstar)

        # Eq. 4: metal mass -- constant return per unit star formation for now
        # Could make a PHP pretty easily.
        y6p = self.pf['pop_metal_yield'] * y3p * (1. - self.pf['pop_mass_escape'])

        results = [y1p, y2p, y3p, y4p]
                
        return np.array(results)    
        
    @property
    def constant_SFE(self):
        if not hasattr(self, '_constant_SFE'):
            self._constant_SFE = self.fstar(z=10, Mh=1e11) \
                              == self.fstar(z=11, Mh=1e11)
        return self._constant_SFE
        
    @property
    def scaling_relations(self):
        if not hasattr(self, '_scaling_relations'):
            if self.constant_SFE:
                self._scaling_relations = self._ScalingRelationsStaticSFE()
            else:
                self._scaling_relations = self._ScalingRelationsGeneralSFE()
            
        return self._scaling_relations    
            
    def _ScalingRelationsGeneralSFE(self):
        """
        In this case, the formation time of a halo matters.
        
        Returns
        -------
        Dictionary of quantities, each having shape (z, z). 
        The first dimension corresponds to formation time, the second axis
        represents trajectories.
        """
        
        
        keys = ['Mh', 'Mg', 'Ms', 'MZ']
                
        results = {key:np.zeros([self.halos.z.size]*2) for key in keys}
        
        #results = {key:[] for key in keys}
        zform = []
        results['nthresh'] = np.zeros_like(self.halos.z)
        #results['zform'] = []
        
        zfreq = int(self.pf['pop_sam_dz'] / np.diff(self.halos.z)[0])

        for i, z in enumerate(self.halos.z):
            if (i == 0) or (i == len(self.halos.z) - 1):
                continue

            if i % zfreq != 0:
                continue    

            _zarr, _results = self._ScalingRelationsStaticSFE(z0=z)

            for key in keys:
                results[key][i,0:i+1] = _results[key].copy()
                
                # Must keep in mind that different redshifts have different
                # halo mass gridding effectively

            ##
            # Could kill this once a user-defined mass and redshift
            # interval is adequately sampled
            ##

            #for key in keys:
            #    results[key].append(_results[key].copy())
            #
            zform.append(z)
            #results['z'].append(_zarr)
            #
            k = np.argmin(np.abs(self.halos.M - self.Mmin[i]))
            results['nthresh'][i] = self.halos.dndm[i,k]   

        return zform, results
        
    def _ScalingRelationsStaticSFE(self, z0=None):
        """
        Evolve a halo from initial mass M0 at redshift z0 forward in time.
        
        Really this should be invoked any time any PHP has 'z' in its vars list.
        
        Returns
        -------
        redshifts, halo mass, gas mass, stellar mass, metal mass

        """

        dz = max(np.diff(self.halos.z)[0], self.pf['sam_dz'])

        solver = ode(self._SAM).set_integrator('vode', method='bdf',
            nsteps=1e4, order=5)

        ##  
        # Outputs have shape (z, z)
        ##

        # Our results don't depend on this, unless SFE depends on z
        if z0 is None:
            z0 = self.halos.z.max()
            M0 = self.Mmin[-1]

            #assert np.all(np.diff(self.Mmin) == 0), \
            #    "Can only do this for constant Mmin at the moment. Sorry!"

        else:
            M0 = np.interp(z0, self.halos.z, self.Mmin)
            
        zf = float(self.halos.z.min())

        # Boundary conditions (pristine halo)
        Mg0 = self.cosm.fbar_over_fcdm * M0
        MZ0 = 0.0
        Mst0 = 0.0

        # Initial stellar mass -> 0, initial halo mass -> Mmin
        solver.set_initial_value(np.array([M0, Mg0, Mst0, MZ0]), z0)

        z = []
        Mh_t = []
        Mg_t = []
        Mst_t = []
        metals = []
        while True:

            z.append(solver.t)
            Mh_t.append(solver.y[0])
            Mg_t.append(solver.y[1])
            Mst_t.append(solver.y[2])
            metals.append(solver.y[3])

            # Annoying. Simple conditional in while was not robust.
            if abs(solver.t - zf) < 1e-10:
                break
                
            solver.integrate(solver.t-dz)

        # Everything will be returned in order of ascending redshift,
        # which will mean masses are (probably) declining from 0:-1
        z = np.array(z)[-1::-1]
        Mh = np.array(Mh_t)[-1::-1]
        Mg = np.array(Mg_t)[-1::-1]
        Ms = np.array(Mst_t)[-1::-1]
        MZ = np.array(metals)[-1::-1]
        
        # Derived
        results = {'Mh': Mh, 'Mg': Mg, 'Ms': Ms, 'MZ': MZ}
        results['Z'] = results['MZ'] / results['Mg'] / self.pf['pop_fpoll']

        return z, results

    def _LuminosityDensity_LW(self, z):
        return self.LuminosityDensity(z, Emin=10.2, Emax=13.6)

    def _LuminosityDensity_LyC(self, z):
        return self.LuminosityDensity(z, Emin=13.6, Emax=24.6)
    
    def _LuminosityDensity_sXR(self, z):
        return self.LuminosityDensity(z, Emin=200., Emax=2e3)
    
    def _LuminosityDensity_hXR(self, z):
        return self.LuminosityDensity(z, Emin=2e3, Emax=1e4)        

    def LuminosityDensity(self, z, Emin=None, Emax=None):
        """
        Return the integrated luminosity density in the (Emin, Emax) band.

        Parameters
        ----------
        z : int, flot
            Redshift of interest.
        
        Returns
        -------
        Luminosity density in erg / s / c-cm**3.
        
        """
        
        return self.Emissivity(z, E=None, Emin=Emin, Emax=Emax)

    def PhotonLuminosityDensity(self, z, Emin=None, Emax=None):
        """
        Return the photon luminosity density in the (Emin, Emax) band.
    
        Parameters
        ----------
        z : int, flot
            Redshift of interest.
    
        Returns
        -------
        Photon luminosity density in photons / s / c-cm**3.
    
        """
            
        # erg / s / cm**3
        if self.is_emissivity_scalable:
            rhoL = self.Emissivity(z, E=None, Emin=Emin, Emax=Emax)
            erg_per_phot = super(GalaxyCohort, 
                self)._get_energy_per_photon(Emin, Emax) * erg_per_ev
                               
            return rhoL / erg_per_phot
        else:
            return self.rho_N(z, Emin, Emax)
                 
