"""

GalaxyCohort.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Wed Jan 13 09:49:00 PST 2016

Description: 

"""

import re
import time
import numpy as np
from ..util import read_lit
from inspect import ismethod
from ..analysis import ModelSet
from scipy.misc import derivative
from scipy.optimize import fsolve, minimize
from types import FunctionType
from ..analysis.BlobFactory import BlobFactory
from ..util import MagnitudeSystem, ProgressBar
from ..phenom.DustCorrection import DustCorrection
from scipy.integrate import quad, simps, cumtrapz, ode
from ..util.ParameterFile import par_info, get_pq_pars
from ..physics.RateCoefficients import RateCoefficients
from scipy.interpolate import interp1d, RectBivariateSpline
from .GalaxyAggregate import GalaxyAggregate
from .Population import normalize_sed
from ..util.Math import central_difference, interp1d_wrapper
from ..phenom.ParameterizedQuantity import ParameterizedQuantity
from ..physics.Constants import s_per_yr, g_per_msun, cm_per_mpc, G, m_p, \
    k_B, h_p, erg_per_ev, ev_per_hz
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str
    
ztol = 1e-4
z0 = 9. # arbitrary
tiny_phi = 1e-18
_sed_tab_attributes = ['Nion', 'Nlw', 'rad_yield', 'L1600_per_sfr']
    
class GalaxyCohort(GalaxyAggregate,BlobFactory):
    
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
            raise KeyError('{!s} already in registry!'.format(name))
        
        self._pq_registry[name] = obj
        
    def __getattr__(self, name):
        """
        This gets called anytime we try to fetch an attribute that doesn't
        exist (yet). The only special case is really L1600_per_sfr, since 
        that requires accessing a SynthesisModel.
        """
                                        
        # Indicates that this attribute is being accessed from within a 
        # property. Don't want to override that behavior!
        # This is in general pretty dangerous but I don't have any better
        # ideas right now. It makes debugging hard but it's SO convenient...
        if (name[0] == '_'):
            raise AttributeError('Couldn\'t find attribute: {!s}'.format(name))
                    
        # This is the name of the thing as it appears in the parameter file.
        full_name = 'pop_' + name
                                
        # Now, possibly make an attribute
        try:
            is_php = self.pf[full_name][0:2] == 'pq'
        except (IndexError, TypeError):
            is_php = False
            
        # A few special cases    
        if self.sed_tab and (name in _sed_tab_attributes):
            if self.pf['pop_Z'] == 'sam':
                tmp = []
                Zarr = np.sort(list(self.src.metallicities.values()))
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
            Mmin = lambda z: self.Mmin
            result = ParameterizedQuantity({'pop_Mmin': Mmin}, self.pf, **pars)

            self._update_pq_registry(name, result)
            
        elif type(self.pf[full_name]) in [int, float, np.int64, np.float64]:
                
            # Need to be careful here: has user-specified units!
            # We've assumed that this cannot be parameterized...
            # i.e., previous elif won't ever catch rad_yield
            if name == 'rad_yield':
                result = lambda **kwargs: normalize_sed(self)
            else:
                result = lambda **kwargs: self.pf[full_name]
            
        elif type(self.pf[full_name]) is FunctionType:
            result = lambda **kwargs: self.pf[full_name](**kwargs)            
        else:
            raise TypeError('dunno how to handle: {!s}'.format(name))
            
        # Check to see if Z?
        setattr(self, name, result)
                
        return result

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
            s = 'Unrecognized band: ({0:.3g}, {1:.3g})'.format(Emin, Emax)
            return 0.0
            #raise NotImplementedError(s)
            
        return self._N_per_Msun[(Emin, Emax)]
        
    @property
    def _spline_nh(self):
        if not hasattr(self, '_spline_nh_'):
            self._spline_nh_ = \
                RectBivariateSpline(self.halos.z, self.halos.lnM, 
                    self.halos.dndm)
        return self._spline_nh_
    
    @property
    def _tab_MAR(self):
        if not hasattr(self, '_tab_MAR_'):
            self._tab_MAR_ = \
                np.array([self.MAR(self.halos.z[i], self.halos.M) \
                    for i in range(self.halos.Nz)]) 
                    
            self._tab_MAR_ = np.maximum(self._tab_MAR_, 0.0)
            
        return self._tab_MAR_
    
    @property
    def _tab_MAR_at_Mmin(self):
        if not hasattr(self, '_tab_MAR_at_Mmin_'):
            self._tab_MAR_at_Mmin_ = \
                np.array([self.MAR(self.halos.z[i], self._tab_Mmin[i]) \
                    for i in range(self.halos.Nz)])                    
        return self._tab_MAR_at_Mmin_ 
    
    @property
    def _tab_nh_at_Mmin(self):
        if not hasattr(self, '_tab_nh_at_Mmin_'):
            self._tab_nh_at_Mmin_ = \
                np.array([self._spline_nh(self.halos.z[i], 
                    np.log(self._tab_Mmin[i])) \
                    for i in range(self.halos.Nz)]).squeeze()
        return self._tab_nh_at_Mmin_
        
    @property
    def _tab_fstar_at_Mmin(self):
        if not hasattr(self, '_tab_fstar_at_Mmin_'):
            self._tab_fstar_at_Mmin_ = \
                self.SFE(z=self.halos.z, Mh=self._tab_Mmin)
        return self._tab_fstar_at_Mmin_

    @property
    def _tab_sfrd_at_threshold(self):
        """
        Star formation rate density from halos just crossing threshold.

        Essentially the second term of Equation A1 from Furlanetto+ 2017.
        """
        if not hasattr(self, '_tab_sfrd_at_threshold_'):
            if not self.pf['pop_sfr_cross_threshold']:
                self._tab_sfrd_at_threshold_ = np.zeros_like(self.halos.z)
                return self._tab_sfrd_at_threshold_

            # Model: const SFR in threshold-crossing halos.    
            if type(self.pf['pop_sfr']) in [int, float, np.float64]:
                self._tab_sfrd_at_threshold_ = self.pf['pop_sfr'] \
                    * self._tab_nh_at_Mmin * self._tab_Mmin
            else:
                active = 1. - self.fsup(z=self.halos.z) 
                self._tab_sfrd_at_threshold_ = active * self._tab_eta \
                    * self.cosm.fbar_over_fcdm * self._tab_MAR_at_Mmin \
                    * self._tab_fstar_at_Mmin * self._tab_Mmin \
                    * self._tab_nh_at_Mmin \
                    * self.focc(z=self.halos.z, Mh=self._tab_Mmin)
                            
            #self._tab_sfrd_at_threshold_ -= * self.Mmin * n * self.dMmin_dt(self.halos.z)    

            self._tab_sfrd_at_threshold_ *= g_per_msun / s_per_yr / cm_per_mpc**3

            # Don't count this "new" star formation once the minimum mass
            # exceeds some value. At this point, it will (probably, hopefully)
            # be included in the star-formation of some other population.
            if np.isfinite(self.pf['pop_sfr_cross_upto_Tmin']):
                Tlim = self.pf['pop_sfr_cross_upto_Tmin']
                Mlim = self.halos.VirialMass(T=Tlim, z=self.halos.z)

                mask = self.halos.Mmin < Mlim
                self._tab_sfrd_at_threshold_ *= mask

        return self._tab_sfrd_at_threshold_

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
        
        if not hasattr(self, '_yield_per_sfr_for_rho'):
            self._yield_per_sfr_for_rho = {}    
    
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
                
        self._yield_per_sfr_for_rho[(Emin, Emax)] = yield_per_sfr

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

            integrand = self._tab_sfr[i] * self.halos.dndlnm[i] \
                * yield_per_sfr(**kw)

            _tot = np.trapz(integrand, x=self.halos.lnM)
            _cumtot = cumtrapz(integrand, x=self.halos.lnM, initial=0.0)

            _tmp = _tot - \
                np.interp(np.log(self._tab_Mmin[i]), self.halos.lnM, _cumtot)
               
            tab[i] = _tmp
                
        tab *= 1. / s_per_yr / cm_per_mpc**3
        
        if self.pf['pop_sfr_cross_threshold']:
            
            y = yield_per_sfr(z=self.halos.z, Mh=self._tab_Mmin)
            
            if self.pf['pop_sfr'] is not None:
                thresh = self.pf['pop_sfr'] \
                    * self._tab_nh_at_Mmin * self._tab_Mmin \
                    * y / s_per_yr / cm_per_mpc**3
            else:
                active = 1. - self.fsup(z=self.halos.z)  
                thresh = active * self._tab_eta * \
                    self.cosm.fbar_over_fcdm * self._tab_MAR_at_Mmin \
                    * self._tab_fstar_at_Mmin * self._tab_Mmin \
                    * self._tab_nh_at_Mmin * y \
                    / s_per_yr / cm_per_mpc**3
        
            tab += thresh
        
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
            integrand = self._tab_sfr[i] * self.halos.dndlnm[i] \
                * N_per_Msun * fesc
    
            tot = np.trapz(integrand, x=self.halos.lnM)
            cumtot = cumtrapz(integrand, x=self.halos.lnM, initial=0.0)
            
            tab[i] = tot - \
                np.interp(np.log(self._tab_Mmin[i]), self.halos.lnM, cumtot)
            
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
            self._SFRD = interp1d(self.halos.z, self._tab_sfrd_total, 
                kind='cubic')

        return self._SFRD
        
    @SFRD.setter
    def SFRD(self, value):
        self._SFRD = value 
        
    @property   
    def nactive(self):
        """
        Compute star-formation rate density (SFRD).
        """
    
        if not hasattr(self, '_nactive'):
            self._nactive = interp1d(self.halos.z, self._tab_nh_active, 
                kind='cubic')
    
        return self._nactive
    
    @property   
    def SMD(self):
        """
        Compute stellar mass density (SMD).
        """
    
        if not hasattr(self, '_SMD'):
            dtdz = np.array(list(map(self.cosm.dtdz, self.halos.z)))
            self._smd_tab = cumtrapz(self._tab_sfrd_total[-1::-1] * dtdz[-1::-1], 
                dx=np.abs(np.diff(self.halos.z[-1::-1])), initial=0.)[-1::-1]
            self._SMD = interp1d(self.halos.z, self._smd_tab, kind='cubic')
    
        return self._SMD
    
    @property
    def MGR(self):
        """
        Mass growth rate of halos of mass M at redshift z.
    
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
        
    def MAR(self, z, Mh):
        return np.maximum(self.MGR(z, Mh) * self.fsmooth(z=z, Mh=Mh), 0.)
    
    def MDR(self, z, Mh):
        # Mass "delivery" rate
        return self.MGR(z, Mh) * (1. - self.fsmooth(z=z, Mh=Mh))
        
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
        if not hasattr(self, '_eta'):
            self._eta = lambda z: np.interp(z, self.halos.z, self._tab_eta)
        return self._eta

    @property
    def _tab_eta(self):
        """
        Correction factor for MAR.
    
        \eta(z) \int_{M_{\min}}^{\infty} \dot{M}_{\mathrm{acc}}(z,M) n(z,M) dM
            = \bar{\rho}_m^0 \frac{df_{\mathrm{coll}}}{dt}|_{M_{\min}}

        """

        # Prepare to compute eta
        if not hasattr(self, '_tab_eta_'):
        
            if self.pf['pop_MAR_conserve_norm']:
                
                _rhs = np.zeros_like(self.halos.z)
                _lhs = np.zeros_like(self.halos.z)
                self._tab_eta_ = np.ones_like(self.halos.z)

                for i, z in enumerate(self.halos.z):

                    # eta = rhs / lhs

                    Mmin = self._tab_Mmin[i]

                    # My Eq. 3
                    rhs = self.cosm.rho_cdm_z0 * self.dfcolldt(z)
                    rhs *= (s_per_yr / g_per_msun) * cm_per_mpc**3
                
                    # Accretion onto all halos (of mass M) at this redshift
                    # This is *matter*, not *baryons*
                    MAR = self._tab_MAR[i]
                
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

                    self._tab_eta_[i] = rhs / lhs
                        
            else:
                self._tab_eta_ = np.ones_like(self.halos.z)
    
        return self._tab_eta_
        
    def SFR(self, z, Mh=None):
        """
        Star formation rate at redshift z in a halo of mass Mh.
        
        If Mh is not supplied
        
        """
        
        # If Mh is None, it triggers use of _tab_sfr, which spans all
        # halo masses in self.halos.M
        if Mh is None:
            k = np.argmin(np.abs(z - self.halos.z))
            if abs(z - self.halos.z[k]) < ztol:
                return self._tab_sfr[k]
            else:
                Mh = self.halos.M
        else:
            
            # Create interpolant to be self-consistent
            # with _tab_sfr. Note that this is slower than it needs to be
            # in cases where we just want to know the SFR at a few redshifts
            # and/or halo masses. But, we're rarely doing such things.
            if not hasattr(self, '_spline_sfr'):
                self._spline_sfr = RectBivariateSpline(self.halos.z, 
                    self.halos.M, self._tab_sfr)
            
            return self._spline_sfr(z, Mh).squeeze()
            
        #return self.cosm.fbar_over_fcdm * self.MAR(z, Mh) * self.eta(z) \
        #    * self.SFE(z=z, Mh=Mh)

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
            
    def StellarMass(self, z, Mh):
        zform, data = self.scaling_relations_sorted(z=z)
        return np.interp(Mh, data['Mh'], data['Ms'])
    
    def MetalMass(self, z, Mh):
        zform, data = self.scaling_relations_sorted(z=z)
        return np.interp(Mh, data['Mh'], data['MZ'])

    def GasMass(self, z, Mh):
        zform, data = self.scaling_relations_sorted(z=z)
        return np.interp(Mh, data['Mh'], data['Mg'])
        
    def StellarMassFunction(self, z, Mh):
        Marr, phi = self.SMF(z)
        return np.interp(Mh, Marr, phi)
        
    def CumulativeSurfaceDensity(self, z, dz=1.):
        """
        Return projected surface density of galaxies in `dz` shell.
        """
        
        mags, phi = self.phi_of_M(z=z)
        
        # Still intrinsic magnitudes, shift to apparent
        Mobs = self.dust.Mobs(z, mags)
        
        
        

    def SMF(self, z):
        if not hasattr(self, '_phi_of_Mst'):
            self._phi_of_Mst = {}
        else:
            if z in self._phi_of_Mst:
                return self._phi_of_Mst[z]

        zform, data = self.scaling_relations_sorted(z)

        Mh = data['Mh']
        Ms = data['Ms']
        
        dndm_func = interp1d(self.halos.z, self.halos.dndm[:,:-1], axis=0)
        dndm_z = dndm_func(z)

        # Interpolate dndm to same Mh grid as SAM
        dndm_sam = np.interp(Mh, self.halos.M[0:-1], dndm_z)

        dndm = dndm_sam * self.focc(z=z, Mh=Mh)
        dMh_dMs = np.diff(Mh) / np.diff(Ms)
                
        dMh_dlogMs = dMh_dMs * Ms[0:-1]
        
        # Only return stuff above Mmin
        Mmin = np.interp(z, self.halos.z, self._tab_Mmin)
        if self.pf['pop_Tmax'] is None:
            Mmax = self.pf['pop_lf_Mmax']
        else:
            Mmax = np.interp(z, self.halos.z, self._tab_Mmax)
        
        i_min = np.argmin(np.abs(Mmin - self.halos.M))
        i_max = np.argmin(np.abs(Mmax - self.halos.M))

        phi_of_Ms = dndm[0:-1] * dMh_dlogMs

        above_Mmin = Mh >= Mmin
        below_Mmax = Mh <= Mmax
        ok = np.logical_and(above_Mmin, below_Mmax)[0:-1]
        mask = self.mask = np.logical_not(ok)

        mass = np.ma.array(Ms[:-1], mask=mask)
        phi = np.ma.array(phi_of_Ms, mask=mask, fill_value=tiny_phi)

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

            ok = phi.mask == False
            
            if ok.sum() == 0:
                return -np.inf

            # Setup interpolant
            interp = interp1d(x_phi[ok], np.log10(phi[ok]), kind='linear',
                bounds_error=False, fill_value=-np.inf)

            phi_of_x = 10**interp(x)
        else:

            x_phi, phi = self.phi_of_L(z)

            ok = phi.mask == False
            
            if ok.sum() == 0:
                return -np.inf

            # Setup interpolant
            interp = interp1d(np.log10(x_phi[ok]), np.log10(phi[ok]), kind='linear',
                bounds_error=False, fill_value=-np.inf)
            
            phi_of_x = 10**interp(np.log10(x))

        return phi_of_x

    def Lh(self, z):
        """
        This is the rest-frame UV band in which the LF is measured.
        
        NOT generally use-able!!!
        
        """
        return self.SFR(z) * self.L1600_per_sfr(z=z, Mh=self.halos.M)

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

        fobsc = (1. - self.fobsc(z=z, Mh=self.halos.M))
        
        Lh = self.Lh(z)
        
        # Means obscuration refers to fractional dimming of individual 
        # objects
        if not self.pf['pop_fobsc_by_num']:
            Lh *= fobsc
        
        logL_Lh = np.log(Lh)
        
        dndm_func = interp1d(self.halos.z, self.halos.dndm[:,:-1], axis=0)

        dndm = dndm_func(z) * self.focc(z=z, Mh=self.halos.M[0:-1])
        
        # In this case, obscuration means fraction of objects you don't see
        # in the UV.
        if self.pf['pop_fobsc_by_num']:
            dndm *= fobsc[0:-1]
            
        dMh_dLh = np.diff(self.halos.M) / np.diff(Lh)
                
        dMh_dlogLh = dMh_dLh * Lh[0:-1]
        
        # Only return stuff above Mmin
        Mmin = np.interp(z, self.halos.z, self._tab_Mmin)
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
        phi = np.ma.array(phi_of_L, mask=mask, fill_value=tiny_phi)

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

    def MUV(self, z, Mh):
        Lh = np.interp(Mh, self.halos.M, self.Lh(z))
        MAB = self.magsys.L_to_MAB(Lh, z=z)
        return MAB

    def MUV_max(self, z):
        """
        Compute the magnitude corresponding to the Tmin threshold.
        """   

        return self.MUV(z, Mmin)

    def Mh_of_MUV(self, z, MUV):
        
        # MAB corresponds to self.halos.M
        MAB, phi = self.phi_of_M(z)
        ok = MAB.mask == 0
        
        if ok.sum() == 0:
            return 0.0

        return np.interp(MUV, MAB[ok][-1::-1], self.halos.M[1:-1][ok][-1::-1])
        
    def MUV_at_peak_SFE(self, z):
        """
        Return the UV magnitude of a halo forming stars at peak efficiency.
        """
        
        slope = lambda M: self.gamma_sfe(z, M)
        
        low = fsolve(slope, 1e11)
        
        Mpeak = low[0]
        
        MAB, phi = self.phi_of_M(z)
        MUV_peak = np.interp(Mpeak, self.halos.M[1:-1], MAB)
        
        return MUV_peak
        
    @property
    def _tab_Mmax_active(self):
        """ most massive star-forming halo. """
        if not hasattr(self, '_tab_Mmax_active_'):
            self._tab_Mmax_active_ = np.zeros_like(self.halos.z)
            for i, z in enumerate(self.halos.z):
                lim = self.pf['pop_fstar_negligible']
                fstar_max = self._tab_fstar[i].max()
                immsfh = np.argmin(np.abs(self._tab_fstar[i] - fstar_max * lim))
                self._tab_Mmax_active_[i] = self.halos.M[immsfh]
        return self._tab_Mmax_active_
    
    @property
    def Mmax_active(self):
        if not hasattr(self, '_Mmax_active_'):
            self._Mmax_active_ = \
                lambda z: np.interp(z, self.halos.z, self._tab_Mmax_active)
        return self._Mmax_active_
        
    def dMmin_dt(self, z):
        """ Solar masses per year. """
        return -1. * derivative(self.Mmin, z) * s_per_yr / self.cosm.dtdz(z)

    @property
    def M_atom(self):
        if not hasattr(self, '_Matom'):
            Mvir = lambda z: self.halos.VirialMass(1e4, z, mu=self.pf['mu'])
            self._Matom = np.array(list(map(Mvir, self.halos.z)))
        return self._Matom    

    @property
    def Mmin(self):
        if not hasattr(self, '_Mmin'):  
            self._Mmin = lambda z: \
                np.interp(z, self.halos.z, self._tab_Mmin)

        return self._Mmin

    @Mmin.setter
    def Mmin(self, value):
        if ismethod(value):
            self._Mmin = value
        else:
            self._tab_Mmin = value
            self._Mmin = lambda z: np.interp(z, self.halos.z, self._tab_Mmin)
    
    def Mmax(self, z):
        # Doesn't have a setter because of how we do things in Composite.
        # Long story.
        return np.interp(z, self.halos.z, self._tab_Mmax)
    
    @property
    def _tab_logMmin(self):
        if not hasattr(self, '_tab_logMmin_'):
            self._tab_logMmin_ = np.log(self._tab_Mmin)
        return self._tab_logMmin_
        
    @property
    def _tab_logMmax(self):
        if not hasattr(self, '_tab_logMmax_'):
            self._tab_logMmax_ = np.log(self._tab_Mmax)
        return self._tab_logMmax_    
    
    @property
    def _tab_Mmin(self):
        if not hasattr(self, '_tab_Mmin_'):
            # First, compute threshold mass vs. redshift
            if self.pf['feedback_LW_guesses'] is not None:
                guess = self._guess_Mmin()
                if guess is not None:
                    self._tab_Mmin = guess
                    return self._tab_Mmin_
            
            if self.pf['pop_Mmin'] is not None:
                if ismethod(self.pf['pop_Mmin']) or \
                   type(self.pf['pop_Mmin']) == FunctionType:
                    self._tab_Mmin_ = \
                        np.array(list(map(self.pf['pop_Mmin'], self.halos.z)))
                elif type(self.pf['pop_Mmin']) is np.ndarray:
                    self._tab_Mmin_ = self.pf['pop_Mmin']
                    assert self._tab_Mmin.size == self.halos.z.size
                else:    
                    self._tab_Mmin_ = self.pf['pop_Mmin'] \
                        * np.ones(self.halos.Nz)
            else:
                Mvir = lambda z: self.halos.VirialMass(self.pf['pop_Tmin'],
                    z, mu=self.pf['mu'])
                self._tab_Mmin_ = np.array(list(map(Mvir, self.halos.z)))
                
            self._tab_Mmin_ = self._apply_lim(self._tab_Mmin_, 'min')
                
        return self._tab_Mmin_
    
    @_tab_Mmin.setter
    def _tab_Mmin(self, value):
        if ismethod(value):
            self.Mmin = value
            self._tab_Mmin_ = np.array(list(map(value, self.halos.z)), dtype=float)
        elif type(value) in [int, float, np.float64]:    
            self._tab_Mmin_ = value * np.ones(self.halos.Nz) 
        else:
            self._tab_Mmin_ = value
            
        self._tab_Mmin_ = self._apply_lim(self._tab_Mmin_, s='min')
            
    @property    
    def _tab_Mmin_floor(self):
        if not hasattr(self, '_tab_Mmin_floor_'):
            self._tab_Mmin_floor_ = self.halos.Mmin_floor(self.halos.z)
        return self._tab_Mmin_floor_
            
    def _apply_lim(self, arr, s='min', zarr=None):
        out = None

        if zarr is None:
            zarr = self.halos.z

        # Might need these if Mmin is being set dynamically
        if self.pf['pop_M{!s}_ceil'.format(s)] is not None:
            out = np.minimum(arr, self.pf['pop_M{!s}_ceil'.format(s)])
        if self.pf['pop_M{!s}_floor'.format(s)] is not None:
            out = np.maximum(arr, self.pf['pop_M{!s}_floor'.format(s)])
        if self.pf['pop_T{!s}_ceil'.format(s)] is not None:
            _f = lambda z: self.halos.VirialMass(self.pf['pop_T{!s}_ceil'.format(s)], 
                z, mu=self.pf['mu'])
            _MofT = np.array(list(map(_f, zarr)))
            out = np.minimum(arr, _MofT)
        if self.pf['pop_T{!s}_floor'.format(s)] is not None:
            _f = lambda z: self.halos.VirialMass(self.pf['pop_T{!s}_floor'.format(s)], 
                z, mu=self.pf['mu'])
            _MofT = np.array(list(map(_f, zarr)))
            out = np.maximum(arr, _MofT)

        if out is None:
            out = arr.copy()

        if s == 'min':
            out = np.maximum(out, self._tab_Mmin_floor)
                
        return out
        
    @property
    def _done_setting_Mmax(self):
        if not hasattr(self, '_done_setting_Mmax_'):
            self._done_setting_Mmax_ = False
        return self._done_setting_Mmax_
    
    @_done_setting_Mmax.setter
    def _done_setting_Mmax(self, value):
        self._done_setting_Mmax_ = value
        
    @property
    def _tab_Mmax(self):
        if not hasattr(self, '_tab_Mmax_'):
                                    
            # First, compute threshold mass vs. redshift
            t_limit = self.pf['pop_time_limit']
            m_limit = self.pf['pop_mass_limit'] 
            a_limit = self.pf['pop_abun_limit'] 
            e_limit = self.pf['pop_bind_limit']
            T_limit = self.pf['pop_temp_limit']
            
            if t_limit == 0:
                t_limit = None
            if e_limit == 0:
                e_limit = None
            
            if (t_limit is not None) or (m_limit is not None) or \
               (e_limit is not None) or (T_limit is not None) or (a_limit is not None):
               
                M0x = self.pf['pop_initial_Mh']
                if (M0x == 0) or (M0x == 1):
                    zform, zfin, Mfin, raw = self.MassAfter()
                    new_data = self._sort_sam(self.pf['initial_redshift'], 
                        zform, raw, sort_by='form')
                else:
                    zform, zfin, Mfin, raw = self.MassAfter(M0=M0x)
                    new_data = self._sort_sam(self.pf['initial_redshift'], 
                        zform, raw, sort_by='form') 
                                                        
                # This is the redshift at which the first star-forming halo,
                # formed at (zi, M0), transitions to PopII.
                zmax = max(zfin)
                
                # This is the mass trajectory of a halo that forms at
                # initial_redshift with initial mass pop_initial_Mh 
                # (in units of Mmin, defaults to 1).
                Moft_zi = lambda z: np.interp(z, zform, new_data['Mh'])
                
                # For each redshift, determine Mmax.
                Mmax = np.zeros_like(self.halos.z)
                for i, z in enumerate(self.halos.z):
                    
                    # If we've specified a maximum initial mass halo, and
                    # we're at a redshift before that halo hits its limit.
                    # Or, we're using a time-limited model.
                    if ((M0x > 0) and (z > zmax)):
                        Mmax[i] = Moft_zi(z)
                    else:
                        Mmax[i] = 10**np.interp(z, zfin, np.log10(Mfin))
                        
                self._tab_Mmax_ = Mmax
                
                self._done_setting_Mmax = True

            elif self.pf['pop_Mmax'] is not None:
                if type(self.pf['pop_Mmax']) is FunctionType:
                    self._tab_Mmax_ = np.array(list(map(self.pf['pop_Mmax'], self.halos.z)))
                elif type(self.pf['pop_Mmax']) is tuple:
                    extra = self.pf['pop_Mmax'][0]
                    assert self.pf['pop_Mmax'][1] == 'Mmin'

                    if type(extra) is FunctionType:
                        self._tab_Mmax_ = np.array(list(map(extra, self.halos.z))) \
                            * self._tab_Mmin
                    else:
                        self._tab_Mmax_ = extra * self._tab_Mmin
                else:    
                    self._tab_Mmax_ = self.pf['pop_Mmax'] * np.ones(self.halos.Nz)
            elif self.pf['pop_Tmax'] is not None:
                Mvir = lambda z: self.halos.VirialMass(self.pf['pop_Tmax'], 
                    z, mu=self.pf['mu'])
                self._tab_Mmax_ = np.array(list(map(Mvir, self.halos.z)))
            else:
                # A suitably large number for (I think) any purpose
                self._tab_Mmax_ = 1e18 * np.ones_like(self.halos.z)
    
            self._tab_Mmax_ = self._apply_lim(self._tab_Mmax_, s='max')
            self._tab_Mmax_ = np.maximum(self._tab_Mmax_, self._tab_Mmin)
    
        return self._tab_Mmax_
    
    @_tab_Mmax.setter
    def _tab_Mmax(self, value):
        if type(value) in [int, float, np.float64]:    
            self._tab_Mmax_ = value * np.ones(self.halos.Nz) 
        else:
            self._tab_Mmax_ = value
        
    @property
    def _tab_sfr(self):
        """
        SFR as a function of redshift and halo mass.

            ..note:: Units are Msun/yr.
            
        This does NOT set the SFR to zero in halos with M < Mmin or M > Mmax!    

        """
        if not hasattr(self, '_tab_sfr_'):            
            self._tab_sfr_ = np.zeros([self.halos.Nz, self.halos.Nm])

            for i, z in enumerate(self.halos.z):
                
                if z > self.zform:
                    continue
                #if (z < self.zdead) or (z < self.pf['final_redshift']):
                #    break
                # Should be a little careful here: need to go one or two
                # steps past edge to avoid interpolation problems in SFRD.

                # SF fueld by accretion onto halos already above threshold
                if self.pf['pop_sfr_above_threshold']:

                    if self.pf['pop_sfr_model'] == 'sfr-func':
                        self._tab_sfr_[i] = self.sfr(z=z, Mh=self.halos.M)
                    else:                            
                        self._tab_sfr_[i] = self._tab_eta[i] \
                            * self.cosm.fbar_over_fcdm \
                            * self._tab_MAR[i] * self._tab_fstar[i]

                    # zero-out star-formation in halos below our threshold
                    #ok = self.halos.M >= self._tab_Mmin[i]
                    #self._tab_sfr_[i] *= ok
                    ## zero-out star-formation in halos above our threshold
                    #ok = self.halos.M <= self._tab_Mmax[i]
                    #self._tab_sfr_[i] *= ok

        return self._tab_sfr_

    @property
    def SFRD_at_threshold(self):
        if not hasattr(self, '_SFRD_at_threshold'):
            self._SFRD_at_threshold = \
                lambda z: np.interp(z, self.halos.z, self._tab_sfrd_at_threshold)
        return self._SFRD_at_threshold
        
    @property
    def _tab_nh_active(self):
        if not hasattr(self, '_tab_nh_active_'):
            self._tab_nh_active_ = np.zeros(self.halos.Nz)

            # Loop from high-z to low-z
            for k, z in enumerate(self.halos.z[-1::-1]):

                i = self.halos.Nz - k - 1
                
                if not self.pf['pop_sfr_above_threshold']:
                    break

                if z > self.zform:
                    continue

                integrand = self.halos.dndlnm[i] \
                    * self.focc(z=z, Mh=self.halos.M)

                # Mmin and Mmax will never be exactly on Mh grid points 
                # so we interpolate to more precisely determine SFRD.    

                c1 = self.halos.M >= self._tab_Mmin[i]
                c2 = self.halos.M <= self._tab_Mmax[i]
                ok = np.logical_and(c1, c2)
                
                if self._tab_Mmin[i] == self._tab_Mmax[i]:
                    self._tab_nh_active_[i] = 0
                    
                    # We 'break' here because once Mmax = Mmin, PopIII
                    # should be gone forever.
                    
                    if z < self.pf['initial_redshift']:
                        break
                    else:
                        continue
                    
                # This can happen if Mmin and Mmax are within the same mass bin,
                # or if they straddle a single mass bin. Need to be a bit careful.
                if ok.sum() in [0, 1]:                                        
                    i1 = np.argmin(np.abs(self.halos.M - self._tab_Mmin[i]))
                    if self.halos.M[i1] > self._tab_Mmin[i]:
                        i1 -= 1
                    i2 = i1 + 1
                    
                    # Trapezoid here we come
                    b = self._tab_logMmax[i] - self._tab_logMmin[i]
                    
                    M1 = self._tab_logMmin[i]
                    M2 = self._tab_logMmax[i]
                    y1 = np.interp(M1, [self.halos.lnM[i1], self.halos.lnM[i2]],
                        [integrand[i1], integrand[i2]])
                    y2 = np.interp(M2, [self.halos.lnM[i1], self.halos.lnM[i2]],
                        [integrand[i1], integrand[i2]])
                    
                    h = abs(y2 - y1)
                    
                    self._tab_nh_active_[i] = 0.5 * b * (y1 + y2)

                    continue

                # Otherwise, do the normal thing.
                #if ok.sum() == 1:
                #    iok = [np.argwhere(ok).squeeze()]
                #else:
                iok = np.argwhere(ok).squeeze()

                Mlo1 = min(iok)
                Mhi1 = max(iok)
                Mlo2 = Mlo1 - 1
                Mhi2 = Mhi1 + 1

                # This happens if Mmin and Mmax straddle a mass bin
                if ok.sum() == 1:
                    raise ValueError('help')
                    b = self.halos.lnM[Mlo1+1] - self.halos.lnM[Mlo1]
                    #h = abs(integrand[Mlo1+1] - integrand[Mlo1])
                    #b = self.halos.lnM[Mlo1] - self.self.halos.lnM[Mlo1+1]
                    
                    M1 = self._tab_logMmin[i]
                    M2 = self._tab_logMmax[i]
                    y1 = np.interp(M1, [self.halos.logM[i1], self.halos.logM[i2]],
                        [integrand[i1], integrand[i2]])
                    y2 = np.interp(M2, [self.halos.logM[i1], self.halos.logM[i2]],
                        [integrand[i1], integrand[i2]])
                    
                    h = abs(y2 - y1)
                    
                    tot = 0.5 * b * h
                else:
                    # This is essentially an integral from Mlo1 to Mhi1
                    tot = np.trapz(integrand[ok], x=self.halos.lnM[ok])
                                
                integ_lo = np.trapz(integrand[Mlo2:Mhi1+1], 
                    x=self.halos.lnM[Mlo2:Mhi1+1])
                                
                # Interpolating over lower integral bound
                sfrd_lo = np.interp(self._tab_logMmin[i], 
                    [self.halos.lnM[Mlo2], self.halos.lnM[Mlo1]], 
                    [integ_lo, tot]) - tot
                                                
                if Mhi2 >= self.halos.Nm:    
                    sfrd_hi = 0.0
                else:
                    integ_hi = np.trapz(integrand[Mlo1:Mhi2+1], 
                        x=self.halos.lnM[Mlo1:Mhi2+1])    
                    sfrd_hi = np.interp(self._tab_logMmax[i], 
                        [self.halos.lnM[Mhi1], self.halos.lnM[Mhi2]], 
                        [tot, integ_hi]) - tot
                
                self._tab_nh_active_[i] = tot + sfrd_lo + sfrd_hi

            self._tab_nh_active_ *= 1. / cm_per_mpc**3

            #if self.pf['pop_sfr_cross_threshold']:
            #    self._tab_sfrd_total_ += self._tab_sfrd_at_threshold

        return self._tab_nh_active_
        
    #def _dndlnm_weighted_integral(self, iz, quantity):
    #    
    #    z = self.halos.z[iz]
    #    Mmin = self._tab_Mmin[iz]
    #    Mmax = self._tab_Mmax[iz]
    #    logMmin = self._tab_logMmin[iz]
    #    logMmax = self._tab_logMmax[iz]
    #    
    #    focc = self.focc(z=z, Mh=self.halos.M) 
    #    integrand = quantity * self.halos.dndlnm[iz] * focc
    #
    #    # Mmin and Mmax will never be exactly on Mh grid points 
    #    # so we interpolate to more precisely determine SFRD.    
    #    c1 = self.halos.M >= self._tab_Mmin[iz]
    #    c2 = self.halos.M <= self._tab_Mmax[iz]
    #    ok = np.logical_and(c1, c2)
    #    
    #    #
    #    if self._tab_Mmin[iz] == self._tab_Mmax[iz]:
    #        integral = 0
    #        
    #        # We 'break' here because once Mmax = Mmin, PopIII
    #        # should be gone forever.
    #        
    #       #if z < self.pf['initial_redshift']:
    #       #    break
    #       #else:
    #       #    continue
    #        
    #    # This can happen if Mmin and Mmax are within the same mass bin,
    #    # or if they straddle a single mass bin. Need to be a bit careful.
    #    if ok.sum() in [0, 1]:                       
    #        
    #        # Find grid point just lower than Mmin
    #        i1 = np.argmin(np.abs(self.halos.M - Mmin))
    #        if self.halos.M[i1] > Mmin:
    #            i1 -= 1
    #            
    #        # And upper grid point
    #        i2 = i1 + 1
    #        
    #        # Let's re-compute the integral as a function of Mmin and
    #        # then interpolate to find answer more precisely.
    #        
    #        integral = []
    #        for k in range(5, -1, -1):
    #            
    #            result = np.trapz(integrand[i1-k:i2+1], 
    #                x=self.halos.lnM[Mlo2:i2+1])
    #        
    #        
    #        
    #        
    #        
    #                         
    #        i1 = np.argmin(np.abs(self.halos.M - Mmin))
    #        if self.halos.M[i1] > Mmin:
    #            i1 -= 1
    #        i2 = i1 + 1
    #        
    #        # Trapezoid here we come
    #        b = logMmax - logMmin
    #        
    #        M1 = logMmin
    #        M2 = logMmax
    #        y1 = np.interp(logMmin, [self.halos.lnM[i1], self.halos.lnM[i2]],
    #            [integrand[i1], integrand[i2]])
    #        y2 = np.interp(logMmax, [self.halos.lnM[i1], self.halos.lnM[i2]],
    #            [integrand[i1], integrand[i2]])
    #        
    #        h = abs(y2 - y1)
    #        
    #        self._tab_sfrd_total_[i] = 0.5 * b * h
    #
    #        continue
    #
    #    iok = np.argwhere(ok).squeeze()
    #
    #    Mlo1 = min(iok)
    #    Mhi1 = max(iok)
    #    Mlo2 = Mlo1 - 1
    #    Mhi2 = Mhi1 + 1
    #
    #    # This is essentially an integral from Mlo1 to Mhi1
    #    tot = np.trapz(integrand[ok], x=self.halos.lnM[ok])
    #                    
    #    integ_lo = np.trapz(integrand[Mlo2:Mhi1+1], 
    #        x=self.halos.lnM[Mlo2:Mhi1+1])
    #                    
    #    # Interpolating over lower integral bound
    #    sfrd_lo = np.interp(self._tab_logMmin[i], 
    #        [self.halos.lnM[Mlo2], self.halos.lnM[Mlo1]], 
    #        [integ_lo, tot]) - tot
    #                                    
    #    if Mhi2 >= self.halos.Nm:    
    #        sfrd_hi = 0.0
    #    else:
    #        integ_hi = np.trapz(integrand[Mlo1:Mhi2+1], 
    #            x=self.halos.lnM[Mlo1:Mhi2+1])
    #        sfrd_hi = np.interp(self._tab_logMmax[i], 
    #            [self.halos.lnM[Mhi1], self.halos.lnM[Mhi2]], 
    #            [tot, integ_hi]) - tot
    #    
    #    self._tab_sfrd_total_[i] = tot + sfrd_lo + sfrd_hi
        
    @property
    def _tab_sfrd_total(self):
        """
        SFRD as a function of redshift.
    
            ..note:: Units are g/s/cm^3 (comoving).

        """

        if not hasattr(self, '_tab_sfrd_total_'):
            self._tab_sfrd_total_ = np.zeros(self.halos.Nz)

            # Loop from high-z to low-z
            for k, z in enumerate(self.halos.z[-1::-1]):

                i = self.halos.Nz - k - 1
                
                if not self.pf['pop_sfr_above_threshold']:
                    break

                if z > self.zform:
                    continue

                integrand = self._tab_sfr[i] * self.halos.dndlnm[i] \
                    * self.focc(z=z, Mh=self.halos.M)

                # Mmin and Mmax will never be exactly on Mh grid points 
                # so we interpolate to more precisely determine SFRD.    

                c1 = self.halos.M >= self._tab_Mmin[i]
                c2 = self.halos.M <= self._tab_Mmax[i]
                ok = np.logical_and(c1, c2)
                
                if self._tab_Mmin[i] == self._tab_Mmax[i]:
                    self._tab_sfrd_total_[i] = 0
                    
                    # We 'break' here because once Mmax = Mmin, PopIII
                    # should be gone forever.
                    
                    if z < self.pf['initial_redshift']:
                        break
                    else:
                        continue
                    
                # This can happen if Mmin and Mmax are within the same mass bin,
                # or if they straddle a single mass bin. Need to be a bit careful.
                if ok.sum() in [0, 1]:                                        
                    i1 = np.argmin(np.abs(self.halos.M - self._tab_Mmin[i]))
                    if self.halos.M[i1] > self._tab_Mmin[i]:
                        i1 -= 1
                    i2 = i1 + 1
                    
                    # Trapezoid here we come
                    b = self._tab_logMmax[i] - self._tab_logMmin[i]
                    
                    M1 = self._tab_logMmin[i]
                    M2 = self._tab_logMmax[i]
                    y1 = np.interp(M1, [self.halos.lnM[i1], self.halos.lnM[i2]],
                        [integrand[i1], integrand[i2]])
                    y2 = np.interp(M2, [self.halos.lnM[i1], self.halos.lnM[i2]],
                        [integrand[i1], integrand[i2]])
                    
                    h = abs(y2 - y1)
                    
                    self._tab_sfrd_total_[i] = 0.5 * b * (y1 + y2)

                    #print z, ok.sum()

                    continue

                iok = np.argwhere(ok).squeeze()

                Mlo1 = min(iok)
                Mhi1 = max(iok)
                Mlo2 = Mlo1 - 1
                Mhi2 = Mhi1 + 1

                # This is essentially an integral from Mlo1 to Mhi1
                tot = np.trapz(integrand[ok], x=self.halos.lnM[ok])
                                
                integ_lo = np.trapz(integrand[Mlo2:Mhi1+1], 
                    x=self.halos.lnM[Mlo2:Mhi1+1])
                                
                # Interpolating over lower integral bound
                sfrd_lo = np.interp(self._tab_logMmin[i], 
                    [self.halos.lnM[Mlo2], self.halos.lnM[Mlo1]], 
                    [integ_lo, tot]) - tot
                                                
                if Mhi2 >= self.halos.Nm:    
                    sfrd_hi = 0.0
                else:
                    integ_hi = np.trapz(integrand[Mlo1:Mhi2+1], 
                        x=self.halos.lnM[Mlo1:Mhi2+1])
                    sfrd_hi = np.interp(self._tab_logMmax[i], 
                        [self.halos.lnM[Mhi1], self.halos.lnM[Mhi2]], 
                        [tot, integ_hi]) - tot
                
                self._tab_sfrd_total_[i] = tot + sfrd_lo + sfrd_hi

            self._tab_sfrd_total_ *= g_per_msun / s_per_yr / cm_per_mpc**3

            if self.pf['pop_sfr_cross_threshold']:
                self._tab_sfrd_total_ += self._tab_sfrd_at_threshold

        return self._tab_sfrd_total_
    
    def SFRD_above_MUV(self, z, MUV=-17):
    
        if not hasattr(self, '_sfrd_above_MUV_tab'):
            self._sfrd_above_MUV_tab = {}
            
        if type(MUV) == np.ndarray:
            res = []
            for limit in MUV:
                if (z, limit) in self._sfrd_above_MUV_tab:
                    res.append(self._sfrd_above_MUV_tab[(z, limit)])
                    continue
                
                self._sfrd_above_MUV_tab[(z, limit)] = \
                    self.SFRD_within(z, limit, None, is_mag=True)
                
                res.append(self._sfrd_above_MUV_tab[(z, limit)])
    
            return np.array(res)    
    
        else:
            
            raise NotImplemented('i think this is broken')
            
            if (z, MUV) in self._sfrd_above_MUV_tab:
                return self._sfrd_above_MUV_tab[(z, MUV)]
                
            self._sfrd_above_MUV_tab[(z, MUV)] = \
                self.SFRD_within(z, MUV, None, is_mag=True)
    
            return self._sfrd_above_MUV_tab[(z, MUV)]
    
    def SFRD_within(self, z, Mlo, Mhi=None, is_mag=False):
        """
        Compute SFRD within given mass range, [Mlo, Mhi].
        """
        
        if not hasattr(self, '_sfrd_within'):
            self._sfrd_within = {}
            
        if (Mlo, Mhi) in self._sfrd_within.keys():
            return self._sfrd_within[(Mlo, Mhi)](z)
        
        _sfrd_tab = np.zeros(self.halos.Nz)

        for i, zz in enumerate(self.halos.z):
            
            if not self.pf['pop_sfr_above_threshold']:
                break

            if zz > self.zform:
                continue

            integrand = self._tab_sfr[i] * self.halos.dndlnm[i] \
                * self.focc(z=zz, Mh=self.halos.M)
            
            if is_mag:
                _Mlo = self.Mh_of_MUV(zz, Mlo)
            # Crudely for now
            elif Mlo == 'Mmin':
                _Mlo = self.Mmin(zz)
            else:
                _Mlo = Mlo
            
            ilo = np.argmin(np.abs(self.halos.M - _Mlo))
                    
            if Mhi is None:
                ihi = self.halos.Nm
            else:
                ihi = np.argmin(np.abs(self.halos.M - Mhi))
            
            _sfrd_tab[i] = np.trapz(integrand[ilo:ihi+1], 
                x=self.halos.lnM[ilo:ihi+1])
                                        
        if self.pf['pop_sfr_cross_threshold'] and type(Mlo) is str:
            if (Mlo == 'Mmin'):
                _sfrd_tab += self._tab_sfrd_at_threshold_
                            
        _sfrd_tab *= g_per_msun / s_per_yr / cm_per_mpc**3

        _sfrd_func = lambda zz: np.interp(zz, self.halos.z, _sfrd_tab)
        
        if type(Mlo) != np.ndarray:
            self._sfrd_within[(Mlo, Mhi)] = _sfrd_func
        
        return _sfrd_func(z)

    @property
    def LLyC_tab(self):
        """
        Number of LyC photons emitted per unit SFR in halos of mass M.
        """
        if not hasattr(self, '_LLyC_tab'):
            M = self.halos.M
            fesc = self.fesc(None, M)
            
            dnu = (24.6 - 13.6) / ev_per_hz

            Nion_per_L1600 = self.Nion(None, M) / (1. / dnu)
            
            self._LLyC_tab = np.zeros([self.halos.Nz, self.halos.Nm])
            
            for i, z in enumerate(self.halos.z):
                self._LLyC_tab[i] = self.L1600_tab[i] * Nion_per_L1600 \
                    * fesc
            
                mask = self.halos.M >= self._tab_Mmin[i]
                self._LLyC_tab[i] *= mask
            
        return self._LLyC_tab
                
    @property
    def LLW_tab(self):
        if not hasattr(self, '_LLW_tab'):
            M = self.halos.M
                
            dnu = (13.6 - 10.2) / ev_per_hz
            #nrg_per_phot = 25. * erg_per_ev
    
            Nlw_per_L1600 = self.Nlw(z=None, M=M) / (1. / dnu)
            fesc_LW = self.fesc_LW(z=None, M=M)
    
            self._LLW_tab = np.zeros([self.halos.Nz, self.halos.Nm])
    
            for i, z in enumerate(self.halos.z):
                self._LLW_tab[i] = self.L1600_tab[i] * Nlw_per_L1600
    
                mask = self.halos.M >= self._tab_Mmin[i]
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
            
            assert self.pf['pop_sfr'] is None

            if self.pf['pop_calib_L1600'] is not None:
                boost = self.pf['pop_calib_L1600'] / self.L1600_per_sfr()
            else:
                boost = 1.

            if self.pf['pop_mlf'] is not None:
                if type(self.pf['pop_mlf']) in [float, np.float64]:
                    # Note that fshock is really fcool
                    self._fstar = lambda **kwargs: boost * self.fshock(**kwargs) \
                        / ((1. / self.pf['pop_fstar_max']) + self.pf['pop_mlf'])
                elif self.pf['pop_mlf'][0:2] == 'pq':
                    pars = get_pq_pars(self.pf['pop_mlf'], self.pf)
                    Mmin = lambda z: np.interp(z, self.halos.z, self._tab_Mmin)
                
                    self._mlf_inst = ParameterizedQuantity({'pop_Mmin': Mmin}, 
                        self.pf, **pars)
                
                    self._update_pq_registry('mlf', self._mlf_inst)    
                    self._fstar = \
                        lambda **kwargs: boost * self.fshock(**kwargs) \
                            / ((1. / self.pf['pop_fstar_max']) + self._mlf_inst(**kwargs))
            
            elif self.pf['pop_fstar'] is not None:
                if type(self.pf['pop_fstar']) in [float, np.float64]:
                    self._fstar = lambda **kwargs: self.pf['pop_fstar'] * boost    
                
                elif self.pf['pop_fstar'][0:2] == 'pq':
                    pars = get_pq_pars(self.pf['pop_fstar'], self.pf)
                                        
                    Mmin = lambda z: np.interp(z, self.halos.z, self._tab_Mmin)
                    self._fstar_inst = ParameterizedQuantity({'pop_Mmin': Mmin}, 
                        self.pf, **pars)
                    
                    self._update_pq_registry('fstar', self._fstar_inst)    
                    
                    self._fstar = \
                        lambda **kwargs: self._fstar_inst.__call__(**kwargs) \
                            * boost
            
                
            else:
                raise ValueError('Unrecognized data type for pop_fstar!')  

        return self._fstar
        
    @fstar.setter
    def fstar(self, value):
        self._fstar = value  
                
    def gamma_sfe(self, z, Mh):
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
        
        fst = lambda MM: self.SFE(z=z, Mh=MM)
        
        return derivative(fst, Mh, dx=1e6) * Mh / fst(Mh)
            
    def slope_lf(self, z, mag):
        logphi = lambda logL: np.log10(self.LuminosityFunction(z, 10**logL, mags=False))
    
        Mdc = mag - self.dust.AUV(z, mag)
        L = self.magsys.MAB_to_L(mag=Mdc, z=z)
    
        return derivative(logphi, np.log10(L), dx=0.1)
    
    #def slope_smf(self, z, M):
    #    logphi = lambda logM: np.log10(self.StellarMassFunction(z, 10**logM))
    #
    #    
    #
    #    return derivative(logphi, np.log10(L), dx=0.1)    
    
    @property
    def _tab_fstar(self):
        if not hasattr(self, '_tab_fstar_'):
            self._tab_fstar_ = np.zeros_like(self.halos.dndm)
    
            for i, z in enumerate(self.halos.z):    
                if i > 0 and self.constant_SFE:
                    self._tab_fstar_[i,:] = self._tab_fstar_[0]
                    continue
                    
                self._tab_fstar_[i,:] = self.SFE(z=z, Mh=self.halos.M)
    
        return self._tab_fstar_
    
    def _SAM(self, z, y):
        if self.pf['pop_sam_nz'] == 1:
            return self._SAM_1z(z, y)
        elif self.pf['pop_sam_nz'] == 2:
            return self._SAM_2z(z, y)
        else:
            raise NotImplementedError('No SAM with nz={}'.format(\
                self.pf['pop_sam_nz']))
                        
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

        Mh, Mg, Mst, MZ, cMst = y

        kw = {'z':z, 'Mh': Mh, 'Ms': Mst, 'Mg': Mg, 'MZ': MZ,
            'cMst': cMst}

        # Assume that MZ, Mg, and Mstell acquired *not* by smooth inflow
        # is same fraction of accreted mass as fractions in this halo
        # right now
        
        fb = self.cosm.fbar_over_fcdm
        
        # Splitting up the inflow. P = pristine, 
        PIR = -1. * fb * self.MAR(z, Mh) * self.cosm.dtdz(z) / s_per_yr
        NPIR = -1. * fb * self.MDR(z, Mh) * self.cosm.dtdz(z) / s_per_yr
        
        # Measured relative to baryonic inflow
        Mb = fb * Mh
        Zfrac = self.pf['pop_acc_frac_metals'] * (MZ / Mb)
        Sfrac = self.pf['pop_acc_frac_stellar'] * (Mst / Mb)
        Gfrac = self.pf['pop_acc_frac_gas'] * (Mg / Mb)
                
        if self.pf['pop_sfr'] is None:
            fstar = self.SFE(**kw)
            SFR = PIR * fstar
        else:
            fstar = 1e-10
            SFR = -self.sfr(**kw) * self.cosm.dtdz(z) / s_per_yr

        # "Quiet" mass growth
        fsmooth = self.fsmooth(**kw)

        # Eq. 1: halo mass.
        y1p = -1. * self.MGR(z, Mh) * self.cosm.dtdz(z) / s_per_yr

        # Eq. 2: gas mass
        if self.pf['pop_sfr'] is None:
            y2p = PIR * (1. - SFR/PIR) + NPIR * Gfrac
        else:
            y2p = PIR * (1. - fstar) + NPIR * Gfrac

        # Add option of parameterized stifling of gas supply, and
        # ejection of gas.
        
        if self._done_setting_Mmax:
            Mmax = self.Mmax(z)
        else:
            Mmax = np.inf
        
        # Eq. 3: stellar mass
        Mmin = self.Mmin(z)
        if (Mh < Mmin) or (Mh > Mmax):
            y3p = SFR = 0.
        else:
            y3p = SFR * (1. - self.pf['pop_mass_yield']) + NPIR * Sfrac

        # Eq. 4: metal mass -- constant return per unit star formation for now
        y4p = self.pf['pop_mass_yield'] * self.pf['pop_metal_yield'] * SFR \
            * (1. - self.pf['pop_mass_escape']) \
            + NPIR * Zfrac

        if (Mh < Mmin) or (Mh > Mmax):
            y5p = 0.
        else:
            y5p = SFR + NPIR * Sfrac

        # Stuff to add: parameterize metal yield, metal escape, star formation
        # from reservoir? How to deal with Mmin(z)? Initial conditions (from PopIII)?

        results = [y1p, y2p, y3p, y4p, y5p]
        
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
        Mmin = self.Mmin(z)
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
            
            if self.constant_SFR:
                self._constant_SFE = 0
                return self._constant_SFE
            
            self._constant_SFE = 1
            for mass in [1e7, 1e8, 1e9, 1e10, 1e11, 1e12]:
                self._constant_SFE *= self.fstar(z=10, Mh=mass) \
                                   == self.fstar(z=20, Mh=mass)
                                   
            self._constant_SFE = bool(self._constant_SFE)    
                               
        return self._constant_SFE

    @property
    def constant_SFR(self):
        if not hasattr(self, '_constant_SFR'):
            if self.pf['pop_sfr'] is not None:
                self._constant_SFR = 1
            else:
                self._constant_SFR = 0
        return self._constant_SFR

    @property
    def scaling_relations(self):
        if not hasattr(self, '_scaling_relations'):
            if self.constant_SFE:
                self._scaling_relations = self._ScalingRelationsStaticSFE()
            else:
                self._scaling_relations = self._ScalingRelationsGeneralSFE()

        return self._scaling_relations

    def duration(self, zend=6.):
        """
        Calculate the duration of this population, i.e., time it takes to get
        from formation redshift to Mmax.
        """
        
        zform, zfin, Mfin, raw = self.MassAfter(M0=self.pf['pop_initial_Mh'])

        duration = []
        for i, zf in enumerate(zfin):
            duration.append(self.cosm.LookbackTime(zf, zform[i]) / s_per_yr / 1e6)

        duration = np.array(duration)

        # This is not quite what Rick typically plots -- his final masses are
        # those at z=6, and many of those halos will have been forming PopII 
        # stars for awhile.

        # So, compute the final halo mass, including time spent after Mh > Mmax.
        Mend = []
        for i, z in enumerate(zform):
            
            new_data = self._sort_sam(z, zform, raw, sort_by='form')
           
            Mend.append(np.interp(zend, zform, new_data['Mh']))
        
        return zform, zfin, Mfin, duration#, np.array(Mend)

    def MassAfter(self, M0=0):
        """
        Compute the final mass of a halos that begin at Mmin and grow for dt.

        Parameters
        ----------
        dt : int, float
            Growth time [years]

        Returns
        -------
        Array of formation redshifts, final redshifts, and final masses.

        """

        # This loops over a bunch of formation redshifts
        # and computes the trajectories for all SAM fields.
        zarr, data = self._ScalingRelationsGeneralSFE(M0=M0)

        # At this moment, all data is in order of ascending redshift
        # Each element in `data` is 2-D: (zform, zarr)

        # Figure out the final mass (after `dt`) of halos formed at each
        # formation redshift, and the redshift at which they reach that mass
        Mfin = []
        zfin = []
        for k, z in enumerate(zarr):
            # z here is the formation redshift
            new_data = self._sort_sam(z, zarr, data, sort_by='form')
            
            # Redshift when whatever transition-triggering event was reached
            zmax = new_data['zmax']
            zfin.append(zmax)
            
            # Interpolate in mass trajectory to find maximum mass
            # more precisely. Reminder: already interpolated to 
            # find 'zmax' in call to _ScalingRelations
            Mmax = np.interp(zmax, zarr, new_data['Mh'])

            # We just want the final mass (before dt killed SAM)
            Mfin.append(Mmax)

        Mfin = self._apply_lim(Mfin, 'max', zarr)
        
        zfin = np.array(zfin)  
        
        # Check for double-valued-ness
        # Just kludge and take largest value.
        
        zrev = zfin[-1::-1]
        for i, z in enumerate(zrev):
            if i == 0:
                continue
                                
            if z == self.pf['final_redshift']:
                break
        
        Nz = len(zfin)
            
        zfin[0:Nz-i] = self.pf['final_redshift']
        Mfin[0:Nz-i] = max(Mfin)
        
        return zarr, zfin, np.array(Mfin), data
        
    def scaling_relations_sorted(self, z=None):
        """

        """
        if not hasattr(self, '_scaling_relations_sorted'):
            self._scaling_relations_sorted = {}

        if z in self._scaling_relations_sorted:
            return self._scaling_relations_sorted[z]

        zform, data = self.scaling_relations
                        
        # Can't remember what this is all about.
        if self.constant_SFE:
            new_data = {}
            sorter = np.argsort(data['Mh'])
            for key in data.keys():
                if key == 'zmax':
                    continue
                new_data[key] = data[key][sorter]
        else:
            
            assert z is not None
            
            zf = max(float(self.halos.z.min()), self.pf['final_redshift'])

            if self.pf['sam_dz'] is not None:
                zfreq = int(round(self.pf['sam_dz'] / np.diff(self.halos.z)[0], 0))
            else:
                zfreq = 1

            zarr = self.halos.z[self.halos.z >= zf][::zfreq]
                        
            new_data = self._sort_sam(z, zarr, data)            
                          
        self._scaling_relations_sorted[z] = zform, new_data    

        return self._scaling_relations_sorted[z]
        
    def _sort_sam(self, z, zarr, data, sort_by='obs'):
        """
        Take results of a SAM and grab data for a single formation redshift.
        
        Parameters
        ----------
        z : int, float
            Redshift of interest
        zarr : np.ndarray
            Array of all redshifts used in SAM.
        data : dict
            
        
        Returns
        -------
        Dictionary containing trajectories 
        """
                
        # First grab all elements with the right redshift.
        tmp = {}
        k = np.argmin(np.abs(z - zarr))
        for key in data.keys():
            if data[key].ndim == 2:
                if sort_by == 'form':
                    tmp[key] = data[key][k]
                else:
                    tmp[key] = data[key][:,k]
            else:
                tmp[key] = data[key][k]

        # Next, make sure they are in order of increasing halo mass
        if sort_by == 'form':
            new_data = tmp
        else:
            new_data = {}
            sorter = np.argsort(tmp['Mh'])
            for key in tmp.keys():
                if data[key].ndim == 2:
                    new_data[key] = tmp[key][sorter]
                else:
                    new_data[key] = tmp[key]
            
        return new_data
        
    def _ScalingRelationsGeneralSFE(self, M0=0):
        """
        In this case, the formation time of a halo matters.
        
        Returns
        -------
        Dictionary of quantities, each having shape (z, z). 
        The first dimension corresponds to formation time, the second axis
        represents trajectories.
        
        """
        
        
        keys = ['Mh', 'Mg', 'Ms', 'MZ', 'cMs', 'Z', 't']
                
        zf = max(float(self.halos.z.min()), self.pf['final_redshift'])
        zi = min(float(self.halos.z.max()), self.pf['initial_redshift'])
        
        if self.pf['sam_dz'] is not None:
            zfreq = int(round(self.pf['sam_dz'] / np.diff(self.halos.z)[0], 0))
        else:
            zfreq = 1

        in_range = np.logical_and(self.halos.z >= zf, self.halos.z <= zi)
        zarr = self.halos.z[in_range][::zfreq]
        results = {key:np.zeros([zarr.size]*2) for key in keys}
                
        zmax = []                        
        zform = []
                
        for i, z in enumerate(zarr):
            #if (i == 0) or (i == len(zarr) - 1):
            #    zmax.append(zarr[i])
            #    zform.append(z)
            #    continue

            _zarr, _results = self._ScalingRelationsStaticSFE(z0=z, M0=M0)

            #print z, len(_zarr), len(_results['Mh'])
            #print _zarr
            #raw_input('<enter>')

            # Need to splice into the right elements of 2-D array
            for key in keys:
                dat = _results[key].copy()
                k = np.argmin(abs(_zarr.min() - zarr))
                results[key][i,k:k+len(dat)] = dat
                
                # Must keep in mind that different redshifts have different
                # halo mass-gridding effectively

            zform.append(z)
            
            zmax.append(_results['zmax'])

        results['zmax'] = np.array(zmax)

        return np.array(zform), results
        
    def _ScalingRelationsStaticSFE(self, z0=None, M0=0):
        """
        Evolve a halo from initial mass M0 at redshift z0 forward in time.

        Really this should be invoked any time any PQ has 'z' in its vars list.
        
        Parameters
        ----------
        z0 : int, float
            Formation redshift.
        dt : int, float
            Duration to consider [years]
        
        Returns
        -------
        redshifts, halo mass, gas mass, stellar mass, metal mass
        
        If dt is provided, the calculation will be truncated `dt` years
        after `z0`, so the resultant arrays will not have the expected
        number of elements.

        """

        zf = max(float(self.halos.z.min()), self.pf['final_redshift'])

        if self.pf['sam_dz'] is not None:
            dz = self.pf['sam_dz']
            zfreq = int(round(self.pf['sam_dz'] / np.diff(self.halos.z)[0], 0))
        else:
            dz = np.diff(self.halos.z)[0]
            zfreq = 1

        solver = ode(self._SAM).set_integrator('lsoda', nsteps=1e4, 
            atol=self.pf['sam_atol'], rtol=self.pf['sam_rtol'])
            
        has_e_limit = self.pf['pop_bind_limit'] is not None    
        has_T_limit = self.pf['pop_temp_limit'] is not None    
        has_t_limit = self.pf['pop_time_limit'] is not None    
        has_m_limit = self.pf['pop_mass_limit'] is not None
        has_a_limit = self.pf['pop_abun_limit'] is not None
        
        has_t_ceil = self.pf['pop_time_ceil'] is not None
                
        if self.pf['pop_time_limit'] == 0:
            has_t_limit = False
        if self.pf['pop_bind_limit'] == 0:
            has_e_limit = False    
                
        ##
        # Outputs have shape (z, z)
        ##
        
        # Our results don't depend on this, unless SFE depends on z
        if (z0 is None) and (M0 == 0):
            z0 = self.halos.z.max()
            M0 = self._tab_Mmin[-1]
        elif (M0 <= 1):
            M0 = np.interp(z0, self.halos.z, self._tab_Mmin)
        elif (M0 > 1):
            if z0 >= self.pf['initial_redshift']:
                M0 = np.interp(z0, self.halos.z, M0 * self._tab_Mmin)
            else:
                M0 = np.interp(z0, self.halos.z, self._tab_Mmin)

        in_range = np.logical_and(self.halos.z >= zf, self.halos.z <= z0)
        zarr = self.halos.z[in_range][::zfreq]
        Nz = zarr.size

        # Boundary conditions (pristine halo)
        Mg0 = self.cosm.fbar_over_fcdm * M0
        MZ0 = 0.0
        Mst0 = 0.0

        # Initial stellar mass -> 0, initial halo mass -> Mmin
        solver.set_initial_value(np.array([M0, Mg0, Mst0, MZ0, Mst0]), z0)

        # Only used occasionally
        zmax = None
        zmax_t = None
        zmax_m = None
        zmax_a = None
        zmax_T = None
        zmax_e = None
        
        # zmax really means the highest redshift where a certain
        # transition criterion is satisfied.

        Mh_t = []
        Mg_t = []
        Mst_t = []
        cMst_t = []
        metals = []
        lbtime = []
        Ehist = []
        redshifts = []
        for i in range(Nz):
            # In descending order
            redshifts.append(zarr[-1::-1][i])
            Mh_t.append(solver.y[0])
            Mg_t.append(solver.y[1])
            Mst_t.append(solver.y[2])
            metals.append(solver.y[3])
            cMst_t.append(solver.y[4])
            
            z = redshifts[-1]

            lbtime_myr = self.cosm.LookbackTime(z, z0) \
                / s_per_yr / 1e6

            lbtime.append(lbtime_myr)

            # t_ceil is a trump card.
            # For example, in some cases the critical metallicity will never
            # be met due to high inflow rates.                     
            if has_t_limit or has_t_ceil:     
                if has_t_limit:
                    tlim = self.time_limit(z=z, Mh=M0)
                elif has_t_ceil:
                    tlim = self.time_ceil(z=z, Mh=M0)
                  
                if lbtime_myr >= tlim:
                    hit_dt = True

                    lbtime_myr_prev = self.cosm.LookbackTime(redshifts[-2], z0) \
                        / s_per_yr / 1e6

                    zmax_t = np.interp(tlim,
                        [lbtime_myr_prev, lbtime_myr], redshifts[-2:])

            if has_m_limit:
                Mnow = solver.y[2]

                if Mnow >= self.mass_limit(z=z, Mh=M0) and (zmax_m is None):
                    zmax_m = np.interp(self.mass_limit(z=z, Mh=M0), cMst_t[-2:], 
                        redshifts[-2:])

            if has_a_limit and (zmax_a is None):
                                
                # Subtract off metals accrued before crossing Eb limit?
                #if (zmax_e is not None) and self.pf['pop_lose_metals'] and i > 0:
                #    MZ_e = np.interp(zmax_e, redshifts[-1::-1], metals[-1::-1])
                #    Mg_e = np.interp(zmax_e, redshifts[-1::-1], Mg_t[-1::-1])
                #                        
                #    Zpre = (metals[-2] - MZ_e) / solver.y[1]
                #    Znow = (solver.y[3] - MZ_e) / solver.y[1]
                #elif self.pf['pop_lose_metals']:
                #    Zpre = Znow = 0.0
                #else:                                      
                    
                Znow = solver.y[3] / solver.y[1]                                                                                          
                if Znow >= self.abun_limit(z=z, Mh=M0) and i > 0:                    
                    Zpre = metals[-2] / Mg_t[-2]
                    Z_t = [Zpre, Znow]                           
                    zmax_a = np.interp(self.abun_limit(z=z, Mh=M0), Z_t,
                        redshifts[-2:])
                                    
            # These next two are different because the condition might
            # be satisfied *at the formation time*, which cannot (by definition)
            # occur for time or mass-limited sources.
            if has_T_limit:
                Mtemp = self.halos.VirialMass(self.pf['pop_temp_limit'], z)
                                
                if solver.y[0] >= Mtemp:
                    zmax_T = np.interp(Mtemp, Mh_t[-2:], redshifts[-2:])
                    
            if has_e_limit and (zmax_e is None):
                
                Eblim = self.pf['pop_bind_limit']
                Ebnow = self.halos.BindingEnergy(Mh_t[-1], redshifts[-1])
                Ehist.append(Ebnow)

                if (Ebnow >= Eblim):
                    
                    # i == 0 means we satisfied this criterion at the
                    # formation redshift.
                    if i == 0:
                        zmax_e = z0
                    else:
                        zmax_e = np.interp(Eblim, Ehist[-2:], redshifts[-2:]) 
            
                    # Potentially require a halo to keep growing
                    # for pop_time_limit *after* crossing this barrier.
                    if has_t_limit and self.pf['pop_time_limit_delay']:
                        tlim = self.time_limit(z=z, Mh=M0)
                        
                        lbtime_myr = self.cosm.LookbackTime(z, zmax_e) \
                            / s_per_yr / 1e6
                    
                        if lbtime_myr >= tlim:
                            hit_dt = True
                        
                            lbtime_myr_prev = self.cosm.LookbackTime(redshifts[-2], z0) \
                                / s_per_yr / 1e6
                        
                            zmax_e = np.interp(tlim,
                                [lbtime_myr_prev, lbtime_myr], redshifts[-2:])
                        
            
            
            # Once zmax is set, keep solving the rate equations but don't adjust
            # zmax.
            if zmax is None:
            
                # If binding energy or Virial temperature are a limiter             
                if ((zmax_e is not None) and has_e_limit) or \
                   ((zmax_T is not None) and has_T_limit):
                    
                    # Only transition if time/mass/Z is ALSO satisfied
                    if (self.pf['pop_limit_logic'] == 'and') and \
                       (has_t_limit or has_m_limit or has_a_limit):
                                                
                        if (zmax_t is not None):
                            zmax = zmax_t
                        if (zmax_m is not None):
                            zmax = zmax_m
                        if (zmax_a is not None):
                            zmax = zmax_a
                        
                        # Take the *lowest* redshift.
                        if zmax is not None:
                            if has_e_limit:
                                zmax = min(zmax, zmax_e)
                            else:
                                zmax = min(zmax, zmax_T)
                            
                    else:
                        zmax = zmax_e if has_e_limit else zmax_T
                
                # If no binding or temperature arguments, use time or mass
                if not (has_e_limit or has_T_limit):
                    if (zmax_t is not None):
                        zmax = zmax_t
                    if (zmax_m is not None):
                        zmax = zmax_m
                    if (zmax_a is not None):
                        zmax = zmax_a 

                # play the trump card
                if has_t_ceil and (not has_t_limit):
                    zmax = max(zmax, zmax_t)

            solver.integrate(solver.t-dz)

        if zmax is None:
            zmax = self.pf['final_redshift']

        # Everything will be returned in order of ascending redshift,
        # which will mean masses are (probably) declining from 0:-1
        z = np.array(redshifts)[-1::-1]
        Mh = np.array(Mh_t)[-1::-1]
        Mg = np.array(Mg_t)[-1::-1]
        Ms = np.array(Mst_t)[-1::-1]
        MZ = np.array(metals)[-1::-1]
        cMs = np.array(cMst_t)[-1::-1]
        tlb = np.array(lbtime)[-1::-1]

        # Derived
        results = {'Mh': Mh, 'Mg': Mg, 'Ms': Ms, 'MZ': MZ, 'cMs': cMs,
            'zmax': zmax, 't': tlb}
        results['Z'] = self.pf['pop_metal_retention'] \
            * (results['MZ'] / results['Mg'])

        for key in results:
            results[key] = np.maximum(results[key], 0.0)

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
                 
    def _guess_Mmin(self):
        """
        Super non-general at the moment sorry.
        """
        
        fn = self.pf['feedback_LW_guesses']
        
        if fn is None:
            return None
        
        if isinstance(fn, basestring):
            anl = ModelSet(fn)
        elif isinstance(fn, ModelSet): 
            anl = fn
        else:
            zarr, Mmin = fn
            
            if np.all(np.logical_or(np.isinf(Mmin), np.isnan(Mmin))):
                print("Provided Mmin guesses are all infinite or NaN.")
                return None
            
            return np.interp(self.halos.z, zarr, Mmin)
        
        # HARD CODING FOR NOW
        blob_name = 'popIII_Mmin'
        Mmin = anl.ExtractData(blob_name)[blob_name]
        zarr = anl.get_ivars(blob_name)[0]
        
        ##
        # Remember: ModelSet will have {}'s and guesses_from will not.
        ##                
        kw = {par: self.pf[par] \
            for par in self.pf['feedback_LW_guesses_from']}
                
        score = 0.0
        pid = self.pf['feedback_LW_sfrd_popid']
        for k, par in enumerate(self.pf['feedback_LW_guesses_from']):
            p_w_id = '{0!s}{{{1}}}'.format(par, pid)
            
            if p_w_id not in anl.parameters:
                continue
        
            ind = list(anl.parameters).index(p_w_id)
        
            vals = anl.chain[:,ind]    
                            
            score += np.abs(np.log10(vals) - np.log10(kw[par]))
        
        best = np.argmin(score)
                
        return np.interp(self.halos.z, zarr, Mmin[best])
        
    def save(self, prefix=None, fn=None, fmt='npz'):
        """
        Save properties of the population.
        """
        pass
        
