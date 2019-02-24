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
from types import FunctionType
from ..util import ProgressBar
from ..analysis import ModelSet
from scipy.misc import derivative
from scipy.optimize import fsolve, minimize
from ..analysis.BlobFactory import BlobFactory
from scipy.integrate import quad, simps, cumtrapz, ode
from ..util.ParameterFile import par_info, get_pq_pars
from ..physics.RateCoefficients import RateCoefficients
from scipy.interpolate import RectBivariateSpline
from .GalaxyAggregate import GalaxyAggregate
from .Population import normalize_sed
from ..util.Stats import bin_c2e, bin_e2c
from ..util.Math import central_difference, interp1d_wrapper, interp1d, \
    LinearNDInterpolator
from ..phenom.ParameterizedQuantity import ParameterizedQuantity
from ..physics.Constants import s_per_yr, g_per_msun, cm_per_mpc, G, m_p, \
    k_B, h_p, erg_per_ev, ev_per_hz, sigma_T, c, t_edd, cm_per_kpc
    
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
            if name.startswith('_tab'):
                return self.__getattribute__(name)
                
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

    def get_field(self, z, field):
        """
        Return results from SAM (all masses) at input redshift.
        """
        zarr, data = self.Trajectories()
        
        iz = np.argmin(np.abs(z - zarr))
        
        Mh = data['Mh'][:,iz]
        
        return 10**np.interp(np.log10(self.halos.tab_M), np.log10(Mh), 
            np.log10(data[field][:,iz]))

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
            k = np.argmin(np.abs(self.halos.tab_z - z))

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
            self._N_per_Msun[(Emin, Emax)] = self.Nion(Mh=self.halos.tab_M) \
                * self.cosm.b_per_msun
        elif (Emin, Emax) == (10.2, 13.6):
            self._N_per_Msun[(Emin, Emax)] = self.Nlw(Mh=self.halos.tab_M) \
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
                RectBivariateSpline(self.halos.tab_z, np.log(self.halos.tab_M), 
                    self.halos.tab_dndm)
        return self._spline_nh_
    
    @property
    def _tab_MAR(self):
        if not hasattr(self, '_tab_MAR_'):
            self._tab_MAR_ = self.halos.tab_MAR
            #self._tab_MAR_ = \
            #    np.array([self.MAR(self.halos.tab_z[i], self.halos.tab_M) \
            #        for i in range(self.halos.tab_z.size)]) 
            #        
            #self._tab_MAR_ = np.maximum(self._tab_MAR_, 0.0)
            
        return self._tab_MAR_
    
    @property
    def _tab_MAR_at_Mmin(self):
        if not hasattr(self, '_tab_MAR_at_Mmin_'):
            self._tab_MAR_at_Mmin_ = \
                np.array([self.MAR(self.halos.tab_z[i], self._tab_Mmin[i]) \
                    for i in range(self.halos.tab_z.size)])                    
        return self._tab_MAR_at_Mmin_ 
    
    @property
    def _tab_nh_at_Mmin(self):
        if not hasattr(self, '_tab_nh_at_Mmin_'):
            self._tab_nh_at_Mmin_ = \
                np.array([self._spline_nh(self.halos.tab_z[i], 
                    np.log(self._tab_Mmin[i])) \
                    for i in range(self.halos.tab_z.size)]).squeeze()
        return self._tab_nh_at_Mmin_
        
    @property
    def _tab_fstar_at_Mmin(self):
        if not hasattr(self, '_tab_fstar_at_Mmin_'):
            self._tab_fstar_at_Mmin_ = \
                self.SFE(z=self.halos.tab_z, Mh=self._tab_Mmin)
        return self._tab_fstar_at_Mmin_

    @property
    def _tab_sfrd_at_threshold(self):
        """
        Star formation rate density from halos just crossing threshold.

        Essentially the second term of Equation A1 from Furlanetto+ 2017.
        """
        if not hasattr(self, '_tab_sfrd_at_threshold_'):
            if not self.pf['pop_sfr_cross_threshold']:
                self._tab_sfrd_at_threshold_ = np.zeros_like(self.halos.tab_z)
                return self._tab_sfrd_at_threshold_

            # Model: const SFR in threshold-crossing halos.    
            if type(self.pf['pop_sfr']) in [int, float, np.float64]:
                self._tab_sfrd_at_threshold_ = self.pf['pop_sfr'] \
                    * self._tab_nh_at_Mmin * self._tab_Mmin
            else:
                active = 1. - self.fsup(z=self.halos.tab_z) 
                self._tab_sfrd_at_threshold_ = active * self._tab_eta \
                    * self.cosm.fbar_over_fcdm * self._tab_MAR_at_Mmin \
                    * self._tab_fstar_at_Mmin * self._tab_Mmin \
                    * self._tab_nh_at_Mmin \
                    * self.focc(z=self.halos.tab_z, Mh=self._tab_Mmin)
                            
            #self._tab_sfrd_at_threshold_ -= * self.Mmin * n * self.dMmin_dt(self.halos.z)    

            self._tab_sfrd_at_threshold_ *= g_per_msun / s_per_yr / cm_per_mpc**3

            # Don't count this "new" star formation once the minimum mass
            # exceeds some value. At this point, it will (probably, hopefully)
            # be included in the star-formation of some other population.
            if np.isfinite(self.pf['pop_sfr_cross_upto_Tmin']):
                Tlim = self.pf['pop_sfr_cross_upto_Tmin']
                Mlim = self.halos.VirialMass(z=self.halos.tab_z, T=Tlim)

                mask = self.Mmin < Mlim
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
            #if self.is_src_uv:
                fesc = lambda **kwargs: self.fesc(**kwargs)
            #elif (self.is_src_lw or self.is_src_lya):
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

        ok = ~self._tab_sfr_mask
        tab = np.zeros(self.halos.tab_z.size)
        for i, z in enumerate(self.halos.tab_z):

            if z > self.zform:
                continue      

            # Must grab stuff vs. Mh and interpolate to self.halos.M
            # They are guaranteed to have the same redshifts.
            if need_sam:

                kw = {'z': z, 'Mh': self.halos.tab_M}                
                if self.constant_SFE:
                    for key in sam_data.keys():
                        if key == 'Mh':
                            continue

                        kw[key] = np.interp(self.halos.tab_M,
                            sam_data['Mh'][-1::-1], sam_data[key][-1::-1])
                else:
                    raise NotImplemented('help')

            else:
                kw = {'z': z, 'Mh': self.halos.tab_M}

            integrand = self._tab_sfr[i] * self.halos.tab_dndlnm[i] \
                * yield_per_sfr(**kw) * ok[i]

            _tot = np.trapz(integrand, x=np.log(self.halos.tab_M))
            _cumtot = cumtrapz(integrand, x=np.log(self.halos.tab_M), initial=0.0)

            _tmp = _tot - \
                np.interp(np.log(self._tab_Mmin[i]), np.log(self.halos.tab_M), _cumtot)
               
            tab[i] = _tmp
                
        tab *= 1. / s_per_yr / cm_per_mpc**3
        
        if self.pf['pop_sfr_cross_threshold']:
            
            y = yield_per_sfr(z=self.halos.tab_z, Mh=self._tab_Mmin)
            
            if self.pf['pop_sfr'] is not None:
                thresh = self.pf['pop_sfr'] \
                    * self._tab_nh_at_Mmin * self._tab_Mmin \
                    * y / s_per_yr / cm_per_mpc**3
            else:
                active = 1. - self.fsup(z=self.halos.tab_z)  
                thresh = active * self._tab_eta * \
                    self.cosm.fbar_over_fcdm * self._tab_MAR_at_Mmin \
                    * self._tab_fstar_at_Mmin * self._tab_Mmin \
                    * self._tab_nh_at_Mmin * y \
                    / s_per_yr / cm_per_mpc**3
        
            tab += thresh
        
        self._rho_L[(Emin, Emax)] = interp1d(self.halos.tab_z, tab, 
            kind=self.pf['pop_interp_sfrd'])
    
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
            
        tab = np.ones_like(self.halos.tab_z)
        
        # For all halos
        N_per_Msun = self.N_per_Msun(Emin=Emin, Emax=Emax)
        
        if (Emin, Emax) == (13.6, 24.6):
            fesc = self.fesc(z=z, Mh=self.halos.tab_M)
        elif (Emin, Emax) == (10.2, 13.6):
            fesc = self.fesc_LW(z=z, Mh=self.halos.tab_M)
        else:
            raise NotImplemented('help!')
    
        ok = ~self._tab_sfr_mask
        for i, z in enumerate(self.halos.tab_z):
            integrand = self._tab_sfr[i] * self.halos.tab_dndlnm[i] \
                * N_per_Msun * fesc * ok[i]
    
            tot = np.trapz(integrand, x=np.log(self.halos.tab_M))
            cumtot = cumtrapz(integrand, x=np.log(self.halos.tab_M), initial=0.0)
            
            tab[i] = tot - \
                np.interp(np.log(self._tab_Mmin[i]), np.log(self.halos.tab_M), cumtot)
            
        tab *= 1. / s_per_yr / cm_per_mpc**3
        
        self._rho_N[(Emin, Emax)] = interp1d(self.halos.tab_z, tab, 
            kind=self.pf['pop_interp_sfrd'])
    
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
            func = interp1d(self.halos.tab_z, self._tab_sfrd_total, 
                kind=self.pf['pop_interp_sfrd'])
            self._SFRD = lambda z: func(z)

        return self._SFRD
        
    @SFRD.setter
    def SFRD(self, value):
        self._SFRD = value 
        
    @property   
    def nactive(self):
        """
        Compute number of active halos.
        """
    
        if not hasattr(self, '_nactive'):
            self._nactive = interp1d(self.halos.tab_z, self._tab_nh_active, 
                kind=self.pf['pop_interp_sfrd'])
    
        return self._nactive
    
    @property   
    def SMD(self):
        """
        Compute stellar mass density (SMD).
        """
    
        if not hasattr(self, '_SMD'):
            dtdz = np.array(list(map(self.cosm.dtdz, self.halos.tab_z)))
            self._smd_tab = cumtrapz(self._tab_sfrd_total[-1::-1] * dtdz[-1::-1], 
                dx=np.abs(np.diff(self.halos.tab_z[-1::-1])), initial=0.)[-1::-1]
            self._SMD = interp1d(self.halos.tab_z, self._smd_tab, 
                kind=self.pf['pop_interp_sfrd'])
    
        return self._SMD
        
    def MAR(self, z, Mh):
        return self.eta(z, Mh) * np.maximum(self.MGR(z, Mh) * self.fsmooth(z=z, Mh=Mh), 0.)
    
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

            i = np.argmin(np.abs(z - self.halos.tab_z))

            # Integrand: convert MAR from DM MAR to total matter MAR
            integ = self.halos.tab_dndlnm[i] \
                * src.MAR(z, self.halos.tab_M) / self.cosm.fcdm

            Mmin = np.interp(z, self.halos.tab_z, self.Mmin)
            j1 = np.argmin(np.abs(Mmin - self.halos.tab_M))
            if Mmin > self.halos.tab_M[j1]:
                j1 -= 1

            p0 = simps(integ[j1-1:], x=np.log(self.halos.tab_M)[j1-1:])
            p1 = simps(integ[j1:], x=np.log(self.halos.tab_M)[j1:])
            p2 = simps(integ[j1+1:], x=np.log(self.halos.tab_M)[j1+1:])
            p3 = simps(integ[j1+2:], x=np.log(self.halos.tab_M)[j1+2:])

            interp = interp1d(np.log(self.halos.tab_M)[j1-1:j1+3], [p0,p1,p2,p3],
                kind=self.pf['pop_interp_MAR'])

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
            MAR = src.MAR(z, self.halos.tab_M)    
        else:
            MAR = super(GalaxyCohort, self).MAR_via_AM(z)
                    
        # Grab redshift
        k = np.argmin(np.abs(z - self.halos.tab_z))

        integ = self.halos.tab_dndlnm[k] * MAR / self.cosm.fcdm

        Mmin = np.interp(z, self.halos.tab_z, self.Mmin)
        j1 = np.argmin(np.abs(Mmin - self.halos.tab_M))
        if Mmin > self.halos.tab_M[j1]:
            j1 -= 1    

        incremental_Macc = cumtrapz(integ[j1:], x=np.log(self.halos.tab_M)[j1:],
            initial=0.0)

        return self.halos.tab_M[j1:], incremental_Macc

    @property
    def eta(self):
        if not hasattr(self, '_eta'):
            if np.all(self._tab_eta == 1):
                self._eta = lambda z, Mh=None: 1.  
            elif self._tab_eta.ndim == 1:
                self._eta = lambda z, Mh=None: np.interp(z, self.halos.tab_z, self._tab_eta)
            else:
                _eta = RectBivariateSpline(self.halos.tab_z, 
                    np.log10(self.halos.tab_M), self._tab_eta)
                self._eta = lambda z, Mh: _eta(z, np.log10(Mh)).squeeze()
                    
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
        
            if self.pf['pop_MAR_corr'] == 'integral':
                
                _rhs = np.zeros_like(self.halos.tab_z)
                _lhs = np.zeros_like(self.halos.tab_z)
                _tab_eta_ = np.ones_like(self.halos.tab_z)

                for i, z in enumerate(self.halos.tab_z):

                    # eta = rhs / lhs

                    Mmin = self._tab_Mmin[i]

                    # My Eq. 3
                    rhs = self.cosm.rho_cdm_z0 * self.dfcolldt(z)
                    rhs *= (s_per_yr / g_per_msun) * cm_per_mpc**3
                
                    # Accretion onto all halos (of mass M) at this redshift
                    # This is *matter*, not *baryons*
                    MAR = self._tab_MAR[i]
                
                    # Find Mmin in self.halos.tab_M
                    j1 = np.argmin(np.abs(Mmin - self.halos.tab_M))
                    if Mmin > self.halos.tab_M[j1]:
                        j1 -= 1
                
                    integ = self.halos.tab_dndlnm[i] * MAR
                        
                    p0 = simps(integ[j1-1:], x=np.log(self.halos.tab_M)[j1-1:])
                    p1 = simps(integ[j1:], x=np.log(self.halos.tab_M)[j1:])
                    p2 = simps(integ[j1+1:], x=np.log(self.halos.tab_M)[j1+1:])
                    p3 = simps(integ[j1+2:], x=np.log(self.halos.tab_M)[j1+2:])
                
                    interp = interp1d(np.log(self.halos.tab_M)[j1-1:j1+3], [p0,p1,p2,p3],
                        kind=self.pf['pop_interp_MAR'])

                    lhs = interp(np.log(Mmin))
                    
                    _lhs[i] = lhs
                    _rhs[i] = rhs

                    _tab_eta_[i] = rhs / lhs
                    
                # Re-shape to be (z, Mh)    
                self._tab_eta_ = np.reshape(np.tile(_tab_eta_, self.halos.tab_M.size), 
                    (self.halos.tab_z.size, self.halos.tab_M.size))
                       
            elif self.pf['pop_MAR_corr'] == 'slope':
                        
                # In this case, we have the same shape as _tab_MAR
                self._tab_eta_ = np.ones_like(self.halos.tab_MAR)
                                
                # Compute power-law slope as function of mass at all redshifts.
                # Prevent dramatic deviations from this slope, and instead
                # extrapolate from high-mass end downward.
                logM = np.log(self.halos.tab_M)
                logMAR = np.log(self.halos.tab_MAR)
                
                alpha = np.diff(logMAR, axis=1) / np.diff(logM)
                self._tab_alpha = alpha
                
                Ms = self.halos.tab_M.size - 1
                
                # Figure out where alpha < 0, use slope well above this 
                # juncture to extrapolate.
                negative = np.zeros_like(self.halos.tab_z)
                for i in range(self.halos.tab_z.size):
                    huge = np.argwhere(alpha[i,:-5] > 2.)
                    if not np.any(huge):
                        continue
                        
                    ilt0 = int(huge[-1])    

                    # Extrapolate
                    _Msafe = min(self.halos.tab_M[ilt0] * 10, 1e13)
                    iM = np.argmin(np.abs(_Msafe - self.halos.tab_M))
                    Msafe = self.halos.tab_M[iM]
                                        
                    new = self.halos.tab_MAR[i,iM] \
                        * (self.halos.tab_M / Msafe)**alpha[i,iM]
                     
                    # Only apply extrapolation at low mass    
                    self._tab_eta_[i,0:iM] = \
                        new[0:iM] / self.halos.tab_MAR[i,0:iM]
                                    
                        
            else:
                self._tab_eta_ = np.ones_like(self.halos.tab_dndm)
                    
        return self._tab_eta_
        
    def SFR(self, z, Mh=None):
        """
        Star formation rate at redshift z in a halo of mass Mh.
        
        P.S. When you plot this, don't freak out if the slope changes at Mmin.
        It's because all masses below this (and above Mmax) are just set to 
        zero, so it looks like a slope change for line plots since it's
        trying to connect to a point at SFR=Mh=0.
        
        """
        
        if self.pf['pop_sfr'] is not None:
            return self.sfr(z=z, Mh=Mh)
                
        # If Mh is None, it triggers use of _tab_sfr, which spans all
        # halo masses in self.halos.tab_M
        if Mh is None:
                    
            k = np.argmin(np.abs(z - self.halos.tab_z))
                        
            if abs(z - self.halos.tab_z[k]) < ztol:
                return self._tab_sfr[k] * ~self._tab_sfr_mask[k]
            else:
                Mh = self.halos.tab_M
                    
        else:
            
            # Create interpolant to be self-consistent
            # with _tab_sfr. Note that this is slower than it needs to be
            # in cases where we just want to know the SFR at a few redshifts
            # and/or halo masses. But, we're rarely doing such things.
            if not hasattr(self, '_spline_sfr'):
                log10sfr = np.log10(self._tab_sfr)
                # Filter zeros since we're going log10
                log10sfr[np.isinf(log10sfr)] = -20
                log10sfr[np.isnan(log10sfr)] = -20
                
                _spline_sfr = RectBivariateSpline(self.halos.tab_z, 
                    np.log10(self.halos.tab_M), log10sfr)
                
                #func = lambda z, log10M: 10**_spline_sfr(z, log10M).squeeze()
                
                def func(z, log10M):
                    sfr = 10**_spline_sfr(z, log10M).squeeze()

                    M = 10**log10M
                    if type(sfr) is np.ndarray:
                        sfr[M < self.Mmin(z)] = 0.0
                        sfr[M > self.Mmax(z)] = 0.0
                    else:
                        if M < self.Mmin(z):
                            return 0.0
                        if M > self.Mmax(z):
                            return 0.0

                    return sfr
                
                self._spline_sfr = func
                        
            return self._spline_sfr(z, np.log10(Mh))
        
        return self.cosm.fbar_over_fcdm * self.MAR(z, Mh) * self.eta(z) \
            * self.SFE(z=z, Mh=Mh)

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

        on = self.on(z)
        if not np.any(on):
            return z * on

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
            return rhoL * self.src.Spectrum(E) * on
        else:
            return rhoL * on
            
    def StellarMass(self, z, Mh):
        zform, data = self.scaling_relations_sorted(z=z)
        return np.interp(Mh, data['Mh'], data['Ms'])
    
    def MetalMass(self, z, Mh):
        zform, data = self.scaling_relations_sorted(z=z)
        return np.interp(Mh, data['Mh'], data['MZ'])

    def GasMass(self, z, Mh):
        zform, data = self.scaling_relations_sorted(z=z)
        return np.interp(Mh, data['Mh'], data['Mg'])
        
    def StellarMassFunction(self, z, bins=None, units='dex'):
        """
        Return stellar mass function (duh).
        """
        zall, traj_all = self.Trajectories()
        iz = np.argmin(np.abs(z - zall))
        Ms = traj_all['Ms'][:,iz]
        Mh = traj_all['Mh'][:,iz]
        nh = traj_all['nh'][:,iz]
        
        if bins is None:
            bin = 0.1
            bin_e = np.arange(6., 13.+bin, bin)
        else:
            dx = np.diff(bins)
            assert np.all(np.diff(dx) == 0)
            bin = dx[0]
            bin_e = bins
            
        bin_c = bin_e2c(bin_e)

        phi, _bins = np.histogram(Ms, bins=10**bin_e, weights=nh)                
        
        if units == 'dex':
            # Convert to dex**-1 units
            phi /= bin
        else:
            raise NotImplemented('help')
        
        if bins is None:
            return 10**bin_c, phi
        else:
            return phi

    def SurfaceDensity(self, z, mag=None, dz=1., dtheta=1.):
        """
        
        Returns
        -------
        Observed magnitudes, then, projected surface density of galaxies in 
        `dz` thick shell, in units of cumulative number of galaxies per 
        square degree.
        
        """
        
        # These are intrinsic (i.e., not dust-corrected) absolute magnitudes 
        _mags, _phi = self.phi_of_M(z=z)
        
        mask = np.logical_or(_mags.mask, _phi.mask)
        
        mags = _mags[mask == 0]
        phi = _phi[mask == 0]
        
        # Observed magnitudes will be dimmer, + AB shift from absolute to apparent mags
        Mobs = self.dust.Mobs(z, mags) + 48.6
        
        # Compute the volume of the shell we're looking at
        vol = self.cosm.ProjectedVolume(z, angle=dtheta, dz=dz)
        
        Ngal = phi * vol
        
        # At this point, magnitudes are in descending order, i.e., faint
        # to bright.
        
        # Because we want the cumulative number *brighter* than m_AB, 
        # reverse the arrays and integrate from bright end down.
        
        Mobs = Mobs[-1::-1]
        Ngal = Ngal[-1::-1]
        
        # Cumulative surface density of galaxies *brighter than* Mobs
        cgal = cumtrapz(Ngal, x=Mobs, initial=Ngal[0])
        
        if mag is not None:
            return np.interp(mag, Mobs, cgal)
        else:
            return Mobs, cgal

    #def SMF(self, z):
    #    if not hasattr(self, '_phi_of_Mst'):
    #        self._phi_of_Mst = {}
    #    else:
    #        if z in self._phi_of_Mst:
    #            return self._phi_of_Mst[z]
    #
    #    zform, data = self.Trajectories()#self.scaling_relations_sorted(z)
    #
    #    Mh = data['Mh']
    #    Ms = data['Ms']
    #    
    #    dndm_func = interp1d(self.halos.tab_z, self.halos.tab_dndm[:,:-1], axis=0,
    #        kind=self.pf['pop_interp_lf'])
    #    dndm_z = dndm_func(z)
    #
    #    # Interpolate dndm to same Mh grid as SAM
    #    dndm_sam = np.interp(Mh, self.halos.tab_M[0:-1], dndm_z)
    #
    #    dndm = dndm_sam * self.focc(z=z, Mh=Mh)
    #    dMh_dMs = np.diff(Mh) / np.diff(Ms)
    #            
    #    dMh_dlogMs = dMh_dMs * Ms[0:-1]
    #    
    #    # Only return stuff above Mmin
    #    Mmin = np.interp(z, self.halos.tab_z, self._tab_Mmin)
    #    if self.pf['pop_Tmax'] is None:
    #        Mmax = self.pf['pop_lf_Mmax']
    #    else:
    #        Mmax = np.interp(z, self.halos.tab_z, self._tab_Mmax)
    #    
    #    i_min = np.argmin(np.abs(Mmin - self.halos.tab_M))
    #    i_max = np.argmin(np.abs(Mmax - self.halos.tab_M))
    #
    #    phi_of_Ms = dndm[0:-1] * dMh_dlogMs
    #
    #    above_Mmin = Mh >= Mmin
    #    below_Mmax = Mh <= Mmax
    #    ok = np.logical_and(above_Mmin, below_Mmax)[0:-1]
    #    mask = self.mask = np.logical_not(ok)
    #
    #    mass = np.ma.array(Ms[:-1], mask=mask)
    #    phi = np.ma.array(phi_of_Ms, mask=mask, fill_value=tiny_phi)
    #
    #    self._phi_of_Mst[z] = mass, phi
    #
    #    return self._phi_of_Mst[z]
        
    @property
    def is_uvlf_parametric(self):
        if not hasattr(self, '_is_uvlf_parametric'):
            self._is_uvlf_parametric = self.pf['pop_uvlf'] is not None
        return self._is_uvlf_parametric 
    
    def UVLF_M(self, MUV, z=None):
        if self.is_uvlf_parametric:
            return self.uvlf(MUV=MUV, z=z)
    
        ##
        # Otherwise, standard SFE parameterized approach.
        ##
            
        x_phi, phi = self.phi_of_M(z)
        
        ok = phi.mask == False

        if ok.sum() == 0:
            return -np.inf
    
        # Setup interpolant. x_phi is in descending, remember!
        interp = interp1d(x_phi[ok][-1::-1], np.log10(phi[ok][-1::-1]), 
            kind=self.pf['pop_interp_lf'],
            bounds_error=False, fill_value=-np.inf)

        phi_of_x = 10**interp(MUV)
        
        return phi_of_x
    
    def UVLF_L(self, LUV, z=None):
        x_phi, phi = self.phi_of_L(z)
    
        ok = phi.mask == False
    
        if ok.sum() == 0:
            return -np.inf    
    
        # Setup interpolant
        interp = interp1d(np.log10(x_phi[ok]), np.log10(phi[ok]), 
            kind=self.pf['pop_interp_lf'], 
            bounds_error=False, fill_value=-np.inf)
    
        phi_of_x = 10**interp(np.log10(LUV))
        
        return phi_of_x
    
    def LuminosityFunction(self, z, x, mags=True, wave=1600.):
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
            phi_of_x = self.UVLF_M(x, z, wave=wave)
        else:
            phi_of_x = self.UVLF_L(x, z, wave=wave)
    
        return phi_of_x

    def Lh(self, z, wave=1600.):
        """
        This is the rest-frame UV band in which the LF is measured.
        
        NOT generally use-able!!!
        
        """
        
        if self.pf['pop_star_formation']:
            
            # This uses __getattr__ in case we're allowing Z to be 
            # updated from SAM.
            sfr = self.SFR(z) 
            
            if self.pf['pop_dust_yield'] > 0:
                
                L_sfr = self.src.L_per_sfr(wave)                                
                Lh = L_sfr * sfr
                
                fcov = self.dust_fcov(Mh=self.halos.tab_M)
                kappa = self.dust_kappa(wave=wave)
                Sd = self.get_field(z, 'Sd')
                tau = kappa * Sd
                
                return Lh * (1 - fcov) + Lh * fcov * np.exp(-tau)
                
            else:
                L_sfr = self.L1600_per_sfr(z=z, Mh=self.halos.tab_M)
                return sfr * L_sfr
            
        elif self.pf['pop_bh_formation']:
            # In this case, luminosity just proportional to BH mass.
            zarr, data = self.Trajectories()
            
            iz = np.argmin(np.abs(zarr - z))
            
            # Interpolate Mbh onto halo mass grid so we can use abundances.
            Mbh = np.exp(np.interp(np.log(self.halos.tab_M), 
                np.log(data['Mh'][:,iz]), 
                np.log(data['Mbh'][:,iz])))

            # Bolometric luminosity: Eddington 
            ledd = 4 * np.pi * G * m_p * c / sigma_T
            Lbol = ledd * Mbh * g_per_msun
            Lbol[np.isnan(Lbol)]= 0.0

            # Need to do bolometric correction.
            E = h_p * c / (wave * 1e-8) / erg_per_ev
            I_E = self.src.Spectrum(E)
            
            return Lbol * I_E * ev_per_hz

            # Don't need to do trajectories unless we're letting
            # BHs grow via accretion, i.e., scaling laws can just get
            # painted on.

        else:
            raise NotImplemented('help')

    def phi_of_L(self, z, wave=1600.):
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
            for red in self._phi_of_L:
                if abs(red - z) < ztol:
                    return self._phi_of_L[red]
                
        Lh = self.Lh(z, wave=wave)
                
        fobsc = (1. - self.fobsc(z=z, Mh=self.halos.tab_M))        
        # Means obscuration refers to fractional dimming of individual 
        # objects
        if self.pf['pop_fobsc_by'] == 'lum':
            Lh *= fobsc
        
        logL_Lh = np.log(Lh)
        
        iz = np.argmin(np.abs(z - self.halos.tab_z))
        
        if abs(z - self.halos.tab_z[iz]) < ztol:
            dndm = self.halos.tab_dndm[iz,:-1]
        else:
            dndm_func = interp1d(self.halos.tab_z, self.halos.tab_dndm[:,:-1], 
                axis=0, kind=self.pf['pop_interp_lf'])
                        
            dndm = dndm_func(z) * self.focc(z=z, Mh=self.halos.tab_M[0:-1])
        
        # In this case, obscuration means fraction of objects you don't see
        # in the UV.
        if self.pf['pop_fobsc_by'] == 'num':
            dndm *= fobsc[0:-1]
            
        dMh_dLh = np.diff(self.halos.tab_M) / np.diff(Lh)
                
        dMh_dlogLh = dMh_dLh * Lh[0:-1]
                
        # Only return stuff above Mmin
        Mmin = np.interp(z, self.halos.tab_z, self._tab_Mmin)
        Mmax = self.pf['pop_lf_Mmax']
        
        i_min = np.argmin(np.abs(Mmin - self.halos.tab_M))
        i_max = np.argmin(np.abs(Mmax - self.halos.tab_M))

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

        above_Mmin = self.halos.tab_M >= Mmin
        below_Mmax = self.halos.tab_M <= Mmax
        ok = np.logical_and(above_Mmin, below_Mmax)[0:-1]
        mask = self.mask = np.logical_not(ok)
        
        lum = np.ma.array(Lh[:-1], mask=mask)
        phi = np.ma.array(phi_of_L, mask=mask, fill_value=tiny_phi)

        self._phi_of_L[z] = lum, phi

        return self._phi_of_L[z]

    def phi_of_M(self, z, wave=1600.):
        if not hasattr(self, '_phi_of_M'):
            self._phi_of_M = {}
        else:
            if z in self._phi_of_M:
                return self._phi_of_M[z]
            for red in self._phi_of_M:
                if np.allclose(red, z):
                    return self._phi_of_M[red]

        Lh, phi_of_L = self.phi_of_L(z, wave=wave)

        MAB = self.magsys.L_to_MAB(Lh, z=z)

        phi_of_M = phi_of_L[0:-1] * np.abs(np.diff(Lh) / np.diff(MAB))

        self._phi_of_M[z] = MAB[0:-1], phi_of_M

        return self._phi_of_M[z]
        
    def Beta(self, z, wave=1600., dlam=10):
        """
        UV slope.
        """
        

        ok = np.logical_and(wave-dlam <= self.src.wavelengths, 
                            self.src.wavelengths <= wave+dlam)

        arr = self.src.wavelengths[ok==1]
                
        Lh = np.array([self.Lh(z, w) for w in arr])
        
        Llam = Lh / self.src.dwdn[ok==1][:,None]
        
        logw = np.log(arr)
        logL = np.log(Llam)
                
        beta = (logL[0,:] - logL[-1,:]) / (logw[0] - logw[-1])

        return beta


    def MUV(self, z, Mh, wave=1600.):
        Lh = np.interp(Mh, self.halos.tab_M, self.Lh(z, wave=wave))
        MAB = self.magsys.L_to_MAB(Lh, z=z)
        return MAB

    def MUV_max(self, z):
        """
        Compute the magnitude corresponding to the Tmin threshold.
        """   

        return self.MUV(z, Mmin)

    def Mh_of_MUV(self, z, MUV, wave=1600.):
        
        # MAB corresponds to self.halos.tab_M
        MAB, phi = self.phi_of_M(z, wave=wave)
        ok = MAB.mask == 0
        
        if ok.sum() == 0:
            return 0.0

        return np.interp(MUV, MAB[ok][-1::-1], self.halos.tab_M[1:-1][ok][-1::-1])
        
    def MUV_at_peak_SFE(self, z):
        """
        Return the UV magnitude of a halo forming stars at peak efficiency.
        """
        
        slope = lambda M: self.gamma_sfe(z, M)
        
        low = fsolve(slope, 1e11)
        
        Mpeak = low[0]
        
        MAB, phi = self.phi_of_M(z)
        MUV_peak = np.interp(Mpeak, self.halos.tab_M[1:-1], MAB)
        
        return MUV_peak
        
    @property
    def _tab_Mmax_active(self):
        """ most massive star-forming halo. """
        if not hasattr(self, '_tab_Mmax_active_'):
            self._tab_Mmax_active_ = np.zeros_like(self.halos.tab_z)
            for i, z in enumerate(self.halos.tab_z):
                lim = self.pf['pop_fstar_negligible']
                fstar_max = self._tab_fstar[i].max()
                immsfh = np.argmin(np.abs(self._tab_fstar[i] - fstar_max * lim))
                self._tab_Mmax_active_[i] = self.halos.tab_M[immsfh]
        return self._tab_Mmax_active_
    
    @property
    def Mmax_active(self):
        if not hasattr(self, '_Mmax_active_'):
            self._Mmax_active_ = \
                lambda z: np.interp(z, self.halos.tab_z, self._tab_Mmax_active)
        return self._Mmax_active_
        
    def dMmin_dt(self, z):
        """ Solar masses per year. """
        return -1. * derivative(self.Mmin, z) * s_per_yr / self.cosm.dtdz(z)

    @property
    def M_atom(self):
        if not hasattr(self, '_Matom'):
            Mvir = lambda z: self.halos.VirialMass(z, 1e4, mu=self.pf['mu'])
            self._Matom = np.array(list(map(Mvir, self.halos.tab_z)))
        return self._Matom    

    @property
    def Mmin(self):
        if not hasattr(self, '_Mmin'):  
            self._Mmin = lambda z: \
                np.interp(z, self.halos.tab_z, self._tab_Mmin)

        return self._Mmin

    @Mmin.setter
    def Mmin(self, value):
        if ismethod(value):
            self._Mmin = value
        else:
            self._tab_Mmin = value
            self._Mmin = lambda z: np.interp(z, self.halos.tab_z, self._tab_Mmin)
    
    def Mmax(self, z):
        # Doesn't have a setter because of how we do things in Composite.
        # Long story.
        return np.interp(z, self.halos.tab_z, self._tab_Mmax)
    
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
    def _loaded_guesses(self):
        if not hasattr(self, '_loaded_guesses_'):
            self._loaded_guesses_ = False
        return self._loaded_guesses_
    
    @_loaded_guesses.setter
    def _loaded_guesses(self, value):
        self._loaded_guesses_ = value
    
    @property
    def _tab_Mmin(self):
        if not hasattr(self, '_tab_Mmin_'):
                        
            # First, compute threshold mass vs. redshift
            if self.pf['feedback_LW_guesses'] is not None and (not self._loaded_guesses):
                guess = self._guess_Mmin()
                if guess is not None:
                    self._tab_Mmin = guess
                    return self._tab_Mmin_
            
            if self.pf['pop_Mmin'] is not None:
                if ismethod(self.pf['pop_Mmin']) or \
                   type(self.pf['pop_Mmin']) == FunctionType:
                    self._tab_Mmin_ = \
                        np.array(map(self.pf['pop_Mmin'], self.halos.tab_z))
                elif type(self.pf['pop_Mmin']) is np.ndarray:
                    self._tab_Mmin_ = self.pf['pop_Mmin']
                    assert self._tab_Mmin.size == self.halos.tab_z.size
                elif type(self.pf['pop_Mmin']) is str:
                    if self.pf['pop_Mmin'] == 'jeans':
                        self._tab_Mmin_ = \
                            np.array(map(self.cosm.JeansMass, self.halos.tab_z))
                    elif self.pf['pop_Mmin'] == 'filtering':
                        self._tab_Mmin_ = \
                            np.array(map(self.halos.FilteringMass, self.halos.tab_z))
                    else:
                        raise NotImplemented('help')
                else:    
                    self._tab_Mmin_ = self.pf['pop_Mmin'] \
                        * np.ones_like(self.halos.tab_z)
            else:
                self._tab_Mmin_ = self.halos.VirialMass(
                    self.halos.tab_z, self.pf['pop_Tmin'], mu=self.pf['mu'])
                
            self._tab_Mmin_ = self._apply_lim(self._tab_Mmin_, 'min')
                
        return self._tab_Mmin_
    
    @_tab_Mmin.setter
    def _tab_Mmin(self, value):
        if ismethod(value):
            self.Mmin = value
            self._tab_Mmin_ = np.array(list(map(value, self.halos.tab_z)), 
                dtype=float)
        elif type(value) in [int, float, np.float64]:    
            self._tab_Mmin_ = value * np.ones_like(self.halos.tab_z)
        else:
            self._tab_Mmin_ = value
            
        self._tab_Mmin_ = self._apply_lim(self._tab_Mmin_, s='min')
            
    @property
    def _tab_n_Mmin(self):
        """
        Number of objects in each Mmin bin. Only use this for setting
        up Ensemble objects?
        """
        if not hasattr(self, '_tab_n_Mmin_'):
            
            # Need to setup spline for n(>M)                        
            log10_ngtm = np.log10(self.halos.tab_ngtm)
            not_ok = np.isinf(log10_ngtm)
            ok = np.logical_not(not_ok)
            
            log10_ngtm[ok==0] = -40.
    
            _spl = RectBivariateSpline(self.halos.tab_z, 
               np.log10(self.halos.tab_M), log10_ngtm)
            spl = lambda z, log10M: 10**_spl(z, log10M).squeeze()

            # Interpolate halo abundances onto Mmin axis.
            ngtm_Mmin = np.array([spl(self.halos.tab_z[i],
                np.log10(self._tab_Mmin)[i]) \
                    for i in range(self.halos.tab_z.size)])

            # Number of halos in this Mmin bin is just the difference
            # in N(M>Mmin) between two redshift steps.
            # Remember, though, that ngtm_Mmin is in *descending* order
            # since it rises with redshift, hence the minus sign.
            n_new = np.concatenate((-np.diff(ngtm_Mmin), [0.0]))

            self._tab_n_Mmin_ = n_new

        return self._tab_n_Mmin_    

    @property    
    def _tab_Mmin_floor(self):
        if not hasattr(self, '_tab_Mmin_floor_'):
            self._tab_Mmin_floor_ = self.halos.Mmin_floor(self.halos.tab_z)
        return self._tab_Mmin_floor_
            
    def _apply_lim(self, arr, s='min', zarr=None):
        out = None

        if zarr is None:
            zarr = self.halos.tab_z

        # Might need these if Mmin is being set dynamically
        if self.pf['pop_M{!s}_ceil'.format(s)] is not None:
            out = np.minimum(arr, self.pf['pop_M{!s}_ceil'.format(s)])
        if self.pf['pop_M{!s}_floor'.format(s)] is not None:
            out = np.maximum(arr, self.pf['pop_M{!s}_floor'.format(s)])
        if self.pf['pop_T{!s}_ceil'.format(s)] is not None:
            _f = lambda z: self.halos.VirialMass(z, 
                self.pf['pop_T{!s}_ceil'.format(s)], mu=self.pf['mu'])
            _MofT = np.array(list(map(_f, zarr)))
            out = np.minimum(arr, _MofT)
        if self.pf['pop_T{!s}_floor'.format(s)] is not None:
            _f = lambda z: self.halos.VirialMass(z, 
                self.pf['pop_T{!s}_floor'.format(s)], mu=self.pf['mu'])
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
                Mmax = np.zeros_like(self.halos.tab_z)
                for i, z in enumerate(self.halos.tab_z):
                    
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
                    self._tab_Mmax_ = np.array(list(map(self.pf['pop_Mmax'], self.halos.tab_z)))
                elif type(self.pf['pop_Mmax']) is tuple:
                    extra = self.pf['pop_Mmax'][0]
                    assert self.pf['pop_Mmax'][1] == 'Mmin'

                    if type(extra) is FunctionType:
                        self._tab_Mmax_ = np.array(list(map(extra, self.halos.tab_z))) \
                            * self._tab_Mmin
                    else:
                        self._tab_Mmax_ = extra * self._tab_Mmin
                else:    
                    self._tab_Mmax_ = self.pf['pop_Mmax'] * np.ones_like(self.halos.tab_z)
            elif self.pf['pop_Tmax'] is not None:
                Mvir = lambda z: self.halos.VirialMass(z, self.pf['pop_Tmax'], 
                    mu=self.pf['mu'])
                self._tab_Mmax_ = np.array(list(map(Mvir, self.halos.tab_z)))
            else:
                # A suitably large number for (I think) any purpose
                self._tab_Mmax_ = 1e18 * np.ones_like(self.halos.tab_z)
    
            self._tab_Mmax_ = self._apply_lim(self._tab_Mmax_, s='max')
            self._tab_Mmax_ = np.maximum(self._tab_Mmax_, self._tab_Mmin)
                
        return self._tab_Mmax_
    
    @_tab_Mmax.setter
    def _tab_Mmax(self, value):
        if type(value) in [int, float, np.float64]:    
            self._tab_Mmax_ = value * np.ones_like(self.halos.tab_z)
        else:
            self._tab_Mmax_ = value
        
    @property
    def _tab_sfr_mask(self):
        if not hasattr(self, '_tab_sfr_mask_'):
            sfr = self._tab_sfr
        return self._tab_sfr_mask_
        
    @property
    def _tab_sfr(self):
        """
        SFR as a function of redshift and halo mass.

            ..note:: Units are Msun/yr.
            
        This does NOT set the SFR to zero in halos with M < Mmin or M > Mmax!    
        Doing so screws up spline fitting in SFR...but if we don't need the
        SFR function for anything...shouldn't we just do it?

        """
        if not hasattr(self, '_tab_sfr_'):
            
            if self.pf['pop_sfr_model'] == 'sfr-func':
                self._tab_sfr_ = \
                    np.zeros((self.halos.tab_z.size, self.halos.tab_M.size))
                
                for i, z in enumerate(self.halos.tab_z):

                    if z > self.zform:
                        continue

                    if z < self.zdead:
                        continue
                    
                    # Should be a little careful here: need to go one or two
                    # steps past edge to avoid interpolation problems in SFRD.

                    # SF fueld by accretion onto halos already above threshold
                    #if self.pf['pop_sfr_above_threshold']:
                
                    if self.pf['pop_sfr_model'] == 'sfr-func':
                        self._tab_sfr_[i] = self.sfr(z=z, Mh=self.halos.tab_M)
                    else:                            
                        raise ValueError('shouldnt happen.')
                
            else:   
                self._tab_sfr_ = self._tab_eta \
                    * self.cosm.fbar_over_fcdm \
                    * self.halos.tab_MAR * self._tab_fstar

            # Mmin is like tab_z, make it like (z, M)
            # M is like tab_M, make it like (z, M)
            Mmin = np.array([self._tab_Mmin] * self.halos.tab_M.size).T
            Mmax = np.array([self._tab_Mmax] * self.halos.tab_M.size).T
            M = np.reshape(np.tile(self.halos.tab_M, self.halos.tab_z.size), 
                    (self.halos.tab_z.size, self.halos.tab_M.size))

            mask = np.zeros_like(self._tab_sfr_, dtype=bool)
            mask[M < Mmin] = True
            mask[M > Mmax] = True
            mask[self.halos.tab_z > self.zform] = True
            mask[self.halos.tab_z < self.zdead] = True
            self._tab_sfr_mask_ = mask

            # Why am I getting a NaN?
            isnan = np.isnan(self._tab_sfr_)

            if isnan.sum() > 1:
                print("WARNING: {} Nans detected in _tab_sfr_".format(isnan.sum()))
                #raise ValueError('Nans!')
            
            self._tab_sfr_[isnan] = 0.
                                
        return self._tab_sfr_

    @property
    def SFRD_at_threshold(self):
        if not hasattr(self, '_SFRD_at_threshold'):
            self._SFRD_at_threshold = \
                lambda z: np.interp(z, self.halos.tab_z, self._tab_sfrd_at_threshold)
        return self._SFRD_at_threshold
        
    @property
    def _tab_nh_active(self):
        if not hasattr(self, '_tab_nh_active_'):
            self._tab_nh_active_ = np.ones_like(self.halos.tab_z)

            # Loop from high-z to low-z
            for k, z in enumerate(self.halos.tab_z[-1::-1]):

                i = self.halos.tab_z.size - k - 1
                
                if not self.pf['pop_sfr_above_threshold']:
                    break

                if z > self.zform:
                    continue

                integrand = self.halos.tab_dndlnm[i] \
                    * self.focc(z=z, Mh=self.halos.tab_M)

                # Mmin and Mmax will never be exactly on Mh grid points 
                # so we interpolate to more precisely determine SFRD.    

                c1 = self.halos.tab_M >= self._tab_Mmin[i]
                c2 = self.halos.tab_M <= self._tab_Mmax[i]
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
                    i1 = np.argmin(np.abs(self.halos.tab_M - self._tab_Mmin[i]))
                    if self.halos.tab_M[i1] > self._tab_Mmin[i]:
                        i1 -= 1
                    i2 = i1 + 1
                    
                    # Trapezoid here we come
                    b = self._tab_logMmax[i] - self._tab_logMmin[i]
                    
                    M1 = self._tab_logMmin[i]
                    M2 = self._tab_logMmax[i]
                    y1 = np.interp(M1, [np.log(self.halos.tab_M[i1]), np.log(self.halos.tab_M[i2])],
                        [integrand[i1], integrand[i2]])
                    y2 = np.interp(M2, [np.log(self.halos.tab_M[i1]), np.log(self.halos.tab_M[i2])],
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
                    b = np.log(self.halos.tab_M[Mlo1+1]) - np.log(self.halos.tab_M[Mlo1])
                    #h = abs(integrand[Mlo1+1] - integrand[Mlo1])
                    #b = self.halos.lnM[Mlo1] - self.self.halos.lnM[Mlo1+1]
                    
                    M1 = self._tab_logMmin[i]
                    M2 = self._tab_logMmax[i]
                    y1 = np.interp(M1, [np.log(self.halos.tab_M[i1]), np.log(self.halos.tab_M[i2])],
                        [integrand[i1], integrand[i2]])
                    y2 = np.interp(M2, [np.log(self.halos.tab_M[i1]), np.log(self.halos.tab_M[i2])],
                        [integrand[i1], integrand[i2]])
                    
                    h = abs(y2 - y1)
                    
                    tot = 0.5 * b * h
                else:
                    # This is essentially an integral from Mlo1 to Mhi1
                    tot = np.trapz(integrand[ok], x=np.log(self.halos.tab_M[ok]))
                                
                integ_lo = np.trapz(integrand[Mlo2:Mhi1+1], 
                    x=np.log(self.halos.tab_M[Mlo2:Mhi1+1]))
                                
                # Interpolating over lower integral bound
                sfrd_lo = np.interp(self._tab_logMmin[i], 
                    [np.log(self.halos.tab_M[Mlo2]), np.log(self.halos.tab_M[Mlo1])], 
                    [integ_lo, tot]) - tot
                                                
                if Mhi2 >= self.halos.tab_M.size:    
                    sfrd_hi = 0.0
                else:
                    integ_hi = np.trapz(integrand[Mlo1:Mhi2+1], 
                        x=np.log(self.halos.tab_M[Mlo1:Mhi2+1]))
                    sfrd_hi = np.interp(self._tab_logMmax[i], 
                        [np.log(self.halos.tab_M[Mhi1]), np.log(self.halos.tab_M[Mhi2])], 
                        [tot, integ_hi]) - tot
                
                self._tab_nh_active_[i] = tot + sfrd_lo + sfrd_hi

            self._tab_nh_active_ *= 1. / cm_per_mpc**3

            #if self.pf['pop_sfr_cross_threshold']:
            #    self._tab_sfrd_total_ += self._tab_sfrd_at_threshold

        return self._tab_nh_active_
        
    def intra_bin_corr(self, z, arr):
        raise NotImplemented('help')
        lo_corr = np.zeros_like(self.halos.tab_z)
        
        # Construct array of masses, all of which are the grid pt just
        # above Mmin at the corresponding redshift.
        lo_i = np.argmax(lo_mask, axis=1)
        lo_M = self.halos.tab_M[lo_i]
        
        # Grab unmasked SFR array so we get intra-bin correction right.
        sfr = self._tab_sfr # no * ok
        num = self.halos.tab_dndlnm
        Mmin = self._tab_Mmin
        
        b = np.log(lo_M) - np.log(self._tab_Mmin)
        y1 = np.array([sfr[i,lo_i[i]] * num[i,lo_i[i]] for i in range(Nz)])
        y2 = np.array([np.interp(Mmin[i], self.halos.tab_M, 
            sfr[i,:] * num[i,:]) for i in range(Nz)])
        
        lo_corr = 0.5 * b * (y1 + y2)
                    
        # Insensitive to this so leave it out for now.
        hi_corr = np.zeros_like(self.halos.tab_z)
        
    @property
    def _tab_sfrd_total(self):
        """
        SFRD as a function of redshift.
    
            ..note:: Units are g/s/cm^3 (comoving).

        """

        if not hasattr(self, '_tab_sfrd_total_'):
            Nz = self.halos.tab_z.size
            
            ok = ~self._tab_sfr_mask
            integrand = self._tab_sfr * self.halos.tab_dndlnm \
                * self._tab_focc
                  
            ##
            # Use cumtrapz instead and interpolate onto Mmin, Mmax
            ##
            self._tab_sfrd_total_ = np.zeros_like(self.halos.tab_z)
            for i, z in enumerate(self.halos.tab_z):
                
                if z < self.pf['final_redshift']:
                    continue
                
                if z > self.pf['initial_redshift']:
                    continue
                
                if z > self.zform:
                    continue

                if z < self.zdead:
                    continue
                
                tot = np.trapz(integrand[i], x=np.log(self.halos.tab_M))
                cumtot = cumtrapz(integrand[i], x=np.log(self.halos.tab_M), 
                    initial=0.0)
                    
                above_Mmin = np.interp(np.log(self._tab_Mmin[i]), 
                        np.log(self.halos.tab_M), tot - cumtot)
                above_Mmax = np.interp(np.log(self._tab_Mmax[i]), 
                        np.log(self.halos.tab_M), tot - cumtot)    
                
                if above_Mmin < above_Mmax:
                    print("WARNING: SFRD(>Mmin) < SFRD(>Mmax) at z={}".format(z))
                                    
                self._tab_sfrd_total_[i] = above_Mmin - above_Mmax
                
            self._tab_sfrd_total_ *= g_per_msun / s_per_yr / cm_per_mpc**3    
                        
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
    
    @property        
    def _tab_focc(self):
        if not hasattr(self, '_tab_focc_'):
            yy, xx = self._tab_Mz
            self._tab_focc_ = self.focc(z=xx, Mh=yy)
        return self._tab_focc_
    
    def SFRD_within(self, z, Mlo, Mhi=None, is_mag=False):
        """
        Compute SFRD within given mass range, [Mlo, Mhi].
        """
        
        #if not hasattr(self, '_sfrd_within'):
        #    self._sfrd_within = {}
        #    
        #if (Mlo, Mhi) in self._sfrd_within.keys():
        #    return self._sfrd_within[(Mlo, Mhi)](z)
        
        #_sfrd_tab = np.ones_like(self.halos.tab_z)
        
        if is_mag:
            _Mlo = self.Mh_of_MUV(z, Mlo)
        # Crudely for now
        elif Mlo == 'Mmin':
            _Mlo = self.Mmin(z)
        else:
            _Mlo = Mlo
        
        # Check for exact match
        iz = np.argmin(np.abs(z - self.halos.tab_z))    
        if abs(self.halos.tab_z[iz] - z) < ztol:
            exact_match = True
        else:
            exact_match = False
            raise NotImplemented('help')

        ok = ~self._tab_sfr_mask
        integrand = ok * self._tab_sfr * self.halos.tab_dndlnm * self._tab_focc        
        
        # Have everything we need.
        if exact_match:
            
            ilo = np.argmin(np.abs(self.halos.tab_M - _Mlo))
                    
            if Mhi is None:
                ihi = self.halos.tab_M.size
            else:
                ihi = np.argmin(np.abs(self.halos.tab_M - Mhi))
                        
            _sfrd_tab = np.trapz(integrand[iz,ilo:ihi+1], 
                x=np.log(self.halos.tab_M[ilo:ihi+1]))
                
            # Should do the intra-bin correction here.
                
            _sfrd_tab *= g_per_msun / s_per_yr / cm_per_mpc**3
        
            return _sfrd_tab
            
        #_sfrd_func = lambda zz: np.interp(zz, self.halos.tab_z, _sfrd_tab)
                
        #if type(Mlo) != np.ndarray:
        #    self._sfrd_within[(Mlo, Mhi)] = _sfrd_func
        
        #return _sfrd_func(z)

    @property
    def LLyC_tab(self):
        """
        Number of LyC photons emitted per unit SFR in halos of mass M.
        """
        if not hasattr(self, '_LLyC_tab'):
            M = self.halos.tab_M
            fesc = self.fesc(None, M)
            
            dnu = (24.6 - 13.6) / ev_per_hz

            Nion_per_L1600 = self.Nion(None, M) / (1. / dnu)
            
            self._LLyC_tab = np.zeros([self.halos.tab_z.size, self.halos.tab_M.size])
            
            for i, z in enumerate(self.halos.tab_z):
                self._LLyC_tab[i] = self.L1600_tab[i] * Nion_per_L1600 \
                    * fesc
            
                mask = self.halos.tab_M >= self._tab_Mmin[i]
                self._LLyC_tab[i] *= mask
            
        return self._LLyC_tab
                
    @property
    def LLW_tab(self):
        if not hasattr(self, '_LLW_tab'):
            M = self.halos.tab_M
                
            dnu = (13.6 - 10.2) / ev_per_hz
            #nrg_per_phot = 25. * erg_per_ev
    
            Nlw_per_L1600 = self.Nlw(z=None, M=M) / (1. / dnu)
            fesc_LW = self.fesc_LW(z=None, M=M)
    
            self._LLW_tab = np.zeros([self.halos.tab_z.size, self.halos.tab_M.size])
    
            for i, z in enumerate(self.halos.tab_z):
                self._LLW_tab[i] = self.L1600_tab[i] * Nlw_per_L1600
    
                mask = self.halos.tab_M >= self._tab_Mmin[i]
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
            
            if not self.pf['pop_star_formation']:
                self._fstar = lambda **kwargs: 0.0
                            
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
                    Mmin = lambda z: np.interp(z, self.halos.tab_z, self._tab_Mmin)
                
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
                                        
                    Mmin = lambda z: np.interp(z, self.halos.tab_z, self._tab_Mmin)
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
    def _tab_Mz(self):
        if not hasattr(self, '_tab_Mz_'):
            yy, xx = np.meshgrid(self.halos.tab_M, self.halos.tab_z)
            self._tab_Mz_ = yy, xx
        return self._tab_Mz_
    
    @property
    def _tab_fstar(self):
        if not hasattr(self, '_tab_fstar_'):
            yy, xx = self._tab_Mz
            # Should be like tab_dndm
            self._tab_fstar_ = self.SFE(z=xx, Mh=yy)
        return self._tab_fstar_
    
    def _SAM(self, z, y):
        if self.pf['pop_sam_nz'] == 1:
            return self._SAM_1z(z, y)
        elif self.pf['pop_sam_nz'] == 2:
            return self._SAM_2z(z, y)
        else:
            raise NotImplementedError('No SAM with nz={}'.format(\
                self.pf['pop_sam_nz']))
                
    def _SAM_jac(self, z, y):
        if self.pf['pop_sam_nz'] == 1:
            return self._SAM_1z_jac(z, y)
        elif self.pf['pop_sam_nz'] == 2:
            return self._SAM_2z_jac(z, y)
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

        Mh, Mg, Mst, MZ, cMst, Mbh = y

        kw = {'z':z, 'Mh': Mh, 'Ms': Mst, 'Mg': Mg, 'MZ': MZ,
            'cMst': cMst, 'Mbh': Mbh}

        # Assume that MZ, Mg, and Mstell acquired *not* by smooth inflow
        # is same fraction of accreted mass as fractions in this halo
        # right now
        
        fb = self.cosm.fbar_over_fcdm
        
        # Convert from s/dz to yr/dz
        dtdz_s = -self.cosm.dtdz(z)
        dtdz = dtdz_s / s_per_yr
        
        # Splitting up the inflow. P = pristine.
        # Units = Msun / yr -> Msun / dz
        PIR = fb * self.MAR(z, Mh) * dtdz
        NPIR = fb * self.MDR(z, Mh) * dtdz
        
        # Measured relative to baryonic inflow
        Mb = fb * Mh
        Zfrac = self.pf['pop_acc_frac_metals'] * (MZ / Mb)
        Sfrac = self.pf['pop_acc_frac_stellar'] * (Mst / Mb)
        Gfrac = self.pf['pop_acc_frac_gas'] * (Mg / Mb)
        
        # Need SFR per dz
        if not self.pf['pop_star_formation']:
            fstar = SFR = 0.0
        elif self.pf['pop_sfr'] is None:
            fstar = self.SFE(**kw)
            SFR = PIR * fstar
        else:
            fstar = 1e-10
            SFR = self.sfr(**kw) * dtdz

        # "Quiet" mass growth
        fsmooth = self.fsmooth(**kw)

        # Eq. 1: halo mass.
        y1p = self.MGR(z, Mh) * dtdz

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

        # BH accretion rate
        if self.pf['pop_bh_formation']:
            if self.pf['pop_bh_facc'] is not None:
                y6p = self.pf['pop_bh_facc'] * PIR
            else:
                
                eta = self.pf['pop_eta']
                fduty = self.pf['pop_fduty']
                if Mbh > 0:
                    y6p = Mbh * dtdz_s * fduty * (1. - eta) / eta / t_edd
                else:
                    y6p = 0.0

        else:
            y6p = 0.0
        
        # Stuff to add: parameterize metal yield, metal escape, star formation
        # from reservoir? How to deal with Mmin(z)? Initial conditions (from PopIII)?
        
        results = [y1p, y2p, y3p, y4p, y5p, y6p]
        
        return np.array(results)
        
    def _SAM_1z_jac(self, z, y):
        """
        Jacobian for _SAM_1z
        """
    
        print('jac!', z, y)
    
        Mh, Mg, Mst, MZ, cMst, Mbh = y
    
        kw = {'z':z, 'Mh': Mh, 'Ms': Mst, 'Mg': Mg, 'MZ': MZ,
            'cMst': cMst, 'Mbh': Mbh}
    
        # Assume that MZ, Mg, and Mstell acquired *not* by smooth inflow
        # is same fraction of accreted mass as fractions in this halo
        # right now
    
        fb = self.cosm.fbar_over_fcdm
    
        # Convert from s/dz to yr/dz
        dtdz_s = -self.cosm.dtdz(z)
        dtdz = dtdz_s / s_per_yr
    
        # Splitting up the inflow. P = pristine.
        # Units = Msun / yr -> Msun / dz
        PIR = fb * self.MAR(z, Mh) * dtdz
        NPIR = fb * self.MDR(z, Mh) * dtdz
        #PIR = lambda _Mh: fb * self.MAR(z, _Mh) * dtdz
        #NPIR = lambda _Mh: fb * self.MDR(z, _Mh) * dtdz
        #
        # Measured relative to baryonic inflow
        Mb = fb * Mh
        Zfrac = self.pf['pop_acc_frac_metals'] * (MZ / Mb)
        Sfrac = self.pf['pop_acc_frac_stellar'] * (Mst / Mb)
        Gfrac = self.pf['pop_acc_frac_gas'] * (Mg / Mb)
    
        # Need SFR per dz
        #if not self.pf['pop_star_formation']:
        #    fstar = SFR = 0.0
        #elif self.pf['pop_sfr'] is None:
        #    fstar = lambda _Mh: self.SFE(z=kw['z'], Mh=_Mh)
        #    SFR = lambda _Mh: PIR(_Mh) * fstar(_Mh)
        #else:
        #    fstar = 1e-10
        #    SFR = lambda _Mh: self.sfr(z=kw['z'], Mh=_Mh) * dtdz
    
        # Need SFR per dz
        if not self.pf['pop_star_formation']:
            fstar = SFR = 0.0
        elif self.pf['pop_sfr'] is None:
            fstar = self.SFE(**kw)
            SFR = PIR * fstar
        else:
            fstar = 1e-10
            SFR = self.sfr(**kw) * dtdz
    
        # "Quiet" mass growth
        fsmooth = self.fsmooth(**kw)
    
        # Eq. 1: halo mass.
        _y1p = lambda _Mh: self.MGR(z, _Mh) * dtdz
        y1p = derivative(_y1p, Mh)
    
        # Eq. 2: gas mass
        if self.pf['pop_sfr'] is None:
            y2p = PIR * (1. - SFR/PIR) + NPIR * Gfrac
        else:
            y2p = PIR * (1. - fstar) + NPIR * Gfrac
            
        #_yp = lambda _Mh: self.MGR(z, _Mh) * dtdz
        #y2p = derivative(_yp2, Mh)
        
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
    
        # Add BHs
        if self.pf['pop_bh_formation']:
            if self.pf['pop_bh_facc'] is not None:
                y6p = self.pf['pop_bh_facc'] * PIR
            else:
    
                C = dtdz_s * 4.0 * np.pi * G * m_p / sigma_T / c
    
                if self.pf['pop_bh_seed_mass'] is not None:
                    Mseed = self.pf['pop_bh_seed_mass']    
                elif self.pf['pop_bh_seed_eff'] is not None:
                    Mseed = self.pf['pop_bh_seed_eff'] * Mg
                else:
                    Mseed = self.pf['pop_bh_seed_ratio'] * Mmin
    
                # Form new BHs
                if (Mh >= Mmin) and (Mbh == 0.0):
                    y6p = Mseed * C
                elif Mbh > 0:
                    # Eddington-limited growth. Remember Mbh is really
                    # just the accreted mass so we need to add in the seed mass.
                    y6p = C * (Mbh + Mseed)
                else:
                    y6p = 0.0
    
        else:
            y6p = 0.0
    
        # Remember that we're building a matrix. Columns:
        # [Mh, Mg, Mst, MZ, cMst, Mbh]
        # First, for testing, just do diagonal elements.
        results = [y1p, 0.0, 0.0, 0.0, 0.0, 0.0]
    
        return np.diag(results)    
    
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
    def constant_BHF(self):
        pass

    def ScalingRelations(self, z):
        """
        Return scaling relations at input redshift `z`.
        """
        
        # For a constant SFE, we're done.
        if self.constant_SFE:
            zarr, data = self.scaling_relations
        else:
            raise NotImplemented('')
        
        return data

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
            
            zf = max(float(self.halos.tab_z.min()), self.pf['final_redshift'])

            if self.pf['sam_dz'] is not None:
                zfreq = int(round(self.pf['sam_dz'] / np.diff(self.halos.tab_z)[0], 0))
            else:
                zfreq = 1

            zarr = self.halos.tab_z[self.halos.tab_z >= zf][::zfreq]
                        
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
        return self.Trajectories(M0)
        
    @property
    def _trajectories(self):
        if not hasattr(self, '_trajectories_'):
            raise AttributeError('Must set by hand or run `Trajectories`.')
        return self._trajectories_    
            
    @_trajectories.setter
    def _trajectories(self, value):
        self._trajectories_ = value
        
    def Trajectories(self, M0=0):
        """
        In this case, the formation time of a halo matters.
        
        Returns
        -------
        Dictionary of quantities, each having shape (z, z). 
        The first dimension corresponds to formation time, the second axis
        repreesents trajectories. So, e.g., to pick out all halo masses at a 
        given observed redshift (say z=6) you would do:
        
            zarr, data = self.Trajectories()
            k = np.argmin(np.abs(zarr - 6))
            Mh = data[:,k]
        
        """
        
        if hasattr(self, '_trajectories'):
            return self._trajectories
        
        
        keys = ['Mh', 'Mg', 'Ms', 'MZ', 'cMs', 'Mbh', 'SFR', 'SFE', 'MAR', 
            'Md', 'Sd', 'nh', 'Z', 't']
                
        zf = max(float(self.halos.tab_z.min()), self.zdead)
        zi = min(float(self.halos.tab_z.max()), self.zform)
        
        if self.pf['sam_dz'] is not None:
            zfreq = int(round(self.pf['sam_dz'] / np.diff(self.halos.tab_z)[0], 0))
        else:
            zfreq = 1

        in_range = np.logical_and(self.halos.tab_z >= zf, self.halos.tab_z <= zi)
        zarr = self.halos.tab_z[in_range][::zfreq]
        results = {key:np.zeros([zarr.size]*2) for key in keys}
                
        zmax = []
        zform = []
        for i, z in enumerate(zarr):
            #if (i == 0) or (i == len(zarr) - 1):
            #    zmax.append(zarr[i])
            #    zform.append(z)
            #    continue

            # If M0 is 0, assume it's the minimum mass at this redshift.
            _zarr, _results = self.RunSAM(z0=z, M0=M0)

            # Need to splice into the right elements of 2-D array.
            # SAM is run from zform to final_redshift, so only a subset
            # of elements in the 2-D table are filled.
            for key in keys:
                dat = _results[key].copy()
                k = np.argmin(abs(_zarr.min() - zarr))
                results[key][i,k:k+len(dat)] = dat
            
            zform.append(z)
            
            zmax.append(_results['zmax'])

        results['zmax'] = np.array(zmax)
        results['z'] = np.array(zform)
        
        self._trajectories = np.array(zform), results

        return np.array(zform), results
        
    def _ScalingRelationsStaticSFE(self, z0=None, M0=0):
        self.RunSAM(z0, M0)

    def Trajectory(self, z0=None, M0=0):
        """
        Just a wrapper around `RunSAM`.
        """
        return self.RunSAM(z0, M0)
        
    def RunSAM(self, z0=None, M0=0):
        """
        Evolve a halo from initial mass M0 at redshift z0 forward in time.
        
        .. note :: If M0 is not supplied, we'll assume it's Mmin(z0).
        
        Parameters
        ----------
        z0 : int, float
            Formation redshift.
        M0 : int, float
            Formation mass (total halo mass).
        
        Returns
        -------
        redshifts, halo mass, gas mass, stellar mass, metal mass
        
        """

        zf = max(float(self.halos.tab_z.min()), self.zdead)

        if self.pf['sam_dz'] is not None:
            dz = self.pf['sam_dz']
            zfreq = int(round(self.pf['sam_dz'] / np.diff(self.halos.tab_z)[0], 0))
        else:
            dz = np.diff(self.halos.tab_z)[0]
            zfreq = 1

        # jac=self._SAM_jac
        solver = ode(self._SAM).set_integrator('lsoda', 
            nsteps=1e4, atol=self.pf['sam_atol'], rtol=self.pf['sam_rtol'],
            with_jacobian=False)
        
        # Criteria used to kill a population.    
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

        n0 = 0.0

        # Our results don't depend on this, unless SFE depends on z
        if (z0 is None) and (M0 == 0):
            z0 = self.halos.tab_z.max()
            M0 = self._tab_Mmin[-1]
        elif (M0 <= 1):
            
            # If we're treating a continuum of halos.
            M0 = np.interp(z0, self.halos.tab_z, self._tab_Mmin)            

            iz = np.argmin(np.abs(z0 - self.halos.tab_z))

            if np.allclose(z0, self.halos.tab_z[iz]):
                n0 = self._tab_n_Mmin[iz]
            else:
                print('hay problemas!')
            
        elif (M0 > 1):
            if z0 >= self.pf['initial_redshift']:
                M0 = np.interp(z0, self.halos.tab_z, M0 * self._tab_Mmin)
            else:
                M0 = np.interp(z0, self.halos.tab_z, self._tab_Mmin)

        in_range = np.logical_and(self.halos.tab_z >= zf, self.halos.tab_z <= z0)
        zarr = self.halos.tab_z[in_range][::zfreq]
        Nz = zarr.size

        # Boundary conditions (pristine halo)
        Mg0 = self.cosm.fbar_over_fcdm * M0
        MZ0 = 0.0
        Mst0 = 0.0
        Mbh0 = 0.0
        seeded = False
        
        # Initial stellar mass -> 0, initial halo mass -> Mmin
        solver.set_initial_value(np.array([M0, Mg0, Mst0, MZ0, Mst0, Mbh0]), z0)

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
        Mbh_t = []
        sfr_t = []
        sfe_t = []
        mar_t = []
        nh_t = []
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
            sfr_t.append(self.SFR(z=redshifts[-1], Mh=Mh_t[-1]))
            mar_t.append(self.MGR(redshifts[-1], Mh_t[-1]))
            nh_t.append(n0)
            
            Mmin = np.interp(redshifts[-1], self.halos.tab_z, self._tab_Mmin)
                        
            if self.pf['pop_bh_seed_mass'] is not None:
                Mseed = self.pf['pop_bh_seed_mass']    
            elif self.pf['pop_bh_seed_eff'] is not None:
                Mseed = self.pf['pop_bh_seed_eff'] * Mg
            else:
                Mseed = self.pf['pop_bh_seed_ratio'] * Mmin
                
            # Form new BHs
            if (Mh_t[-1] >= Mmin) and (not seeded):
                Mbh_t.append(Mseed)   
                # Update solver position. 
                pos = np.array([Mh_t[-1], Mg_t[-1], Mst_t[-1], metals[-1], cMst_t[-1], Mseed])
                solver.set_initial_value(pos, redshifts[-1])
                seeded = True
            elif (not seeded):
                Mbh_t.append(0.0)
            else:
                Mbh_t.append(solver.y[5])
            
            if 'sfe' in self.pf['pop_sfr_model']:
                sfe_t.append(self.SFE(z=redshifts[-1], Mh=Mh_t[-1]))
            
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
                Mtemp = self.halos.VirialMass(z, self.pf['pop_temp_limit'])
                                
                if solver.y[0] >= Mtemp:
                    zmax_T = np.interp(Mtemp, Mh_t[-2:], redshifts[-2:])
                    
            if has_e_limit and (zmax_e is None):
                
                Eblim = self.pf['pop_bind_limit']
                Ebnow = self.halos.BindingEnergy(redshifts[-1], Mh_t[-1])
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
            zmax = self.zdead

        # Everything will be returned in order of ascending redshift,
        # which will mean masses are (probably) declining from 0:-1
        z = np.array(redshifts)[-1::-1]
        Mh = np.array(Mh_t)[-1::-1]
        Mg = np.array(Mg_t)[-1::-1]
        Ms = np.array(Mst_t)[-1::-1]
        MZ = np.array(metals)[-1::-1]
        Md = self.pf['pop_dust_yield'] * MZ
        Rd = self.pf['pop_dust_scale']
        Sd = Md * g_per_msun / 4. / np.pi / (Rd * cm_per_kpc)**2
        
        if self.pf['pop_dust_yield'] > 0:
            tau = self.dust_kappa(wave=1600.)
        else:
            tau = None
        
        cMs = np.array(cMst_t)[-1::-1]
        Mbh = np.array(Mbh_t)[-1::-1]
        SFR = np.array(sfr_t)[-1::-1]
        SFE = np.array(sfe_t)[-1::-1]
        MAR = np.array(mar_t)[-1::-1]
        nh = np.array(nh_t)[-1::-1]
        tlb = np.array(lbtime)[-1::-1]

        # Derived
        results = {'Mh': Mh, 'Mg': Mg, 'Ms': Ms, 'MZ': MZ, 'Md': Md, 'cMs': cMs,
            'Mbh': Mbh, 'SFR': SFR, 'SFE': SFE, 'MAR': MAR, 'nh': nh, 
            'Sd': Sd, 'tau': tau, 'zmax': zmax, 't': tlb}
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
            erg_per_phot = self._get_energy_per_photon(Emin, Emax) * erg_per_ev
                               
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
            
            return np.interp(self.halos.tab_z, zarr, Mmin)
        
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
                        
        return np.interp(self.halos.tab_z, zarr, Mmin[best])
        
    def save(self, prefix=None, fn=None, fmt='npz'):
        """
        Save properties of the population.
        """
        pass
        
