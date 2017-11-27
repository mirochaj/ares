"""

Population.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu May 28 13:59:41 MDT 2015

Description: 

"""

from ..physics import Cosmology
from ..util import ParameterFile
from ..physics.Constants import E_LyA, E_LL

_multi_pop_error_msg = "Parameters for more than one population detected! "
_multi_pop_error_msg += "Population objects are by definition for single populations."

class Population(object):
    def __init__(self, grid=None, **kwargs):

        # why is this necessary?
        if 'problem_type' in kwargs:
            del kwargs['problem_type']

        self.pf = ParameterFile(**kwargs)

        assert self.pf.Npops == 1, _multi_pop_error_msg
        
        self.grid = grid

        self.zform = self.pf['pop_zform']
        self.zdead = self.pf['pop_zdead']

    @property
    def id_num(self):
        if not hasattr(self, '_id_num'):
            self._id_num = None
        return self._id_num

    @id_num.setter
    def id_num(self, value):
        self._id_num = int(value)    
    
    @property
    def cosm(self):
        if not hasattr(self, '_cosm'):    
            if self.grid is None:
                self._cosm = Cosmology(
                    omega_m_0=self.pf['omega_m_0'], 
                    omega_l_0=self.pf['omega_l_0'], 
                    omega_b_0=self.pf['omega_b_0'],  
                    hubble_0=self.pf['hubble_0'],  
                    helium_by_number=self.pf['helium_by_number'],
                    cmb_temp_0=self.pf['cmb_temp_0'],
                    approx_highz=self.pf['approx_highz'],
                    sigma_8=self.pf['sigma_8'],
                    primordial_index=self.pf['primordial_index'])
            else:
                self._cosm = grid.cosm
                
        return self._cosm
        
    @property
    def zone(self):
        if not hasattr(self, '_zone'):
            if self.affects_cgm and (not self.affects_igm):
                self._zone = 'cgm'
            elif self.affects_igm and (not self.affects_cgm):
                self._zone = 'igm'
            else:
                raise ValueError("Populations should only affect one zone!")
                
        return self._zone    
        
    @property
    def affects_cgm(self):
        if not hasattr(self, '_affects_cgm'):
            self._affects_cgm = self.is_src_ion_cgm 
        return self._affects_cgm
    
    @property
    def affects_igm(self):
        if not hasattr(self, '_affects_igm'):
            self._affects_igm = self.is_src_ion_igm or self.is_src_heat_igm
        return self._affects_igm    
    
    @property
    def is_src_lya(self):
        if not hasattr(self, '_is_src_lya'):
            if self.pf['pop_sed_model']:
                self._is_src_lya = \
                    (self.pf['pop_Emin'] <= 10.2 <= self.pf['pop_Emax']) \
                    and self.pf['pop_lya_src']
            else:
                self._is_src_lya = self.pf['pop_lya_src']
    
        return self._is_src_lya
    
    @property
    def is_src_lya_fl(self):
        if not hasattr(self, '_is_src_lya_fl'):
            self._is_src_lya_fl = False
            if not self.is_src_lya:
                pass
            else:
                if self.pf['pop_lya_fl'] and self.pf['include_lya_fl']:
                    self._is_src_lya_fl = True
    
        return self._is_src_lya_fl
    
    @property
    def is_src_ion_cgm(self):
        if not hasattr(self, '_is_src_ion_cgm'):
            if self.pf['pop_sed_model']:
                self._is_src_ion_cgm = \
                    (self.pf['pop_Emax'] > E_LL) \
                    and self.pf['pop_ion_src_cgm']
            else:
                self._is_src_ion_cgm = self.pf['pop_ion_src_cgm']        
    
        return self._is_src_ion_cgm
        
    @property
    def is_src_ion_igm(self):
        if not hasattr(self, '_is_src_ion_igm'):
            if self.pf['pop_sed_model']:
                self._is_src_ion_igm = \
                    (self.pf['pop_Emax'] > E_LL) \
                    and self.pf['pop_ion_src_igm']
            else:
                self._is_src_ion_igm = self.pf['pop_ion_src_igm']        
    
        return self._is_src_ion_igm
        
    @property
    def is_src_ion(self):
        if not hasattr(self, '_is_src_ion'):    
            self._is_src_ion = self.is_src_ion_cgm #or self.is_src_ion_igm
        return self._is_src_ion
        
    @property
    def is_src_ion_fl(self):
        if not hasattr(self, '_is_src_ion_fl'):
            self._is_src_ion_fl = False
            if not self.is_src_ion:
                pass
            else:
                if self.pf['pop_ion_fl'] and self.pf['include_ion_fl']:
                    self._is_src_ion_fl = True
    
        return self._is_src_ion_fl    
        
    @property
    def is_src_heat(self):
        return self.is_src_heat_igm
        
    @property
    def is_src_heat_igm(self):
        if not hasattr(self, '_is_src_heat_igm'):
            if self.pf['pop_sed_model']:
                self._is_src_heat_igm = \
                    (E_LL <= self.pf['pop_Emin']) \
                    and self.pf['pop_heat_src_igm']
            else:
                self._is_src_heat_igm = self.pf['pop_heat_src_igm']        
    
        return self._is_src_heat_igm
        
    @property
    def is_src_heat_fl(self):
        if not hasattr(self, '_is_src_heat_fl'):
            self._is_src_heat_fl = False
            if not self.is_src_heat:
                pass
            else:
                if self.pf['pop_temp_fl'] and self.pf['include_temp_fl']:
                    self._is_src_heat_fl = True
    
        return self._is_src_heat_fl
    
    @property
    def is_src_uv(self):
        # Delete this eventually but right now doing so will break stuff
        if not hasattr(self, '_is_src_uv'):
            if self.pf['pop_sed_model']:
                self._is_src_uv = \
                    (self.pf['pop_Emax'] > E_LL) \
                    and self.pf['pop_ion_src_cgm']
            else:
                self._is_src_uv = self.pf['pop_ion_src_cgm']        
    
        return self._is_src_uv
        
    @property
    def is_src_lya(self):
        if not hasattr(self, '_is_src_lya'):
            if self.pf['pop_sed_model']:
                self._is_src_lya = \
                    (self.pf['pop_Emin'] <= 10.2 <= self.pf['pop_Emax']) \
                    and self.pf['pop_lya_src']
            else:
                return self.pf['pop_lya_src']
    
        return self._is_src_lya
    
    @property
    def is_src_uv(self):
        if not hasattr(self, '_is_src_uv'):
            if self.pf['pop_sed_model']:
                self._is_src_uv = \
                    (self.pf['pop_Emax'] > E_LL) \
                    and self.pf['pop_ion_src_cgm']
            else:
                self._is_src_uv = self.pf['pop_ion_src_cgm']        
    
        return self._is_src_uv    
    
    @property
    def is_src_xray(self):
        if not hasattr(self, '_is_src_xray'):
            if self.pf['pop_sed_model']:
                self._is_src_xray = \
                    (E_LL <= self.pf['pop_Emin']) \
                    and self.pf['pop_heat_src_igm']
            else:
                self._is_src_xray = self.pf['pop_heat_src_igm']        
        
        return self._is_src_xray    

    @property
    def is_src_lw(self):
        if not hasattr(self, '_is_src_lw'):
            if not self.pf['radiative_transfer']:
                self._is_src_lw = False
            elif not self.pf['pop_lw_src']:
                self._is_src_lw = False
            elif self.pf['pop_sed_model']:
                self._is_src_lw = \
                    (self.pf['pop_Emin'] <= 11.2 <= self.pf['pop_Emax']) and \
                    (self.pf['pop_Emin'] <= E_LL <= self.pf['pop_Emax'])
            else:
                raise NotImplementedError('help')
    
        return self._is_src_lw    

    @property
    def is_emissivity_separable(self):
        """
        Are the frequency and redshift-dependent components independent?
        """
        return True
    
    @property
    def is_emissivity_scalable(self):
        """
        Can we just determine a luminosity density by scaling the SFRD?
    
        The answer will be "no" for any population with halo-mass-dependent
        values for photon yields (per SFR), escape fractions, or spectra.
        """
        
        if not hasattr(self, '_is_emissivity_scalable'):
            self._is_emissivity_scalable = True
    
            if self.pf.Npqs == 0:
                return self._is_emissivity_scalable
    
            for par in self.pf.pqs:
    
                # If this is the only Mh-dep parameter, we're still scalable.
                if par == 'pop_fstar':
                    continue
    
                if type(self.pf[par]) is str:
                    self._is_emissivity_scalable = False
                    break
    
                for i in xrange(self.pf.Npqs):
                    pn = '{0!s}[{1}]'.format(par,i)
                    if pn not in self.pf:
                        continue
    
                    if type(self.pf[pn]) is str:
                        self._is_emissivity_scalable = False
                        break
    
        return self._is_emissivity_scalable
    
        