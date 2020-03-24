"""

DerivedQuantities.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Feb 12 14:13:19 MST 2015

Description: 

"""

import re, time
import numpy as np
from ..physics import Cosmology
from collections import namedtuple
from ..physics.Constants import nu_0_mhz

def total_heat(data):
    """
    Convert heating rate coefficients to a total heating rate using
    number densities of relevant ion species.

    Returns
    -------
    Total heating in units of [erg / s / cMpc**3]
    """

    pf = data.pf

    heat = np.zeros_like(data['z'])
    for i, sp in enumerate(['h_1', 'he_1', 'he_2']):

        if i > 0 and (data['igm_{!s}'.format(sp)] is None):
            continue

        for k in range(pf.Npops):

            if pf.Npops == 1:
                suffix = ''
            else:
                suffix = '{{{}}}'.format(k)

            heat_by_pop = 'igm_heat_{0!s}{1!s}'.format(sp, suffix)

            if data[heat_by_pop] is None:
                continue

            n = data['igm_n_{!s}'.format(sp)]

            # Multiply by number density of absorbers, convert to 
            # co-moving units
            heat += n * data[heat_by_pop] / (1. + data['z'])**3

    return heat
    
def total_gamma(data, sp):

    pf = data.pf
    n1 = data['igm_n_{!s}'.format(sp)]

    gamma = np.zeros_like(data['z'])
    for donor in ['h_1', 'he_1', 'he_2']:
        
        if data['igm_n_{!s}'.format(donor)] is None:
            continue
            
        for k in range(pf.Npops):
        
            if pf.Npops == 1:
                suffix = ''
            else:
                suffix = '{{{}}}'.format(k)
        
            gamma_by_donor_by_pop =\
                'igm_gamma_{0!s}_{1!s}{2!s}'.format(sp, donor, suffix)
            
            if data[gamma_by_donor_by_pop] is None:
                continue
                
            n2 = data['igm_n_{!s}'.format(donor)] 
            gamma += data[gamma_by_donor_by_pop] * n2 / n1
    
    return gamma
    
def total_sfrd(data):
    
    if 'sfrd' in data:
        return data['sfrd']
    
    tot = 0.0
    for key in data:
        if re.search('sfrd', key):
            tot += data[key]
    
    return tot
    
# Simple things
registry_cosm_Q = \
{
 'Tcmb': lambda data, cosm: cosm.TCMB(data['z']),
 'nH': lambda data, cosm: cosm.nH(data['z']),
 'nHe': lambda data, cosm: cosm.nHe(data['z']), 
 't': lambda data, cosm: 2. / 3. \
    / np.array(list(map(cosm.HubbleParameter, data['z']))),
 'logt': lambda data, cosm: np.log10(data['t']),
}    
    
# State quantities
registry_state_Q = \
{
 'nu': lambda data: nu_0_mhz / (1. + data['z']),
 'dTb': lambda data: data['igm_dTb'],
 'igm_dTb': lambda data: data['dTb'], # axe eventually
 'contrast': lambda data: 1. - data['Tcmb'] / data['Ts'],
 'igm_h_2': lambda data: 1. - data['igm_h_1'],
 'igm_he_3': lambda data: 1. - data['igm_he_1'] - data['igm_he_2'],
 'igm_n_h_1': lambda data: data['nH'] * data['igm_h_1'],
 'igm_n_h_2': lambda data: data['nH'] * data['igm_h_2'],
 'igm_n_he_1': lambda data: data['nHe'] * data['igm_he_1'],
 'igm_n_he_2': lambda data: data['nHe'] * data['igm_he_2'],
 'igm_n_he_3': lambda data: data['nHe'] * data['igm_he_3'],
 'de': lambda data: data['nH'] * data['igm_e'],
}

registry_rate_Q = \
{
 'xavg': lambda data: data['cgm_h_2'] \
      + (1. - data['cgm_h_2']) * data['igm_h_2'],
 'igm_heat': lambda data: total_heat(data),
 'igm_gamma_h_1': lambda data: total_gamma(data, 'h_1'),
 'igm_gamma_he_1': lambda data: total_gamma(data, 'he_1'),
 'igm_gamma_he_2': lambda data: total_gamma(data, 'he_2'),
}

registry_special_Q = \
{
 'Jx': lambda data: 0.0,
 'Jlw': lambda data: 0.0,
 'sfrd': lambda data: total_sfrd(data),
}

class DerivedQuantities(object):
    def __init__(self, ModelSet=None, pf=None, data=None, cosm=None):
        self._ms = ModelSet
        self._cosm_ = cosm
        
        if pf is None:
            if ModelSet is not None:
                self.pf = ModelSet.pf
            else:
                self.pf = namedtuple
                self.pf.Npops = 1
        else:
            self.pf = pf
                
        self._data = {}
        
        if data is not None:
            self._data.update(data)
        
    @property
    def cosm(self):
        if not hasattr(self, '_cosm'):
            if self._cosm_ is not None:
                self._cosm = self._cosm_
            else:    
                try:
                    self._cosm = self._ms.cosm
                except AttributeError:
                    self._cosm = Cosmology(**self.pf)   
        return self._cosm_
                             
    @property
    def _shape(self):
        if not hasattr(self, '__shape'):
            # Number of samples x number of redshifts
            self.__shape = list(self._ms.blobs.shape[:-1])
        return self.__shape
        
    def _add_data(self, data):
        self._data.update(data)    
        
    def __name__(self):
        return 'DerivedQuantities'
        
    def __getitem__(self, name):
        if name in self._data:
            return self._data[name]
            
        if self._ms is None:
            return self._get_item(name)
        else:
            return self._get_mset_item(name)

    def _get_item(self, name):
        if name in registry_cosm_Q:
            self._data[name] = registry_cosm_Q[name](self, self.cosm)
        elif name in registry_state_Q:
            self._data[name] = registry_state_Q[name](self)
        elif name in registry_rate_Q:
            self._data[name] = registry_rate_Q[name](self)
        elif name in registry_special_Q:
            raise NotImplemented('havent done registry_special_Q yet.')
        else:
            return None
            #raise ValueError(('Unrecognized derived quantity: {!s}').format(\
            #    name))
        
        return self._data[name]
        
    def _get_mset_item(self, name):

        # Why is this so expensive!?
        if name == 'z': 
            z = np.zeros(self._shape)
            for i, redshift in enumerate(self._ms.blob_redshifts):
                z[:,i] = self._ms.extract_blob('z', redshift)

            self._data['z'] = np.array(z)

        elif name == 'nu':

            # Create frequency array -- be careful about preserving the mask
            self._data['nu'] = nu_0_mhz / (1. + self['z'])
            
            mask = np.logical_not(np.isfinite(self['z']))
            self._data['nu'][mask] = np.inf
          
        elif name in self._ms.blob_names:
            j = self._ms.blob_names.index(name)
            self._data[name] = self._ms.blobs[:,:,j]
          
        # Simple quantities that depend on redshift and cosmology
        elif name in registry_cosm_Q:
            self._data[name] = registry_cosm_Q[name](self, self.cosm)
        
        # Simple derived quantities
        elif name in registry_state_Q:
            self._data[name] = registry_state_Q[name](self)
        elif name in registry_rate_Q:
            
            self._data[name] = registry_rate_Q[name](self)
        elif name in registry_special_Q:
            raise NotImplemented('havent done registry_special_Q yet.')
        else:
            return None
            #raise ValueError('Unrecognized derived quantity: {!s}'.format(\
            #    name))    
            
        return self._data[name]
                
            
