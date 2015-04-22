"""

DerivedQuantities.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Feb 12 14:13:19 MST 2015

Description: 

"""

import re
import numpy as np
from ..physics import Cosmology
from ..physics.Constants import nu_0_mhz

def total_heat(data):
    """
    Convert heating rate coefficients to a total heating rate using
    number densities of relevant ion species.

    Returns
    -------
    Total heating in units of [erg / s / cMpc**3]
    """

    pf = data['pf']
    names = data.keys()

    heat = np.zeros_like(data['z'])
    for i, sp in enumerate(['h_1', 'he_1', 'he_2']):

        if i > 0 and 'igm_%s' % sp not in names:
            continue

        for k in range(pf.Npops):

            if pf.Npops == 1:
                suffix = ''
            else:
                suffix = '{%i}' % k

            heat_by_pop = 'igm_heat_%s%s' % (sp, suffix)

            if heat_by_pop not in names:
                continue

            n = data['igm_n_%s' % sp]

            # Multiply by number density of absorbers
            heat += n * data[heat_by_pop]

    return heat
    
def total_gamma(data, sp):

    pf = data['pf']
    n1 = data['igm_n_%s' % sp]

    gamma = np.zeros_like(data['z'])
    for donor in ['h_1', 'he_1', 'he_2']:
        
        if 'igm_n_%s' % donor not in data:
            continue
            
        for k in range(pf.Npops):
        
            if pf.Npops == 1:
                suffix = ''
            else:
                suffix = '{%i}' % k
        
            gamma_by_donor_by_pop = 'igm_gamma_%s_%s%s' % (sp, donor, suffix)
            
            if gamma_by_donor_by_pop not in data:
                continue
                
            n2 = data['igm_n_%s' % donor] 
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
    
# State quantities
registry_state_Q = \
{
 #'nu': lambda data: nu_0_mhz / (1. + data['z']),
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

class DerivedQuantities:
    def __init__(self, data, pf):

        self.data = data

        try:
            self.cosm = data.cosm
        except AttributeError:
            self.cosm = Cosmology()

        # Stuff we might need (cosmology)
        self.data['Tcmb'] = self.cosm.TCMB(self.data['z'])
        self.data['nH'] = self.cosm.nH(self.data['z'])
        self.data['nHe'] = self.cosm.nHe(self.data['z'])
        self.data['t'] = 2. / 3. \
            / np.array(map(self.cosm.HubbleParameter, self.data['z']))
        self.data['logt'] = np.log10(self.data['t'])
        
        self.data['pf'] = pf

        self.derived_quantities = {}

        # Create frequency array -- be careful about preserving the mask
        self.derived_quantities['nu'] = self.data['nu'] = \
            nu_0_mhz / (1. + self.data['z'])

        mask = np.logical_not(np.isfinite(self.data['z']))
        self.data['nu'][mask] = np.inf
        self.derived_quantities['nu'][mask] = np.inf

        self.build(**registry_state_Q)
        self.build(**registry_rate_Q)

        del self.data['pf']

    def build(self, **registry):
        """
        Go through the registry and calculate everything we can.
        """
        
        for item in registry:
                        
            if item in self.data:
                continue

            try:
                self.data[item] = registry[item](self.data)
                self.derived_quantities[item] = self.data[item]
            except:
                pass
    
