"""

BackwardCompatibility.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Jul 10 15:20:12 MDT 2015

Description:

"""

import re
from .SetDefaultParameterValues import PopulationParameters
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

pop_pars = PopulationParameters()

fesc_default = pop_pars['pop_fesc']
fstar_default = pop_pars['pop_fstar']

def par_supplied(var, **kwargs):
    if var in kwargs:
        if kwargs[var] is None:
            return False
        return True
    return False

def backward_compatibility(ptype, **kwargs):
    """
    Handle some conventions used in the pre "pop_*" parameter days.

    .. note :: Only applies to simple global 21-cm models right now, i.e.,
        problem_type=101, ParameterizedQuantity parameters, and the
        pop_yield vs. pop_rad_yield change.

    Parameters
    ----------
    ptype : int, float
        Problem type.

    Returns
    -------
    Dictionary of parameters to subsequently be updated.

    """

    pf = {}

    if ptype == 101:
        pf = {}

        if par_supplied('Tmin', **kwargs):
            for i in range(3):
                pf['pop_Tmin{{{}}}'.format(i)] = kwargs['Tmin']

        if par_supplied('Mmin', **kwargs):
            assert not par_supplied('Tmin'), "Must only supply Tmin OR Mmin!"
            for i in range(3):
                pf['pop_Mmin{{{}}}'.format(i)] = kwargs['Mmin']
                pf['pop_Tmin{{{}}}'.format(i)] = None

        # Fetch star formation efficiency. If xi_* kwargs are passed, must
        # 'undo' this as it will be applied later.
        if par_supplied('fstar', **kwargs):
            for i in range(3):
                pf['pop_fstar{{{}}}'.format(i)] = kwargs['fstar']

        if par_supplied('fesc', **kwargs):
            print('what')
            pf['pop_fesc{2}'] = kwargs['fesc']
        elif par_supplied('pop_fesc{2}'):
            print('the')
            pf['pop_fesc{2}'] = kwargs['pop_fesc{2}']
        else:
            print('fuck')
            pf['pop_fesc{2}'] = fesc_default

        if par_supplied('Nlw', **kwargs) or par_supplied('xi_LW', **kwargs):
            y = kwargs['Nlw'] if par_supplied('Nlw', **kwargs) else kwargs['xi_LW']

            if par_supplied('xi_LW', **kwargs):
                y /= pf['pop_fstar{0}']

            pf['pop_rad_yield{0}'] = y
            pf['pop_rad_yield_units{0}'] = 'photons/baryon'

        if par_supplied('Nion', **kwargs) or par_supplied('xi_UV', **kwargs):
            y = kwargs['Nion'] if par_supplied('Nion', **kwargs) else kwargs['xi_UV']

            if par_supplied('xi_UV', **kwargs):
                y /= pf['pop_fstar{2}'] * pf['pop_fesc{2}']

            pf['pop_rad_yield{2}'] = y
            pf['pop_rad_yield_units{2}'] = 'photons/baryon'

        # Lx-SFR
        if par_supplied('cX', **kwargs):
            yield_X = kwargs['cX']
            if par_supplied('fX', **kwargs):
                yield_X *= kwargs['fX']

            pf['pop_rad_yield{1}'] = yield_X

        elif par_supplied('fX', **kwargs):
            pf['pop_rad_yield{1}'] = kwargs['fX'] * kwargs['pop_rad_yield{1}']

        elif par_supplied('xi_XR', **kwargs):
            pf['pop_rad_yield{1}'] = kwargs['xi_XR'] * kwargs['pop_rad_yield{1}'] \
               / pf['pop_fstar{1}']

    fixes = {}
    for element in kwargs:

        if re.search('pop_yield', element):
            fixes[element.replace('pop_yield', 'pop_rad_yield')] = \
                kwargs[element]
            continue

        if element[0:3] == 'php':

            if isinstance(kwargs[element], basestring):
                fixes[element.replace('php', 'pq')] = \
                    kwargs[element].replace('php', 'pq')
            else:
                fixes[element.replace('php', 'pq')] = kwargs[element]

        if isinstance(kwargs[element], basestring):
            if kwargs[element][0:3] == 'php':
                fixes[element] = kwargs[element].replace('php', 'pq')

            if kwargs[element] == 'mass':
                fixes[element] = 'Mh'

    pf.update(fixes)

    return pf
