"""

BackwardCompatibility.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Jul 10 15:20:12 MDT 2015

Description: 

"""

from .SetDefaultParameterValues import PopulationParameters

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
        problem_type=101.
        
    Parameters
    ----------
    ptype : int, float
        Problem type.
    
    Returns
    -------
    Dictionary of parameters to subsequently be updated.
    
    """
    
    if ptype == 101:
        pf = {}
        
        if par_supplied('Tmin'):
            for i in range(3):
                pf['pop_Tmin{%i}' % i] = kwargs['Tmin']
        
        if par_supplied('Mmin'):
            assert not par_supplied('Tmin'), "Must only supply Tmin OR Mmin!"
            for i in range(3):
                pf['pop_Mmin{%i}' % i] = kwargs['Mmin']        
                pf['pop_Tmin{%i}' % i] = None
            
        # Fetch star formation efficiency. If xi_* kwargs are passed, must
        # 'undo' this as it will be applied later.
        if par_supplied('fstar'):
            for i in range(3):
                pf['pop_fstar{%i}' % i] = kwargs['fstar']
        else:
            for i in range(3):
                if 'pop_fstar' in pf:
                    pf['pop_fstar{%i}' % i] = pf['pop_fstar']
                else:
                    pf['pop_fstar{%i}' % i] = fstar_default    
                
        if par_supplied('fesc'):
            pf['pop_fesc{2}'] = kwargs['fesc']
        else:
            pf['pop_fesc{2}'] = fesc_default
        
        if par_supplied('Nlw') or par_supplied('xi_LW'):
            y = kwargs['Nlw'] if par_supplied('Nlw') else kwargs['xi_LW']
            
            if par_supplied('xi_LW'):
                y /= pf['pop_fstar{0}']
                
            pf['pop_yield{0}'] = y    
            pf['pop_yield_units{0}'] = 'photons/baryon'
            
        if par_supplied('Nlw') or par_supplied('xi_UV'):
            y = kwargs['Nion'] if par_supplied('Nion') else kwargs['xi_UV']
            
            if par_supplied('xi_UV'):
                y /= pf['pop_fstar{2}'] * pf['pop_fesc{2}']
            
            pf['pop_yield{2}'] = y     
            pf['pop_yield_units{2}'] = 'photons/baryon'    
        
        # Lx-SFR
        if par_supplied('cX'):
            yield_X = kwargs['cX']
            if par_supplied('fX'):
                yield_X *= kwargs['fX']
            
            pf['pop_yield{1}'] = yield_X 
            
        elif par_supplied('fX'):
            pf['pop_yield{1}'] = kwargs['fX'] * kwargs['pop_yield{1}']    
            
        elif par_supplied('xi_XR'):
            pf['pop_yield{1}'] = kwargs['xi_XR'] * kwargs['pop_yield{1}'] \
               / pf['pop_fstar{1}']
        
    else:
        pf = {}
        
    return pf
