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
        
        # Fetch star formation efficiency. If xi_* kwargs are passed, must
        # 'undo' this as it will be applied later.
        if 'fstar' in kwargs:
            for i in range(3):
                pf['pop_fstar{%i}' % i] = kwargs['fstar']
        else:
            for i in range(3):
                if 'pop_fstar' in pf:
                    pf['pop_fstar{%i}' % i] = pf['pop_fstar']
                else:
                    pf['pop_fstar{%i}' % i] = fstar_default    
                
        if 'fesc' in kwargs:
            pf['pop_fesc{2}'] = kwargs['fesc']
        else:
            pf['pop_fesc{2}'] = fesc_default
        
        if ('Nlw' in kwargs) or ('xi_LW' in kwargs):
            y = kwargs['Nlw'] if ('Nlw' in kwargs) else kwargs['xi_LW']
            
            if 'xi_LW' in kwargs:
                y /= pf['pop_fstar{0}']
                
            pf['pop_yield{0}'] = y    
            pf['pop_yield_units{0}'] = 'photons/baryon'
            
        if ('Nion' in kwargs) or ('xi_UV' in kwargs):
            y = kwargs['Nion'] if ('Nion' in kwargs) else kwargs['xi_UV']
            
            if 'xi_UV' in kwargs:
                y /= pf['pop_fstar{2}'] * pf['pop_fesc{2}']
            
            pf['pop_yield{2}'] = y     
            pf['pop_yield_units{2}'] = 'photons/baryon'    
        
        # Lx-SFR
        if 'cX' in kwargs:
            yield_X = kwargs['cX']
            if 'fX' in kwargs:
                yield_X *= kwargs['fX']
            
            pf['pop_yield{1}'] = yield_X 
            
        elif 'fX' in kwargs:        
            pf['pop_yield{1}'] = kwargs['fX'] * kwargs['pop_yield{1}']    
            
        elif 'xi_XR' in kwargs:
            pf['pop_yield{1}'] = kwargs['xi_XR'] * kwargs['pop_yield{1}'] \
               / pf['pop_fstar{1}']
        
    else:
        pf = {}
        
    return pf
