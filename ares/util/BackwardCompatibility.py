"""

BackwardCompatibility.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Jul 10 15:20:12 MDT 2015

Description: 

"""

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
        
        if 'Tmin' in kwargs:
            for i in range(3):
                pf['pop_Tmin{%i}' % i] = kwargs['Tmin']
        
        if 'Mmin' in kwargs:
            assert 'Tmin' not in kwargs, "Must only supply Tmin OR Mmin!"
            for i in range(3):
                pf['pop_Mmin{%i}' % i] = kwargs['Mmin']        
                pf['pop_Tmin{%i}' % i] = None
            
        if 'fstar' in kwargs:
            for i in range(3):
                pf['pop_fstar{%i}' % i] = kwargs['fstar']
        
        if 'fesc' in kwargs:
            pf['pop_fesc{2}'] = kwargs['fesc']
        
        if ('Nlw' in kwargs) or ('xi_LW' in kwargs):
            y = kwargs['Nlw'] if ('Nlw' in kwargs) else kwargs['xi_LW']
            pf['pop_yield{0}'] = y
            pf['pop_yield_units{0}'] = 'photons/baryon'
        
        if ('Nion' in kwargs) or ('xi_UV' in kwargs):
            y = kwargs['Nion'] if ('Nion' in kwargs) else kwargs['xi_UV']
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
            pf['pop_yield{1}'] = kwargs['xi_XR'] * kwargs['pop_yield{1}']    
        
    else:
        pf = {}
        
    return pf
