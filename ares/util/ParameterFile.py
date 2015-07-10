"""

ParameterFile.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Nov 28 16:19:26 CST 2014

Description: 

"""

import re
from .ProblemTypes import ProblemType
from .SetDefaultParameterValues import SetAllDefaults
from .CheckForParameterConflicts import CheckForParameterConflicts

def backwards_compatibility(ptype, **kwargs):
    """
    Handle some conventions used in the pre "pop_*" parameter days.
    
    .. note :: Only applies to simple global 21-cm models right now, i.e., 
        problem_type=100.
        
    Parameters
    ----------
    ptype : int, float
        Problem type.
    
    Returns
    -------
    Dictionary of parameters to subsequently be updated.
    
    """
    
    if ptype == 100:
        pf = {}
        
        # LW/UV
        if 'fstar' in kwargs:
            pf['pop_fstar{0}'] = kwargs['fstar']
            pf['pop_fstar{1}'] = kwargs['fstar']
        
        if 'fesc' in kwargs:
            pf['pop_fesc{0}'] = kwargs['fesc']
        
        if 'Nion' in kwargs:
            pf['pop_yield{0}'] = kwargs['Nion']
            pf['pop_yield_units{0}'] = 'photons/baryon'
        
        # Lx-SFR
        if 'cX' in kwargs:
            yield_X = kwargs['cX']
            if 'fX' in kwargs:
                yield_X *= kwargs['fX']
            
            pf['pop_yield{1}'] = yield_X
            
        elif 'fX' in kwargs:        
            pf['pop_yield{1}'] *= yield_X    
            
        if 'cX' or 'fX' in kwargs:
            pf["pop_Emin{1}"] = 2e2
            pf["pop_Emax{1}"] = 3e4
            pf["pop_EminNorm{1}"] = 2e2
            pf["pop_EmaxNorm{1}"] = 3e4
        
    else:
        pf = {}
        
    return pf

def count_populations(**kwargs):
    
    # Count populations
    popIDs = [0]
    for par in kwargs:

        m = re.search(r"\{([0-9])\}", par)

        if m is None:
            continue

        num = int(m.group(1))

        if num not in popIDs:
            popIDs.append(num)

    return len(popIDs)

class ParameterFile(dict):
    def __init__(self, pf=None, **kwargs):
        """
        Build parameter file instance.
        """

        self.defaults = SetAllDefaults()

        if pf is not None:
            self._kwargs = pf._kwargs
            for key in pf:
                self[key] = pf[key]
        else:
            self._kwargs = kwargs
            self._parse(**kwargs)
            self._check_for_conflicts(self)

    def _parse(self, **kwargs):
        """
        Parse kwargs dictionary.
        
        This means:
            1) Set all parameters to default values.
            2) Determine if there are multiple populations.
              i) If so, separate them apart.
            ...
              
        """    
                
        pf = self.defaults.copy()
    
        if 'problem_type' in kwargs:
            pf.update(ProblemType(kwargs['problem_type']))
    
        # Count populations
        Npops = self.Npops = count_populations(**kwargs)
    
        # Update parameter file
        if Npops == 1:
            pf.update(kwargs)
            self.pfs = [pf]
        else:
            pfs_by_pop = [{} for i in range(Npops)]
    
            linked_pars = []
    
            # Construct parameter file
            for par in kwargs:
    
                # Look for populations
                m = re.search(r"\{([0-9])\}", par)
    
                if m is None:
                    pf[par] = kwargs[par]
                    continue
                    
                # If we're here, it means this parameter has a population 
                # or source tag (i.e., an ID number in {}'s)   
    
                # Population ID number
                num = int(m.group(1))
    
                # ID including curly braces
                prefix = par.split(m.group(0))[0]
    
                # See if this parameter is linked to another population
                if type(kwargs[par]) is str:
                    mlink = re.search(r"\{([0-9])\}", kwargs[par])
                else:
                    mlink = None   
    
                # If it is linked, we'll handle it in just a sec
                if mlink is not None:
                    linked_pars.append(par)
                    continue
    
                # Otherwise, save it
                pfs_by_pop[num][prefix] = kwargs[par]
    
            # Update linked parameters
            for par in linked_pars:
    
                m = re.search(r"\{([0-9])\}", par)
                num = int(m.group(1))
    
                mlink = re.search(r"\{([0-9])\}", kwargs[par])
                num_link = int(mlink.group(1))
    
                # Pop ID including curly braces
                prefix_link = kwargs[par].split(mlink.group(0))[0]
    
                # If we didn't supply this parameter for the linked population,
                # assume default parameter value
                if kwargs[par] not in kwargs:
                    val = self.defaults[prefix_link]
                else:
                    val = kwargs[kwargs[par]]

                pfs_by_pop[num][prefix] = val

            # Make sure versions of source/spectrum parameters don't exist
            # in main parameter file if they are conflicting           
            tmp = {}
            
            # Loop through dicts, one per population
            for i in range(Npops):
            
                pop_pars = pfs_by_pop[i]
            
                # For each parameter...
                for element in pop_pars:
                    
                    # Only copy over new parameters
                    if element not in tmp:
                        tmp[element] = [pop_pars[element]]
                        continue
                        
                    #if pop_pars[element] not in tmp[element]:
                    #    tmp[element].append(pop_pars[element])
            
            # Delete elements that differ between populations
            #for key in tmp:    
            #    if len(tmp[key]) != 1 and key in pf:
            #        del pf[key]
            
            self.pfs = [pf.copy() for i in range(Npops)]
            for i in range(Npops):
                self.pfs[i].update(pfs_by_pop[i])
            
        pf.update(backwards_compatibility(pf['problem_type'], **pf))    
            
        for key in pf:
            self[key] = pf[key]
    
    def _check_for_conflicts(self, pf):
        """
        Run through parsed parameter file looking for conflicts.
        """
        
        try:
            verbose = pf['verbose']
        except KeyError:
            verbose = defaults['verbose']
        
        for kwarg in pf:
            
            m = re.search(r"\{([0-9])\}", kwarg)

            if m is None:
                par = kwarg

            else:
                # Population ID number
                num = int(m.group(1))
                
                # Pop ID including curly braces
                par = kwarg.split(m.group(0))[0]
            
            if par in self.defaults.keys():
                continue
            
            if verbose:
                print 'WARNING: Unrecognized parameter: %s' % par        
    
        conflicts = CheckForParameterConflicts(pf)
    
        if conflicts:
            raise Exception('Conflict(s) in input parameters.')
    
    