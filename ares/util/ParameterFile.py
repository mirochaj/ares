"""

ParameterFile.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Nov 28 16:19:26 CST 2014

Description: 

"""

import re
from collections import defaultdict
from .ProblemTypes import ProblemType
from .SetDefaultParameterValues import SetAllDefaults
from .CheckForParameterConflicts import CheckForParameterConflicts

def count_populations(**kwargs):
    
    if 'source_kwargs' in kwargs:
        if kwargs['source_kwargs'] is not None:
            return len(kwargs['source_kwargs'])
    
    if 'spectrum_kwargs' in kwargs:
        if kwargs['spectrum_kwargs'] is not None:
            return len(kwargs['spectrum_kwargs'])
    
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
        else:
            src_kw = [{} for i in range(Npops)]
            spec_kw = [{} for i in range(Npops)]
    
            linked_pars = []
    
            # Construct parameter file
            for par in kwargs:
    
                # Look for populations
                m = re.search(r"\{([0-9])\}", par)
    
                if m is None:
                    pf[par] = kwargs[par]
                    continue
    
                # Population ID number
                num = int(m.group(1))
    
                # Pop ID including curly braces
                prefix = par.strip(m.group(0))
    
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
                if re.search('spectrum', par):
                    spec_kw[num][prefix] = kwargs[par]
                else:
                    src_kw[num][prefix] = kwargs[par]
    
            # Update linked parameters
            for par in linked_pars:
    
                m = re.search(r"\{([0-9])\}", par)
                num = int(m.group(1))
    
                mlink = re.search(r"\{([0-9])\}", kwargs[par])
                num_link = int(mlink.group(1))
    
                # Pop ID including curly braces
                prefix_link = kwargs[par].strip(mlink.group(0))
    
                # If we didn't supply this parameter for the linked population,
                # assume default parameter value
                if kwargs[par] not in kwargs:
                    val = self.defaults[prefix_link]
                else:
                    val = kwargs[kwargs[par]]

                if re.search('spectrum', par):
                    spec_kw[num][prefix_link] = val
                else:
                    src_kw[num][prefix_link] = val

            # Insert defaults so all populations have the same size dictionaries
            for thing in [src_kw, spec_kw]:        
                all_varying_keys = []
                for pop_pars in thing:
                    for key in pop_pars:

                        if key not in all_varying_keys:
                            all_varying_keys.append(key)

                for pop_pars in thing:
                    for key in all_varying_keys:
                        if key not in pop_pars:
                            pop_pars[key] = self.defaults[key]

            # Update parameter file
            pf.update({'source_kwargs': src_kw})
            pf.update({'spectrum_kwargs': spec_kw})

            # Make sure versions of source/spectrum parameters don't exist
            # in main parameter file if they are conflicting
            for thing in ['source_kwargs', 'spectrum_kwargs']:

                tmp = {}

                # Loop through dicts, one per population
                for i, pop_pars in enumerate(pf[thing]):

                    # For each parameter...
                    for element in pop_pars:
                        
                        if element not in tmp:
                            tmp[element] = [pop_pars[element]]
                            continue
                            
                        if pop_pars[element] not in tmp[element]:
                            tmp[element].append(pop_pars[element])
                
                # Delete elements that differ between populations
                for key in tmp:    
                    if len(tmp[key]) != 1 and key in pf:
                        del pf[key]
            
        for key in pf:
            self[key] = pf[key]
    
    def _check_for_conflicts(self, pf):
        """
        Run through parsed parameter file looking for conflicts.
        """
        for kwarg in pf:
            
            m = re.search(r"\{([0-9])\}", kwarg)

            if m is None:
                par = kwarg

            else:
                # Population ID number
                num = int(m.group(1))
                
                # Pop ID including curly braces
                par = kwarg.strip(m.group(0))
            
            if par in self.defaults.keys():
                continue
            
            print 'WARNING: Unrecognized parameter: %s' % par        
    
        conflicts = CheckForParameterConflicts(pf)
    
        if conflicts:
            raise Exception('Conflict(s) in input parameters.')
    
    