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
from .BackwardCompatibility import backward_compatibility
from .CheckForParameterConflicts import CheckForParameterConflicts

old_pars = ['fX', 'cX', 'fstar', 'fesc', 'Nion', 'Nlw']

def bracketify(**kwargs):
    kw = {}
    for par in kwargs:
        m = re.search(r"\_([0-9])\_", par)
        
        if m is None:
            kw[par] = kwargs[par]
            continue
            
        prefix = par.split(m.group(0))[0]
        
        kw['%s{%i}' % (prefix, int(m.group(1)))] = kwargs[par]
    
    return kw

def pop_id_num(par):
    """
    Split apart parameter prefix and ID number.
    """
        
    # Look for integers within curly braces
    m = re.search(r"\{([0-9])\}", par)

    if m is None:
        
        # Look for integers within underscores
        m = re.search(r"\_([0-9])\_", par)
        
        if m is None:
            return par, None
    
    prefix = par.replace(m.group(0), '')
    
    return prefix, int(m.group(1))

def par_info(par):
    
    prefix1, popid = pop_id_num(par)
        
    if prefix1 is not None:
        m = re.search(r"\[(\d+(\.\d*)?)\]", prefix1)
    else:
        m = None
        prefix1 = par

    if m is None:
        return prefix1, popid, None

    prefix2 = prefix1.replace(m.group(0), '')
    z = m.group(1)

    return prefix2, popid, float(z)

def count_populations(**kwargs):
    
    # Count populations
    popIDs = [0]
    for par in kwargs:

        prefix, num = pop_id_num(par)
        if num is None:
            continue

        if num not in popIDs:
            popIDs.append(num)

    return len(popIDs)

class ParameterFile(dict):
    def __init__(self, **kwargs):
        """
        Build parameter file instance.
        """

        self.defaults = SetAllDefaults()
        
        # Defaults w/o all parameters that are population-specific
        self.defaults_pop_dep = {}
        self.defaults_pop_indep = {}
        for key in self.defaults:
            if re.search('pop_', key) or re.search('source_', key):
                self.defaults_pop_dep[key] = self.defaults[key]
                continue
            
            self.defaults_pop_indep[key] = self.defaults[key]        
        
        # Keep user-supplied kwargs as attribute  
        self._kwargs = kwargs.copy()
        
        # Fix up everything
        self._parse(**kwargs)
        
        # Check for stuff that'll break...stuff
        self._check_for_conflicts(**kwargs)
    
    @property
    def Npops(self):
        if not hasattr(self, '_Npops'):
            tmp = self._kwargs.copy()
            if 'problem_type' in self._kwargs:
                tmp.update(ProblemType(self._kwargs['problem_type']))
                
            self._Npops = count_populations(**tmp)

        return self._Npops

    def _parse(self, **kw):
        """
        Parse kwargs dictionary.
        
        There has to be a better way...
        
        If Npops == 1, the master dictionary should *not* have any parameters
        with curly braces.
        If Npops > 1, all population-specific parameters *must* be associated
        with a population, i.e., have curly braces in the name.
              
        """    
                
        # Start w/ problem specific parameters (always)
        if 'problem_type' not in kw:
            kw['problem_type'] = self.defaults['problem_type']    
            
        # Change names of parameters to ensure backward compatibility        
        kw.update(backward_compatibility(kw['problem_type'], **kw))
        kw = bracketify(**kw)
            
        kwargs = ProblemType(kw['problem_type'])
        kwargs.update(kw)
                        
        ##    
        # Up until this point, just problem_type-specific kwargs and any
        # modifications passed in by the user.
        ##
            
        pf_base = {}  # Temporary master parameter file   
                      # Should have no {}'s
                    
        pf_base.update(self.defaults)            
                    
        # For single-population calculations, we're done for the moment
        if self.Npops == 1:
            pf_base.update(kwargs)
            
            self.pfs = [pf_base]
            
        # Otherwise, we need to go through and make separate dictionaries
        # for each population
        else:
            
            # Can't add kwargs yet (all full of curly braces)
            
            # Only add non-pop-specific parameters from ProblemType defaults
            prb = ProblemType(kwargs['problem_type'])
            for par in self.defaults_pop_indep:
                if par not in prb:
                    continue
                    
                pf_base[par] = prb[par]
            
            # and kwargs
            for par in kwargs:
                if par in self.defaults_pop_indep:
                    pf_base[par] = kwargs[par]
            
            # Sort parameter files by population
            # This has all parameters that *don't* vary from one
            # population to the next, and as such is the basis for all
            # population-specific dictionaries
            pfs_by_pop = [pf_base.copy() for i in range(self.Npops)]
            
            linked_pars = []
    
            # Add population-specific changes
            for par in kwargs:
                    
                # See if this parameter belongs to a particular population
                prefix, num = pop_id_num(par)
                if num is None:
                    # We already handled non-pop-specific parameters
                    continue
                    
                # If we're here, it means this parameter has a population
                # or source tag (i.e., an ID number in {}'s or _'s)

                # See if this parameter is linked to another population
                if type(kwargs[par]) is str:
                    prefix_link, num_link = pop_id_num(kwargs[par])
                else:
                    prefix_link, num_link = None, None
                
                # If it is linked, we'll handle it in just a sec
                if num_link is not None:
                    linked_pars.append(par)
                    continue
    
                # Otherwise, save it
                pfs_by_pop[num][prefix] = kwargs[par]
    
            # Update linked parameters
            for par in linked_pars:
            
                prefix, num = pop_id_num(par)
                prefix_link, num_link = pop_id_num(kwargs[par])
            
                # If we didn't supply this parameter for the linked population,
                # assume default parameter value
                if prefix not in pfs_by_pop[num_link]:
                    val = self.defaults[prefix_link]
                else:
                    val = pfs_by_pop[num_link][prefix_link]
            
                pfs_by_pop[num][prefix] = val

            # Save as attribute
            self.pfs = pfs_by_pop
                                    
        # Master parameter file    
        # Only tag ID number to pop or source parameters
        for i, poppf in enumerate(self.pfs):

            for key in poppf:
                
                if self.Npops > 1 and key in self.defaults_pop_dep:
                    self['%s{%i}' % (key, i)] = poppf[key]
                else:
                    self[key] = poppf[key]
                    
    @property
    def unique(self):
        """
        Show the parameters that are not defaults.
        """                    
        if not hasattr(self, '_unique'):
            self._unique = {}
            
            ptype = ProblemType(self['problem_type'])
            
            for key in self:
                if key in self.defaults_pop_indep:
                    if self[key] != self.defaults_pop_indep[key]:
                        self._unique[key] = self[key]
                
                elif key in ptype:
                    if self[key] != ptype[key]:
                        self._unique[key] = self[key]
                
                # Additional population?
                else:
                    prefix, num = pop_id_num(key)
                    
                    if num is None:
                        continue
                    
                    if prefix in self.defaults and (num >= self.Npops):
                        if self[key] != self.defaults[prefix]:
                            self._unique[key] = self[key]
                
        return self._unique

    def _check_for_conflicts(self, **kwargs):
        """
        Run through parsed parameter file looking for conflicts.
        """
        
        if 'need_for_speed' in kwargs:
            if kwargs['need_for_speed']:
                return 
        
        try:
            verbose = kwargs['verbose']
        except KeyError:
            verbose = self.defaults['verbose']
        
        for kwarg in kwargs:
            
            par, num = pop_id_num(kwarg)
            if num is None:
                par = kwarg
            
            if par in self.defaults.keys():
                continue
                
            if par in old_pars:
                continue
            
            if re.search('\[', par):
                continue 
            
            if verbose:
                print 'WARNING: Unrecognized parameter: %s' % par        
    
        #conflicts = CheckForParameterConflicts(kwargs)
    
        #if conflicts:
        #    raise Exception('Conflict(s) in input parameters.')
    
    