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
from .SetDefaultParameterValues import HaloPropertyParameters
from .CheckForParameterConflicts import CheckForParameterConflicts

old_pars = ['fX', 'cX', 'fstar', 'fesc', 'Nion', 'Nlw']

def bracketify(**kwargs):
    """
    Convert underscores to brackets.
    """
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
    """
    Break apart parameter name, a population ID #, and potentially another
    identification number that corresponds to a ParameterizedHaloProperty.
    """
    
    prefix1, popid = pop_id_num(par)    
        
    if prefix1 is not None:
        m = re.search(r"\[(\d+(\.\d*)?)\]", prefix1)
    else:
        m = None
        prefix1 = par

    if m is None:
        return prefix1, popid, None

    prefix2 = prefix1.replace(m.group(0), '')
    phpid = m.group(1)

    return prefix2, popid, int(phpid)

def count_populations(**kwargs):
    """
    Count the number of populations to be used for this calculation.
    """
    # Count populations
    popIDs = [0]
    for par in kwargs:

        prefix, num = pop_id_num(par)
        if num is None:
            continue

        if num not in popIDs:
            popIDs.append(num)

    return len(popIDs)
    
def count_properties(**kwargs):
    """
    Count the number of parameterized halo properties in this model.
    """
    
    phps = []    
    phpIDs = []
    for par in kwargs:

        if type(kwargs[par]) is not str:
            continue

        if kwargs[par][0:3] != 'php':
            continue
        
        prefix, popid, phpid = par_info(kwargs[par])
                
        if phpid is None:
            phpid = 0

        if phpid not in phpIDs:
            phps.append(par)
            phpIDs.append(phpid)

    return len(phpIDs), phps
    
def identify_phps(**kwargs):
    """
    Count the number of parameterized halo properties in this model.
    
    Sort them by population ID #.
    
    Returns
    -------
    List of lists: parameterized halo properties for each population.
    NOTE: they are not in order
    """

    Npops = count_populations(**kwargs)
    phps = [[] for i in range(Npops)]

    for par in kwargs:

        if type(kwargs[par]) is not str:
            continue

        if (kwargs[par] != 'php') and (kwargs[par][0:4] != 'php['):
            continue

        # This will NOT have a pop ID
        just_php, nothing, phpid = par_info(kwargs[par])
        
        prefix, popid, age = par_info(par)
        
        if (popid is None) and (Npops == 1):
            # I think this is guaranteed to be true
            if prefix not in phps[0]:
                phps[0].append(prefix) 
        else:
            if prefix in phps[popid]:
                continue
                
            phps[popid].append(prefix)
            
    return phps
    
    ## Sort PHPs
    #for i, pop in enumerate(phps):
    #    if not pop:
    #        continue
    #        
    #    tmp = [None for k in range(len(pop))]
    #    for j, php in enumerate(pop):
    #        
    #        k = kwargs['']
    #        tmp[]
            
            
        
        

class ParameterFile(dict):
    def __init__(self, **kwargs):
        """
        Build parameter file instance.
        """

        self.defaults = SetAllDefaults()
        
        # Defaults w/o all parameters that are population-specific
        # This is to-be-used in reconstructing a master parameter file
        self.defaults_pop_dep = {}
        self.defaults_pop_indep = {}
        for key in self.defaults:
            if re.search('pop_', key) or re.search('source_', key) or \
               re.search('php_', key):
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
    
    @property
    def Nphps(self):
        if not hasattr(self, '_Nphps'):
            tmp = self._kwargs.copy()
            if 'problem_type' in self._kwargs:
                tmp.update(ProblemType(self._kwargs['problem_type']))
    
            self._Nphps, self._phps = count_properties(**tmp)

        return self._Nphps
    
    @property
    def phps(self):
        """
        List of parameterized halo properties.
        """
        if not hasattr(self, '_phps'):
            tmp = self.Nphps
        return self._phps
    
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
            
        # Change underscores to brackets in parameter names
        kw = bracketify(**kw)
        
        # Read in kwargs for this problem type
        kwargs = ProblemType(kw['problem_type'])
        
        tmp = kwargs.copy()
        tmp.update(kw)
        
        # Change names of parameters to ensure backward compatibility        
        tmp.update(backward_compatibility(kw['problem_type'], **tmp))
        kwargs.update(tmp)            
                        
        ##    
        # Up until this point, just problem_type-specific kwargs and any
        # modifications passed in by the user.
        ##
            
        pf_base = {}  # Temporary master parameter file   
                      # Should have no {}'s
                    
        pf_base.update(self.defaults)  
                    
        # For single-population calculations, we're done for the moment
        if self.Npops == 1:
            pfs_by_pop = self.update_php_pars([pf_base], **kwargs)
            pfs_by_pop[0].update(kwargs)
            
            self.pfs = pfs_by_pop
            
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
                    
            # We now have a parameter file containing all non-pop-specific
            # parameters, which we can use as a base for all pop-specific
            # parameter files.        
            pfs_by_pop = [pf_base.copy() for i in range(self.Npops)]
            
            pfs_by_pop = self.update_php_pars(pfs_by_pop, **kwargs)
             
            # Some pops are linked together: keep track of them, apply
            # fixes at the end.
            linked_pars = []
    
            # Add population-specific changes
            for par in kwargs:
                    
                # See if this parameter belongs to a particular population
                # We DON'T care at this stage about []'s
                prefix, popid = pop_id_num(par)
                if popid is None:
                    # We already handled non-pop-specific parameters
                    continue
                    
                # If we're here, it means this parameter has a population
                # or source tag (i.e., an ID number in {}'s or _'s)

                # See if this parameter is linked to another population
                # OR another parameter within the same population.
                # The latter only occurs for PHPs.
                if (type(kwargs[par]) is str):
                    prefix_link, popid_link, phpid_link = par_info(kwargs[par])
                    if (popid_link is None) and (phpid_link is None):
                        # Move-on: nothing to see here
                        # Just a parameter that can be a string
                        pass
                    
                    if (phpid_link is None):
                        pass
                    # In this case, might have some intra-population link-age    
                    elif kwargs[par] == 'php[%i]' % phpid_link:
                        # This is the only false alarm I think
                        prefix_link, popid_link, phpid_link = None, None, None
                else:
                    prefix_link, popid_link, phpid_link = None, None, None
                
                # If it is linked, we'll handle it in just a sec
                if (popid_link is not None) or (phpid_link is not None):                    
                    linked_pars.append(par)
                    continue
    
                # Otherwise, save it
                pfs_by_pop[popid][prefix] = kwargs[par]
    
            # Update linked parameters
            for par in linked_pars:
                
                # Grab info for linker and linkee
                prefix, popid, phpid = par_info(par)
                prefix_link, popid_link, phpid_link = par_info(kwargs[par])
                            
                # Account for the fact that the parameter name might have []'s            
                if phpid is None:
                    name = prefix
                else:
                    name = '%s[%i]' % (prefix, phpid)
                    
                if phpid_link is None:
                    name_link = prefix_link
                else:
                    name_link = '%s[%i]' % (prefix_link, phpid_link)
            
                # If we didn't supply this parameter for the linked population,
                # assume default parameter value
                if name_link not in pfs_by_pop[popid_link]:
                    val = self.defaults[prefix_link]
                else:
                    val = pfs_by_pop[popid_link][name_link]
            
                pfs_by_pop[popid_link][name] = val

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

    def update_php_pars(self, pfs_by_pop, **kwargs):
        # In a given population, there may be 1+ parameterized halo
        # properties ('phps') denoted by []'s. We need to update the
        # defaults to have these square brackets!
        phps = identify_phps(**kwargs)
        php_defs = HaloPropertyParameters()
        
        # Need to do this even for single population runs
        for i, pf in enumerate(pfs_by_pop):
            if len(phps[i]) < 2:
                continue

            for key in php_defs:
                del pf[key]
                for k in range(len(phps[i])):
                    pf['%s[%i]' % (key, k)] = php_defs[key]

        return pfs_by_pop

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
    
    