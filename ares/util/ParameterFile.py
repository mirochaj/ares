"""

ParameterFile.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Nov 28 16:19:26 CST 2014

Description:

"""

import re
from .ProblemTypes import ProblemType
from .BackwardCompatibility import backward_compatibility
from .SetDefaultParameterValues import ParameterizedQuantityParameters
from .SetDefaultParameterValues import SetAllDefaults, CosmologyParameters
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
except ImportError:
    rank = 0

old_pars = ['fX', 'cX', 'fstar', 'fesc', 'Nion', 'Nlw', 'Tmin', 'Mmin', 'fXh']

_cosmo_params = CosmologyParameters()

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

        kw['{0!s}{{{1}}}'.format(prefix, int(m.group(1)))] = kwargs[par]

    return kw

def check_for_brackets(pf):
    has_brackets = False
    for par in pf:
        # Spare us from using re.search if we can.
        if not (par.startswith('pop') or par.startswith('pq') or par.startswith('source')):
            continue

        # Look for integers within curly braces
        m = re.search(r"\{([0-9])\}", par)

        if m is None:
            continue

        has_brackets = True
        break

    return has_brackets

def pop_id_num(par):
    """
    Split apart parameter prefix and ID number.
    """

    # Spare us from using re.search if we can.
    if not (par.startswith('pop') or par.startswith('pq') or par.startswith('source')):
        return par, None

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
    identification number that corresponds to a ParameterizedQuantity.
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

        if not isinstance(kwargs[par], basestring):
            continue

        if kwargs[par][0:2] != 'pq':
            continue

        prefix, popid, phpid = par_info(kwargs[par])

        if phpid is None:
            phpid = 0

        if phpid not in phpIDs:
            phps.append(par)
            phpIDs.append(phpid)

    return len(phpIDs), phps

def identify_pqs(**kwargs):
    """
    Count the number of parameterized halo properties in this model.

    Sort them by population ID #.

    Returns
    -------
    List of lists: ParameterizedQuantity ID numbers for each population.
    NOTE: they are not in order.
    """

    Npops = count_populations(**kwargs)
    phps = [[] for i in range(Npops)]

    for par in kwargs:

        if not isinstance(kwargs[par], basestring):
            continue

        if (kwargs[par] != 'pq') and (kwargs[par][0:3] != 'pq['):
            continue

        # This will NOT have a pop ID
        just_php, nothing, phpid = par_info(kwargs[par])

        # This might have a popid
        prefix, popid, age = par_info(par)

        if (popid is None) and (Npops == 1):
            # I think this is guaranteed to be true
            if prefix not in phps[0]:
                phps[0].append(prefix)
        elif (popid is not None):
            if prefix in phps[popid]:
                continue

            phps[popid].append(prefix)
        else:
            # Not clear why this happens...
            continue

    return phps

def get_pq_pars(par, pf):
    """
    Find ParameterizedQuantity parameters...for this parameter.

    ..note:: par isn't the name of the parameter, it is the value. Usually,
        it's something like 'pq[0]'.

    For example, if in the parameter file, you set:

        'pop_fesc{0}'='pq[1]'

    This routine runs off and finds all parameters that look like:

        'pq_*par?{0}[1]'

    Returns
    -------
    Dictionary of parameters to be used to initialize a new HaloProperty.

    """

    prefix, popid, phpid = par_info(par)

    pars = {}
    for key in pf:

        # If this isn't a PQ parameter, move on
        if key[0:2] != 'pq':
            continue

        # This is to prevent unset PQ parameters from causing
        if not re.search('\[{}\]'.format(phpid), key):

            if (pf.Npqs == 1):
                # In this case, the default for this parameter will
                # be found since PQs are listed in defaults without any
                # ID number.
                pass
            else:
                # In this case, no default will be found, so an error will
                # get thrown unless we tag an ID on?
                continue

        # Break apart parameter name, pop ID number, and PQ ID number
        p, popid, phpid_ = par_info(key)

        # If there's only one PQ, not having an ID number is OK. We'll just
        # grab whatever is in the parameter file already, which will be
        # the default if nothing else was supplied.
        if (phpid is None) and (pf.Npqs == 1):
            pars[p] = pf['{!s}'.format(p)]

        # This means we probably have some parameters bracketed
        # and some not...should make it so this doesn't happen
        elif (phpid is not None) and (pf.Npqs == 1):

            try:
                pars[p] = pf['{0!s}[{1}]'.format(p, phpid)]
            except KeyError:
                # This means it's just default values
                pars[p] = pf['{!s}'.format(p)]
        else:
            pars[p] = pf['{0!s}[{1}]'.format(p, phpid)]

    return pars

# All defaults
defaults = SetAllDefaults()

# Defaults w/o all parameters that are population-specific
# This is to-be-used in reconstructing a master parameter file
pops_need = 'pop_', 'source_'
defaults_pop_dep = {}
defaults_pop_indep = {}
for key in defaults:

    ct = 0
    for pop_par in pops_need:
        if re.search(pop_par, key):
            ct += 1

    if ct > 0:
        defaults_pop_dep[key] = defaults[key]
        continue

    defaults_pop_indep[key] = defaults[key]

class ParameterFile(dict):
    def __init__(self, **kwargs):
        """
        Build parameter file instance.
        """

        # Keep user-supplied kwargs as attribute
        self._kwargs = kwargs.copy()

        #print len(kwargs), len(defaults)
        #if len(kwargs) < 0.5 * len(defaults):
        #    for par in self._kwargs:
        #        if par not in _cosmo_params:
        #            continue
        #
        #        if self._kwargs[par] == _cosmo_params[par]:
        #            continue
        #
        #        print "WARNING: {!s} is cosmological parameter.".format(par)
        #        print "       : Must update initial conditions and HMF tables!"

        # Fix up everything
        self._parse(**kwargs)

        # Check for stuff that'll break...stuff
        if self['debug']:
            self._check_for_conflicts(**kwargs)

            #if self.orphans:
            #    if (rank == 0) and self['verbose']:
            #        for key in self.orphans:
            #            print("WARNING: {!s} is an `orphan` parameter.".format(\
            #                key))

    @property
    def Npops(self):
        if not hasattr(self, '_Npops'):

            tmp = {}
            if 'problem_type' in self._kwargs:
                tmp.update(ProblemType(self._kwargs['problem_type']))
            tmp.update(self._kwargs)

            self._Npops = count_populations(**tmp)

        return self._Npops

    @property
    def Npqs(self):
        if not hasattr(self, '_Npqs'):
            tmp = {}
            if 'problem_type' in self._kwargs:
                tmp.update(ProblemType(self._kwargs['problem_type']))
            tmp.update(self)

            self._Npqs, self._pqs = count_properties(**tmp)

        return self._Npqs

    @property
    def pqs(self):
        """
        List of parameterized halo properties.
        """
        if not hasattr(self, '_pqs'):
            tmp = self.Npqs
        return self._pqs

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
            kw['problem_type'] = defaults['problem_type']

        # Change underscores to brackets in parameter names
        kw = bracketify(**kw)

        # Read in kwargs for this problem type
        kwargs = ProblemType(kw['problem_type'])

        # Add in user-supplied kwargs
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

        pf_base.update(defaults)

        self.pf_base = pf_base.copy()

        # For single-population calculations, we're done for the moment
        if self.Npops == 1:
            has_brackets = check_for_brackets(kwargs)

            if has_brackets:
                s = "For single population models, must eliminate ID numbers"
                s += " from parameter names!"
                raise ValueError(s)

            pfs_by_pop = self.update_pq_pars([pf_base], **kwargs)
            pfs_by_pop[0].update(kwargs)

            self.pfs = pfs_by_pop

        # Otherwise, we need to go through and make separate dictionaries
        # for each population
        else:

            # First: make base parameter file that contains only parameters
            # that are NOT population specific.
            # Second:

            # Can't add kwargs yet (all full of curly braces)

            # Only add non-pop-specific parameters from ProblemType defaults
            prb = ProblemType(kwargs['problem_type'])
            for par in defaults_pop_indep:

                # Just means this parameter is not default to the
                # problem type.
                if par not in prb:
                    continue

                pf_base[par] = prb[par]

            # and kwargs
            for par in kwargs:
                if par in defaults_pop_indep:
                    pf_base[par] = kwargs[par]
                else:

                    # This is exclusively to handle the case where
                    # we have a PQ that's NOT attached to a population.
                    prefix, popid, phpid = par_info(par)

                    if (phpid is not None) and (popid is None):
                        pf_base[par] = kwargs[par]

            # We now have a parameter file containing all non-pop-specific
            # parameters, which we can use as a base for all pop-specific
            # parameter files.
            pfs_by_pop = [pf_base.copy() for i in range(self.Npops)]

            pfs_by_pop = self.update_pq_pars(pfs_by_pop, **kwargs)

            # Some pops are linked together: keep track of them, apply
            # fixes at the end.
            linked_pars = []

            # Add population-specific changes
            for par in kwargs:

                # See if this parameter belongs to a particular population
                # We DON'T care at this stage about []'s
                #prefix, popid, phpid = par_info(par)
                prefix, popid = pop_id_num(par)

                if (popid is None):
                    # We already handled non-pop-specific parameters
                    continue

                # If we're here, it means this parameter has a population
                # or source tag (i.e., an ID number in {}'s or _'s)

                # See if this parameter is linked to another population
                # OR another parameter within the same population.
                # The latter only occurs for PHPs.
                if isinstance(kwargs[par], basestring):
                    prefix_link, popid_link, phpid_link = par_info(kwargs[par])
                    if (popid_link is None) and (phpid_link is None):
                        # Move-on: nothing to see here
                        # Just a parameter that can be a string
                        pass

                    if (phpid_link is None):
                        pass
                    # In this case, might have some intra-population link-age
                    elif kwargs[par] == 'pq[{}]'.format(phpid_link):
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

                # Info for the parameter whose value is linked to another
                prefix, popid, phpid = par_info(par)

                # Parameter whose value were taking
                prefix_link, popid_link, phpid_link = par_info(kwargs[par])

                # Account for the fact that the parameter name might have []'s
                if phpid is None:
                    name = prefix
                else:
                    name = '{0!s}[{1}]'.format(prefix, phpid)

                if phpid_link is None:
                    name_link = prefix_link
                else:
                    name_link = '{0!s}[{1}]'.format(prefix_link, phpid_link)

                # If we didn't supply this parameter for the linked population,
                # assume default parameter value
                if name_link not in pfs_by_pop[popid_link]:
                    val = defaults[prefix_link]
                else:
                    val = pfs_by_pop[popid_link][name_link]

                pfs_by_pop[popid][name] = val

            # Save as attribute
            self.pfs = pfs_by_pop

        # Master parameter file
        # Only tag ID number to pop or source parameters
        for i, poppf in enumerate(self.pfs):

            # Loop over all population parameters and add them to the
            # master parameter file with their {ID}.
            for key in poppf:

                # Remember, `key` won't have any {}'s

                if self.Npops > 1 and key in defaults_pop_dep:
                    self['{0!s}{{{1}}}'.format(key, i)] = poppf[key]
                else:
                    self[key] = poppf[key]

        # Distribute 'master' parameters.

    def update_pq_pars(self, pfs_by_pop, **kwargs):
        # In a given population, there may be 1+ parameterized halo
        # properties ('phps') denoted by []'s. We need to update the
        # defaults to have these square brackets!
        phps = identify_pqs(**kwargs)
        php_defs = ParameterizedQuantityParameters()

        # Need to do this even for single population runs
        for i, pf in enumerate(pfs_by_pop):
            if len(phps[i]) < 2:
                continue

            for key in php_defs:
                del pf[key]
                for k in range(len(phps[i])):
                    pf['{0!s}[{1}]'.format(key, k)] = php_defs[key]

        return pfs_by_pop

    @property
    def not_default(self):
        """
        Show the parameters that are not defaults.
        """
        if not hasattr(self, '_not_default'):
            self._not_default = {}

            ptype = ProblemType(self['problem_type'])

            for key in self:
                if key in defaults_pop_indep:
                    if self[key] != defaults_pop_indep[key]:
                        self._not_default[key] = self[key]

                elif key in ptype:
                    if self[key] != ptype[key]:
                        self._not_default[key] = self[key]

                # Additional population?
                else:
                    prefix, num = pop_id_num(key)

                    if prefix in defaults:
                        if self[key] != defaults[prefix]:
                            self._not_default[key] = self[key]

        return self._not_default

    def _check_for_conflicts(self, **kwargs):
        """
        Run through parsed parameter file looking for conflicts.
        """

        try:
            verbose = kwargs['verbose']
        except KeyError:
            verbose = defaults['verbose']

        for kwarg in kwargs:

            par, num = pop_id_num(kwarg)
            if num is None:
                par = kwarg

            if par in defaults.keys():
                continue

            if par in old_pars:
                continue

            if re.search('\[', par):
                continue

            if verbose:
                print('WARNING: Unrecognized parameter: {!s}'.format(par))
