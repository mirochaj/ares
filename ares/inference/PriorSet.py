"""
File: $ARES/ares/inference/PriorSet.py

Author: Keith Tauscher
Affiliation: University of Colorado at Boulder
Created on: Sat Mar 19 15:01:00 2016
Updated on: Tue Feb 28 00:37:34 2017

Description: A container which can hold an arbitrary number of priors, each of
             which can have any number of parameters which it describes (as
             long as the specific prior supports that number of parameters).
             Priors can be added through PriorSet.add_prior(prior, params)
             where prior is a _Prior and params is a list of the parameters to
             which prior applies. Once all the priors are added, points can be
             drawn using PriorSet.draw() and the log_prior of the entire set of
             priors can be evaluated at a point using
             PriorSet.log_prior(point). See documentation of individual
             functions for further details.
"""

import numpy as np
from .Priors import _Prior

list_types = [list, tuple, np.ndarray]
valid_transforms = ['log', 'log10', 'square', 'arcsin', 'logistic']

ln10 = np.log(10)

def _check_if_valid_transform(transform):
    #
    # Checks if the given variable is either None
    # or a string describing a valid transform.
    #
    if type(transform) is str:
        if transform not in valid_transforms:
            raise ValueError("The transform given to apply" +\
                             " to a variable was not valid.")
    elif (transform is not None):
        raise ValueError("The type of the transform given to a PriorSet " +\
                         "to apply to a parameter was not recognized.")

def _log_prior_addition(value, transform):
    #
    # Finds the term which should be added to the log prior due to the
    # transform (pretty much the log of the derivative of the transformed
    # parameter with respect to the original parameter evaluated at value.
    #
    if transform is None:
        return 0.
    elif transform == 'log':
        return -1. * np.log(value)
    elif transform == 'log10':
        return -1. * np.log(ln10 * value)
    elif transform == 'square':
        return np.log(2 * value)
    elif transform == 'arcsin':
        return -np.log(1.-np.power(value, 2.)) / 2.
    elif transform == 'logistic':
        return -np.log(value * (1. - value))
    else:
        raise ValueError("For some reason the _log_prior_addition " +\
                         "function wasn't implemented for the transform " +\
                         "given, which is \"%s\"." % (transform,))

def _apply_transform(value, transform):
    #
    # Applies the given transform to the value and returns the result.
    #
    if transform is None:
        return value
    elif transform == 'log':
        return np.log(value)
    elif transform == 'log10':
        return np.log10(value)
    elif transform == 'square':
        return np.power(value, 2.)
    elif transform == 'arcsin':
        return np.arcsin(value)
    elif transform == 'logistic':
        return np.log(value / (1. - value))
    else:
        raise ValueError("Something went wrong and an attempt to evaluate " +\
                         "an invalid transform was made. This should " +\
                         "have been caught by previous error catching!")

def _apply_inverse_transform(value, transform):
    #
    # Applies the inverse of the given transform
    # to the value and returns the result.
    #
    if transform is None:
        return value
    elif transform == 'log':
        return np.e ** value
    elif transform == 'log10':
        return 10. ** value
    elif transform == 'square':
        return np.sqrt(value)
    elif transform == 'arcsin':
        return np.sin(value)
    elif transform == 'logistic':
        return 1 / (1. + (np.e ** (-value)))
    else:
        raise ValueError("Something went wrong and an attempt to evaluate" +\
                         " an invalid (inverse) transform was made. This" +\
                         "should've been caught by previous error catching!")
        

class PriorSet(object):
    """
    An object which keeps track of many priors which can be univariate or
    multivariate. It provides methods like log_prior, which calls log_prior on
    all of its constituent priors, and draw, which draws from all of its
    constituent priors.
    """
    def __init__(self, prior_tuples=[]):
        """
        Creates a new PriorSet with the given priors inside.
        
        prior_tuples a list of lists/tuples of the form (prior, params) where
                     prior is an instance of the _Prior class and params is
                     a list of parameters (strings) which prior describes.
        """
        self._data = []
        self._params = []
        if type(prior_tuples) in list_types:
            for iprior in range(len(prior_tuples)):
                this_tup = prior_tuples[iprior]
                if (type(this_tup) in list_types) and len(this_tup) == 3:
                    (prior, params, transforms) = prior_tuples[iprior]
                    self.add_prior(prior, params, transforms)
                else:
                    raise ValueError("One of the prior tuples provided to " +\
                                     "the initializer of a PriorSet was not" +\
                                     " a list/tuple of length 2 like " +\
                                     "(prior, params).")
        else:
            raise ValueError("The prior_tuples argument given to the " +\
                             "initializer was not list-like. It should be " +\
                             "a list of tuples of the form (prior, params, " +\
                             "transformations) where prior is a _Prior " +\
                             "object and params and transformations " +\
                             "are lists of strings.")

    @property
    def empty(self):
        """
        Finds whether this PriorSet is empty.
        
        returns True if no priors have been added, False otherwise
        """
        return (len(self._data) == 0)

    @property
    def params(self):
        """
        Finds and returns the parameters which this PriorSet describes.
        """
        return self._params

    def add_prior(self, prior, params, transforms=None):
        """
        Adds a prior and the parameters it describes to the PriorSet.
        
        prior prior describing the given parameters (should be a _Prior object)
        params list of parameters described by the given prior
               (can be a single string if the prior is univariate)
        transforms list of transformations to apply to the parameters
                   (can be a single string if the prior is univariate)
        """
        if isinstance(prior, _Prior):
            if transforms is None:
                transforms = [None] * prior.numparams
            elif (type(transforms) is str):
                _check_if_valid_transform(transforms)
                if (prior.numparams == 1):
                    transforms = [transforms]
                else:
                    raise ValueError("The transforms variable applied" +\
                                     " to parameters of a PriorSet was " +\
                                     "provided as a string even though the " +\
                                     "prior being provided was multivariate.")
            elif type(transforms) in list_types:
                if len(transforms) == prior.numparams:
                    for itransform in range(len(transforms)):
                        _check_if_valid_transform(transforms[itransform])
                else:
                    raise ValueError("The list of transforms applied to " +\
                                     "parameters in a PriorSet was not " +\
                                     "the same length as the list of " +\
                                     "parameters of the prior.")
            else:
                raise ValueError("The type of the transforms variable " +\
                                 "supplied to PriorSet's add_prior function" +\
                                 " was not recognized. It should be a " +\
                                 "single valid string (if prior is " +\
                                 "univariate) or list of valid strings (if" +\
                                 " prior is multivariate).")
            if prior.numparams == 1:
                if type(params) is str:
                    self._check_name(params)
                    self._data.append((prior, [params], transforms))
                elif type(params) in list_types:
                    if len(params) > 1:
                        raise ValueError("The prior given to a PriorSet was" +\
                                         " univariate, but more than one " +\
                                         "parameter was given.")
                    else:
                        self._check_name(params[0])
                        self._data.append((prior, [params[0]], transforms))
                else:
                    raise ValueError("The type of the parameters given " +\
                                     "to a PriorSet was not recognized.")
            elif type(params) is str:
                raise ValueError("A single parameter was given even though" +\
                                 " the prior given is multivariate " +\
                                 ("(numparams=%i)." % (prior.numparams,)))
            elif type(params) in list_types:
                if (len(params) == prior.numparams):
                    for name in params:
                        self._check_name(name)
                    data_tup = (prior,\
                        [params[i] for i in range(len(params))], transforms)
                    self._data.append(data_tup)
                else:
                    raise ValueError("The number of parameters of the given" +\
                                     (" prior (%i) " % (prior.numparams,)) +\
                                     "was not equal to the number of para" +\
                                     ("meters given (%i)." % (len(params),)))
            else:
                raise ValueError("The params given to a PriorSet" +\
                                 " (along with a prior) was not " +\
                                 "a string nor a list of strings.")
        else:
            raise ValueError("The prior given to a PriorSet" +\
                             " was not recognized as a prior.")
        for iparam in range(prior.numparams):
            # this line looks weird but it works for any input
            self._params.append(self._data[-1][1][iparam])

    def draw(self):
        """
        Draws a point from all priors.
        
        returns a dictionary of random values indexed by parameter name
        """
        point = {}
        for iprior in range(len(self._data)):
            (prior, params, transforms) = self._data[iprior]
            if (prior.numparams == 1):
                point[params[0]] = _apply_inverse_transform(prior.draw(),\
                    transforms[0])
            else:
                this_draw = prior.draw()
                for iparam in range(len(params)):
                    point[params[iparam]] = _apply_inverse_transform(\
                        this_draw[iparam], transforms[iparam])
        return point

    def log_prior(self, point):
        """
        Evaluates the log of the product of the priors contained in this
        PriorSet.
        
        point should be a dictionary of values indexed by the parameter names
        
        returns the total log_prior coming from contributions from all priors
        """
        if type(point) is dict:
            result = 0.
            for iprior in range(len(self._data)):
                (prior, params, transforms) = self._data[iprior]
                if (prior.numparams == 1):
                    result += prior.log_prior(\
                        _apply_transform(point[params[0]], transforms[0]))
                else:
                    result += prior.log_prior(\
                        [_apply_transform(point[params[i]], transforms[i])\
                                                  for i in range(len(params))])
                for i in range(len(params)):
                    result +=\
                        _log_prior_addition(point[params[i]], transforms[i])
            return result
        else:
            raise ValueError("point given to log_prior function " +\
                             "of a PriorSet was not a dictionary " +\
                             "of values indexed by parameter names.")

    def find_prior(self, parameter):
        """
        Finds the prior associated with the given parameter. Also finds the
        index of the parameter in that prior and the transformation applied to
        the parameter.
        
        parameter string name of parameter
        """
        found = False
        for (this_prior, these_params, these_transforms) in self._data:
            for iparam in range(len(these_params)):
                if parameter == these_params[iparam]:
                    return (this_prior, iparam, these_transforms[iparam])
        raise ValueError(("The parameter searched for (%s) " % (parameter,)) +\
                         "in a PriorSet was not found.")
    
    def __getitem__(self, parameter):
        """
        Returns the same thing as: self.find_prior(parameter)
        """
        return self.find_prior(parameter)

    def delete_prior(self, parameter, throw_error=True):
        """
        Deletes a prior from this PriorSet.
        
        parameter a parameter in the prior
        throw_error if True (default), an error is thrown if the parameter
                    is not found
        """
        for iprior in range(len(self._data)):
            (this_prior, these_params, these_transforms) = self._data[iprior]
            if parameter in these_params:
                to_delete = iprior
                break
        try:
            for par in self._data[to_delete][1]:
                self._params.remove(par)
            self._data = self._data[:to_delete] + self._data[to_delete + 1:]
        except:
            if throw_error:
                raise ValueError('The parameter given to ' +\
                                 'PriorSet.delete_prior was not in ' +\
                                 'the PriorSet.')
    
    def __delitem__(self, parameter):
        """
        Deletes the prior associated with the given parameter. For
        documentation, see delete_prior function.
        """
        self.delete_prior(parameter, throw_error=True)
    
    def parameter_strings(self, parameter):
        """
        Makes an informative string about this parameter's place in this
        PriorSet.
        
        parameter string name of parameter
        
        returns (param_string, transform) in tuple form
        """
        string = ""
        (prior, index, transform) = self.find_prior(parameter)
        if prior.numparams != 1:
            string += (self._numerical_adjective(index) + ' param of ')
        string += prior.to_string()
        return (string, transform)
    
    def __eq__(self, other):
        """
        Checks for equality of this PriorSet with other. Returns True if other
        has the same prior_tuples (though they need not be internally stored in
        the same order) and False otherwise.
        """
        def prior_tuples_equal(first, second):
            #
            # Checks whether two prior_tuple's are equal. Returns True if the
            # prior, params, and transforms stored in first are the same as
            # those stored in second and False otherwise.
            #
            fprior, fparams, ftfms = first
            sprior, sparams, stfms = second
            numparams = fprior.numparams
            if sprior.numparams == numparams:
                for iparam in range(numparams):
                    if fparams[iparam] != sparams[iparam]:
                        return False
                    if ftfms[iparam] != sparams[iparam]:
                        return False
                return (fprior == sprior)
            else:
                return False
        if isinstance(other, PriorSet):
            numtuples = len(self._data)
            if len(other._data) == numtuples:
                for iprior_tuple in range(numtuples):
                    match = False
                    prior_tuple = self._data[iprior_tuple]
                    for other_prior_tuple in other._data:
                        if prior_tuples_equal(prior_tuple, other_prior_tuple):
                            match = True
                            break
                    if not match:
                        return False
                return True
            else:
                return False        
        else:
            return False

    def _numerical_adjective(self, num):
        #
        # Creates a numerical adjective, such as '1st', '2nd', '6th' and so on.
        #
        if (type(num) in [int, np.int8, np.int16, np.int32, np.int64]) and\
            (num >= 0):
            base_string = str(num)
            if num == 0:
                return '0th'
            elif num == 1:
                return '1st'
            elif num == 2:
                return '2nd'
            elif num == 3:
                return '3rd'
            else:
                return str(num) + 'th'
        else:
            raise ValueError("Numerical adjectives apply " +\
                             "only to non-negative integers.")

    def _check_name(self, name):
        #
        # Checks the given name to see if it is already taken in the parameters
        # of the priors in this PriorSet.
        #
        if not (type(name) is str):
            raise ValueError("A parameter provided to a " +\
                             "PriorSet was not a string.")
        broken = False
        for iprior in range(len(self._data)):
            for param in self._data[iprior]:
                if name == param:
                    broken = True
                    break
            if broken:
                break
        if broken:
            raise ValueError("The name of a parameter provided" +\
                             " to a PriorSet is already taken.")

