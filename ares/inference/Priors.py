"""
Priors.py

Author: Keith Tauscher
Affiliation: University of Colorado at Boulder
Created on: Thu Mar 17 12:48:00 MDT 2016

Description: This file contains different useful priors. For examples, see below and the test in tests/test_priors.py.

The included univariate priors (and their initializations, equals signs
indicate optional parameters and their default values) are (pdf's apply only
in support):

(1) BetaPrior(alpha, beta)
    (pdf) --- f(x) = x^(alpha - 1) * (1 - x)^(beta - 1) / Beta(alpha, beta)
    (support) --- 0 < x < 1
    (mean) --- alpha / (alpha+beta)
    (variance) (alpha * beta) / (alpha + beta)^2 / (alpha + beta + 1)

(2) GammaPrior(shape, scale=1)
    (pdf) --- f(x) = (x/scale)^(shape-1) * e^(-x/scale) / scale / Gamma(shape)
    (support) --- x > 0
    (mean) --- shape * scale
    (variance) --- shape * scale^2

(3) ExponentialPrior(rate, shift=0)
    (pdf) --- f(x) = rate * e^(-rate * (x - shift))
    (support) --- x>0
    (mean) --- 1 / rate
    (variance) --- 1 / rate^2

(4) UniformPrior(low, high)
    (pdf) --- f(x) = 1 / (high - low)
    (support) --- low < x < high
    (mean) --- (low + high) / 2
    (variance) --- (high - low)^2 / 12

(5) GaussianPrior(mean, variance)
    (pdf) --- f(x) = e^(- (x - mean)^2 / (2 * variance)) / sqrt(2pi * variance)
    (support) --- -infty < x < infty
    (mean) --- mean
    (variance) --- variance

(6) TruncatedGaussianPrior(mean, variance, low, high)
    (pdf) --- rescaled and truncated version of pdf of GaussianPrior
    (support) --- low < x < high
    (mean) --- no convenient expression; in limit, approaches mean
    (variance) --- no convenient expression; in limit, approaches variance

And the following multivariate priors are included:

(1) GaussianPrior(mean, covariance)
    (pdf) ---         e^(- (x-mean)^T covariance^(-1) (x-mean) / 2)
              f(x) = -----------------------------------------------
                      np.sqrt( (2 * pi)^numparams * det(covariance))
    (support) --- x is vector in R^N where N is number of dimensions of mean
    (mean) --- mean
    (variance) --- covariance

(2) ParallelepipedPrior(center, face_directions, distances, norm_dirs=True)
    (pdf) --- f(x) = 1 / det(matrix_of_directions_from_vertex)
    (support) --- in the parallelogram described by
                  | (x - center) dot face_directions[i] | <= distances[i]
                  --- If norm_dirs=False, face_directions can have mag != 1
    (mean) --- center
    (variance) --- no nice expression
                   depends on angles between face directions

(3) LinkedPrior(shared_prior, numparams)
    (pdf) --- f(x1, x2, x3, ...) = shared_prior(x1) * prod_{i=2}^N delta(xi-x1)
    (support) --- x1=x2=x3=...=xN where all are in support of shared_prior
    (mean) (mu, mu, mu, ...) where mu is mean of shared_prior
    (variance) no convenient expression

(4) SequentialPrior(shared_prior, numparams)
    (pdf) --- f(x1, x2, x3, ...) = prod_{i=1}^N shared_prior(xi)
                                                         * is_sorted(x1,x2,...)
    (support) x1<x2<x3<...<xN where all are in support of shared_prior
    (mean) no convenient expression
    (variance) no convenient expression
"""

import numpy as np
import numpy.random as rand
import numpy.linalg as lalg
from scipy.misc import factorial
from scipy.special import beta as beta_func
from scipy.special import gammaln as log_gamma
from scipy.special import erf, erfinv

numerical_types = [int, float, np.int32, np.int64, np.float32, np.float64]
list_types = [list, tuple, np.ndarray, np.matrix]

two_pi = 2 * np.pi

def _normed(vec):
    arrvec = np.array(vec)
    return (arrvec / lalg.norm(arrvec))

def search_sorted(array, value):
    """
    Searches the given sorted array for the given value using a
    BinarySearch which should execute in O(log N).
    
    array a 1D sorted numerical array
    value the numerical value to search for

    returns index of array closest to value
            returns None if value is outside variable bounds
    """
    def index_to_check(rmin, rmax):
        return (rmin + rmax) / 2
    range_min = 0
    range_max_0 = len(array)
    range_max = range_max_0
    numloops = 0
    while numloops < 100:
        numloops += 1
        if (range_max - range_min) == 1:
            if range_max == range_max_0:
                raise NotImplementedError("For some reason, range_max-" +\
                                          "range_min reached 1 before " +\
                                          "the element was found. The " +\
                                          "element being searched for " +\
                                          ("was %s. (min,max)" % (value,) +\
                                          ("=%s" % ((range_min, range_max),))))
            else:
                high_index = range_max
        else:
            high_index = index_to_check(range_min, range_max)
        high_val = array[high_index]
        low_val = array[high_index - 1]
        if value < low_val:
            range_max = high_index
        elif value > high_val:
            range_min = high_index
        else: # low_val <= value <= high_val
            if (2 * (high_val - value)) < (high_val - low_val):
                return high_index
            else:
                return high_index - 1
    raise NotImplementedError("Something went wrong! I " +\
                              "caught a pseudo-infinite loop!")

class _Prior():
    """
    This class exists for error catching. Since it exists as
    a superclass of all the priors, one can call isinstance(to_check, _Prior)
    to see if to_check is indeed a kind of prior.
    
    All subclasses of this one will implement
    
    self.draw() --- draws a point from the distribution
    self.log_prior(point) --- evaluates the log_prior at the given point
    self.numparams --- property, not function
    
    In both of these functions, point is a configuration. It could be a
    single number for a univariate prior or a numpy.ndarray for a multivariate
    prior.
    """
    pass

############################# Univariate Priors ###############################

class GammaPrior(_Prior):
    """
    A prior having a Gamma distribution. This is useful for
    variables which are naturally non-negative.
    """
    def __init__(self, shape, scale=1.):
        """
        Initializes a new gamma prior with the given parameters.
        
        shape the exponent of x in the gamma pdf (must be greater than 0).
        scale amount to scale x by (x is divided by scale where it appears)
              (must be greater than 0).
        """
        self._check_if_greater_than_zero(shape, 'shape')
        self._check_if_greater_than_zero(scale, 'scale')
        self.shape = (shape * 1.)
        self._shape_min_one = self.shape - 1.
        self.scale = (scale * 1.)

    @property
    def numparams(self):
        """
        Gamma pdf is univariate, so numparams always returns 1.
        """
        return 1

    def draw(self):
        """
        Draws and returns a value from this distribution using numpy.random.
        """
        return rand.gamma(self.shape, scale=self.scale)

    def log_prior(self, value):
        """
        Evaluates and returns the log of this prior when the variable is value.
        
        value numerical value of the variable
        """
        return (self._shape_min_one * np.log(value)) -\
               (self.shape * np.log(self.scale)) -\
               (value / self.scale) - log_gamma(self.shape)

    def _check_if_greater_than_zero(self, value, name):
        #
        # Function which checks if the given value is positive.
        # If so, the function runs smoothly and returns nothing.
        # Otherwise, useful errors are raised.
        #
        if type(value) in numerical_types:
            if value <= 0:
                raise ValueError(("The %s given to " % (name,)) +\
                                 "a GammaPrior wasn't positive.")
        else:
            raise ValueError(("The %s given to a " % (name,)) +\
                             "GammaPrior wasn't of a numerical type.")

class BetaPrior(_Prior):
    """
    Class representing a prior with a beta distribution. Useful for
    parameters which must lie between 0 and 1. Classically, this is the
    distribution of the probability of success in a binary experiment where
    alpha successes and beta failures have been observed.
    """
    def __init__(self, alpha, beta):
        """
        Initializes a new BetaPrior.
        
        alpha, beta parameters representing number of successes/failures
                    (both must be greater than 0)
        """
        if (type(alpha) in numerical_types) and\
            (type(beta) in numerical_types):
            if (alpha >= 0) and (beta >= 0):
                self.alpha = (alpha * 1.)
                self.beta  = (beta * 1.)
                self._alpha_min_one = self.alpha - 1.
                self._beta_min_one = self.beta - 1.
            else:
                raise ValueError('The alpha or beta parameter given ' +\
                                 'to a BetaPrior was not non-negative.')
        else:
            raise ValueError('The alpha or beta parameter given to a ' +\
                             'BetaPrior were not of a numerical type.')
    
    @property
    def numparams(self):
        """
        Beta pdf is univariate, so numparams always returns 1.
        """
        return 1
    
    def draw(self):
        """
        Draws and returns a value from this distribution using numpy.random.
        """
        return rand.beta(self.alpha, self.beta)
    
    def log_prior(self, value):
        """
        Evaluates and returns the log of this prior when the variable is value.
        
        value numerical value of the variable
        """
        if (value <= 0) or (value >= 1):
            return -np.inf
        return (self._alpha_min_one * np.log(value)) +\
               (self._beta_min_one * np.log(1. - value)) -\
               np.log(beta_func(self.alpha, self.beta))

class ExponentialPrior(_Prior):
    """
    Prior on a single distribution with exponential distribution. Exponential
    distributions are ideal for parameters which are naturally non-negative
    (or, if shift is used, are naturally above some certain lower cutoff)
    """
    def __init__(self, rate, shift=0.):
        """
        Initializes a new ExponentialPrior with the given parameters.
        
        rate the rate parameter of the distribution (number multiplied by x in
             exponent of pdf) (must be greater than 0)
        shift lower limit of the support of the distribution (defaults to 0)
        """
        if type(rate) in numerical_types:
            if rate > 0:
                self.rate = (rate * 1.)
            else:
                raise ValueError('The rate parameter given to an ' +\
                                 'ExponentialPrior was not positive.')
        else:
            raise ValueError('The rate parameter given ' +\
                             'to an ExponentialPrior was ' +\
                             'not of a numerical type.')
        if type(shift) in numerical_types:
            self.shift = (1. * shift)
        else:
            raise ValueError('The shift given to an ExponentialPrior' +\
                             ' was not of numerical type.')
    
    @property
    def numparams(self):
        """
        Exponential pdf is univariate so numparams always returns 1.
        """
        return 1
    
    def draw(self):
        """
        Draws and returns a value from this distribution using numpy.random.
        """
        return rand.exponential(scale=(1./self.rate)) + self.shift
    
    def log_prior(self, value):
        """
        Evaluates and returns the log of this prior when the variable is value.
        
        value numerical value of the variable
        """
        val_min_shift = value - self.shift
        if val_min_shift < 0:
            return -np.inf
        return (np.log(self.rate) - (self.rate * val_min_shift))
    

class UniformPrior(_Prior):
    """
    Class representing a uniform prior. Uniform distributions are
    the least informative possible distributions (on a given
    support) and are thus ideal when ignorance abound.
    """
    def __init__(self, low=0., high=1.):
        """
        Creates a new UniformPrior with the given range.
        
        low lower limit of pdf (defaults to 0)
        high upper limit of pdf (defaults to 1)
        """
        if (type(low) in numerical_types) and (type(high) in numerical_types):
            if low < high:
                self.low = low
                self.high = high
            elif high < low:
                self.low = high
                self.high = low
            else:
                raise ValueError('The high and low endpoints ' +\
                                 'of a uniform prior are equal!')
        else:
            raise ValueError('Either the low or high endpoint of a ' +\
                             'UniformPrior was not of a numerical type.')
        self._log_P = - np.log(self.high - self.low)
        self.numparams = 1

    @property
    def numparams(self):
        """
        Only univariate uniform priors are included
        here so numparams always returns 1.
        """
        return 1

    def draw(self):
        """
        Draws and returns a value from this distribution using numpy.random.
        """
        return rand.uniform(low=self.low, high=self.high)


    def log_prior(self, value):
        """
        Evaluates and returns the log of this prior when the variable is value.
        
        value numerical value of the variable
        """
        if (value >= self.low) and (value <= self.high):
            return self._log_P
        return -np.inf

class TruncatedGaussianPrior(_Prior):
    """
    Class representing a truncated Gaussian distribution. These are useful if
    one has some knowledge of the actual value of the parameter but also needs
    it to lie outside a given region.
    """
    def __init__(self, mean, var, low=None, high=None):
        """
        Initializes a new TruncatedGaussianPrior using the given parameters.
        
        mean the mean of the (untruncated!) Gaussian
        variance the variance of the (untrucated!) Gaussian
        low lower limit of distribution. If None then it is assumed to be -inf
        high upper limit of distribution. If None then it is assumed to be +inf
        """
        self.mean = float(mean)
        self.var = float(var)
        
        if low is None:
            self.lo = None
            self._lo_term = -1.
        else:
            self.lo = float(low)
            self._lo_term = erf((self.lo - self.mean) / np.sqrt(2 * self.var))

        if high is None:
            self.hi = None
            self._hi_term = 1.
        else:
            self.hi = float(high)
            self._hi_term = erf((self.hi - self.mean) / np.sqrt(2 * self.var))

        self._cons_lp_term = -(np.log(np.pi * self.var / 2) / 2)
        self._cons_lp_term -= np.log(self._hi_term - self._lo_term)

    @property
    def numparams(self):
        """
        As of now, only univariate TruncatedGaussianPrior's
        are implemented so numparams always returns 1.
        """
        return 1

    def draw(self):
        """
        Draws and returns a value from this distribution using numpy.random.
        """
        unif = rand.rand()
        arg_to_erfinv = (unif * self._hi_term) + ((1. - unif) * self._lo_term)
        return self.mean + (np.sqrt(2 * self.var) * erfinv(arg_to_erfinv))

    def log_prior(self, value):
        """
        Evaluates and returns the log of this prior when the variable is value.
        
        value numerical value of the variable
        """
        if (value < self.lo and self.lo is not None) or\
                (value > self.hi and self.hi is not None):
            return -np.inf
        return self._cons_lp_term - ((value - self.mean) ** 2) / (2 * self.var)

########### Multivariate priors (Gaussian can also be univariate) #############

class GaussianPrior(_Prior):
    """
    A multivariate (or univariate) Gaussian prior. The classic. Useful when
    some knowledge of the parameters exists and those parameters can be any
    real number.
    """
    def __init__(self, mean, covariance):
        """
        Initializes either a univariate or a multivariate Gaussian.

        mean the mean must be either a number (if univariate)
                                  or a 1D array (if multivariate)
        covariance the covariance must be either a number (if univariate)
                                              or a 2D array (if multivariate)
        """
        if type(mean) in numerical_types:
            self._check_covariance_when_mean_has_size_1(mean,\
                                                        covariance)
        elif type(mean) in list_types:
            arrmean = np.array(mean)
            if arrmean.ndim != 1:
                raise ValueError("The mean of a Gaussian prior" +\
                                 " was not 1 or 2 dimensional.")
            elif arrmean.size == 0:
                raise ValueError("The mean of a Gaussian prior was " +\
                                 "set to something like an empty array.")
            elif arrmean.size == 1:
                self._check_covariance_when_mean_has_size_1(mean[0],\
                                                            covariance)
            elif type(covariance) in list_types:
                arrcov = np.array(covariance)
                if arrcov.shape == (len(arrmean), len(arrmean)):
                    self.mean = np.matrix(arrmean)
                    self._numparams = len(arrmean)
                    self.covariance = np.matrix(arrcov)
                else:
                    raise ValueError("The covariance given to a Gaussian " +\
                                     "prior was not castable to an array of" +\
                                     " the correct shape. It should be a " +\
                                     "square shape with the same side " +\
                                     "length as length of mean.")
            else:
                raise ValueError("The mean of a Gaussian prior " +\
                                 "is array-like but its covariance" +\
                                 " isn't matrix like.")
        else:
            raise ValueError("The mean of a Gaussian prior " +\
                             "is not of a recognizable type.")
        self.invcov = lalg.inv(self.covariance)
        self.detcov = lalg.det(self.covariance)
    
    def _check_covariance_when_mean_has_size_1(self, true_mean, covariance):
        #
        # If the mean is a single number, then the covariance should be
        # castable into a single number as well. This function checks that and
        # raises an error if something unexpected happens. This function sets
        # self.mean and self.covariance.
        #
        # true_mean the single number mean (should be a numerical_type)
        # covariance the covariance which *should* be castable into a number
        #
        if type(covariance) in numerical_types:
            # covariance is single number, as it should be
            self.covariance = np.matrix([[covariance]])
        elif type(covariance) in list_types:
            # covariance should be number but at first glance, it isn't
            arrcov = np.array(covariance)
            if arrcov.size == 1:
                self.covariance = np.matrix([[arrcov[(0,) * arrcov.ndim]]])
            else:
                raise ValueError("The mean of a Gaussian prior was set " +\
                                 "to a number but the covariance can't " +\
                                 "be cast into a number.")
        else:
            raise ValueError("The covariance of a Gaussian prior" +\
                             " is not of a recognizable type.")
        self.mean = np.matrix([true_mean])
        self._numparams = 1

    @property
    def numparams(self):
        """
        Finds and returns the number of parameters which this Gaussian
        describes (same as dimension of mean and covariance).
        """
        if not hasattr(self, '_numparams'):
            raise AttributeError("For some reason, I don't know how" +\
                                 " many parameters this GaussianPrior has!")
        return self._numparams


    def draw(self):
        """
        Draws a point from this distribution using numpy.random.

        returns a numpy.ndarray containing the values from this draw
        """
        if (self.numparams == 1):
            loc = self.mean.A[0,0]
            scale = np.sqrt(self.covariance.A[0,0])
            return rand.normal(loc=loc, scale=scale)
        return rand.multivariate_normal(self.mean.A[0,:], self.covariance.A)

    def log_prior(self, point):
        """
        Evaluates the log prior at the given point.
        
        point single number if univariate, numpy.ndarray if multivariate
        
        returns the log of the prior at the given point
        """
        if type(point) in numerical_types:
            minus_mean = np.matrix([point]) - self.mean
        elif type(point) in list_types:
            minus_mean = np.matrix(point) - self.mean
        else:
            raise ValueError("The type of point provided to a " +\
                             "GaussianPrior was not of a numerical type " +\
                             "(should be if prior is univariate) or of a " +\
                             "list type (should be if prior is multivariate).")
        expon = np.float64(minus_mean * self.invcov * minus_mean.T) / 2.
        return -1. * (np.log(self.detcov) / 2. + expon +\
            ((self.numparams * np.log(two_pi)) / 2.))


class ParallelepipedPrior(_Prior):
    """
    Class representing a uniform prior over a parallelepiped shaped region.
    The region is defined by constraints on linear combinations of the
    variables. See __init__ for more details.
    """
    def __init__(self, center, face_directions, distances, norm_dirs=True):
        """
        Initializes a new ParallelepipedPrior.
        
        center the vector to the center of the region
        face_directions list of directions to the faces of the parallelepiped
        distances distances from center in given directions
        norm_dirs if True, then face_directions are normalized. This means that
                           the distances provided to this
                           method are "true distances"
                  if False, then face_directions are not normalized so the
                            region condition
                            dot(face_directions[i], from_center) < distances[i]
                            implies that distances are measured "in terms of
                            the combined quantity"
                            dot(face_directions[i], from_center)
        """
        if (type(center) in list_types):
            to_set = np.array(center)
            if (to_set.ndim == 1):
                self.center = to_set
                self._numparams = len(self.center)
            else:
                raise ValueError('The number of dimensions of the center ' +\
                                 'given to a ParallelepipedPrior is not 1.' +\
                                 (' It is %i dimensional.' % (to_set.ndim,)))
        else:
            raise ValueError('A ParallelepipedPrior was given with' +\
                             ' a center of an unrecognizable type.')
        if (type(face_directions) in list_types):
            to_set = np.matrix(face_directions)
            if (to_set.shape == ((self.numparams,) * 2)):
                if norm_dirs:
                    self.face_directions =\
                        np.matrix([_normed(face_directions[i])\
                                   for i in range(self.numparams)])
                else:
                    self.face_directions =\
                        np.matrix([face_directions[i]\
                                   for i in range(self.numparams)])
            else:
                raise ValueError('The shape of the face directions in ' +\
                                 'matrix form was not the expected value, ' +\
                                 'which is (self.numparams, self.numparams).')
        else:
            raise ValueError('A ParallelepipedPrior was given ' +\
                             'face_directions of an unrecognizable type.')
        if (type(distances) in list_types):
            arrdists = np.array(distances)
            if (arrdists.ndim == 1) and (len(arrdists) == self.numparams):
                self.distances = arrdists
            else:
                raise ValueError('distances given to ParallelepipedPrior' +\
                                 ' are either of the wrong dimension' +\
                                 ' or the wrong length.')
        self.inv_face_directions = lalg.inv(self.face_directions)

    @property
    def numparams(self):
        """
        Finds and returns the number of parameters to which this
        ParallelepipedPrior applies.
        """
        if not hasattr(self, '_numparams'):
            raise AttributeError("For some reason, I don't know how many " +\
                                 "params this ParallelepipedPrior describes!")
        return self._numparams

    @property
    def matrix(self):
        """
        Finds the matrix whose rows are vectors pointing from the vertex to
        all adjacent vertices.
        
        returns the matrix which has the directions from the vertex as its rows
        """
        if not hasattr(self, '_matrix'):
            def row(index):
                #
                # Finds the index'th row of the matrix. Essentially, this is
                # the vector from the vertex to the index'th adjacent vertex.
                #
                mod_dists = self.distances.copy()
                mod_dists[index] = (mod_dists[index] * (-1))
                from_cent = self.inv_face_directions * np.matrix(mod_dists).T
                from_cent = np.array(from_cent).squeeze()
                return self.center + from_cent - self.vertex
            self._matrix = np.matrix([row(i) for i in range(self.numparams)])
        return self._matrix

    @property
    def vertex(self):
        """
        Finds and returns the vertex which satisfies:
        
        (vec(v)-vec(c)) dot face_directions[i] == distances[i]   for all i
        
        where vec(v) is the vector to the vertex and vec(c) is the vector to
        the center
        """
        if not hasattr(self, '_vertex'):
            from_cent = self.inv_face_directions * np.matrix(self.distances).T
            from_cent = np.array(from_cent).squeeze()
            self._vertex = self.center + from_cent
        return self._vertex

    @property
    def area(self):
        """
        Finds the "area" (more like hypervolume in the general case) of the
        Parallelepiped shaped region described by this ParallelepipedPrior.
        """
        if not hasattr(self, '_area'):
            self._area = np.abs(lalg.det(self.matrix))
        return self._area

    def draw(self):
        """
        Draws a value from the parallelepiped this object describes (uniform
        distribution over support).
        
        returns random draw in form of numpy.ndarray
        """
        transformed_point = rand.rand(self.numparams)
        ret_val = self.vertex
        for i in range(self.numparams):
            ret_val = ret_val + (transformed_point[i] * self.matrix.A[i,:])
        return ret_val

    def log_prior(self, point):
        """
        Computes the log prior at the given point.
        
        point the point at which to evaluate the log prior; a numpy.ndarray
        
        returns log prior at the given point
        """
        if self._in_region(point):
            return -np.log(self.area)
        return -np.inf

    def _in_region(self, point):
        #
        # Finds if the given point is in the region defined by this
        # ParallelepipedPrior.
        #
        # point the point to test for inclusion
        #
        # returns True if point in region, False otherwise
        #
        if type(point) not in list_types:
            raise ValueError('point given to log_prior ' +\
                             'was not of an arraylike type.')
        arrpoint = np.array(point)
        if (arrpoint.ndim != 1) or (len(arrpoint) != self.numparams):
            raise ValueError('The point given is either of the ' +\
                             'wrong direction or the wrong length.')
        from_center = arrpoint - self.center
        return_val = True
        for i in range(self.numparams):
            dotp = np.dot(from_center, self.face_directions.A[i,:])
            return_val =\
                (return_val and (np.abs(dotp) <= np.abs(self.distances[i])))
        return return_val


class LinkedPrior(_Prior):
    """
    Class representing a prior which is shared by an arbitrary number of
    parameters. It piggybacks on another prior (called the "shared_prior") by
    drawing from it and evaluating its log_prior while ensuring that the
    variables linked by this prior must be identical.
    """
    def __init__(self, shared_prior, numpars):
        """
        Initializes a new linked prior with the given shared_prior and number
        of parameters.
        
        shared_prior the prior which describes how the individual values are
                     distributed (must be a _Prior object)
        numpars the number of parameters which this prior describes
        """
        if isinstance(shared_prior, _Prior):
            if shared_prior.numparams == 1:
                self.shared_prior = shared_prior
            else:
                raise NotImplementedError("The shared_prior provided to a " +\
                                          "LinkedPrior was multivariate (I " +\
                                          "don't know how to deal with this).")
        else:
            raise ValueError("The shared_prior given to a LinkedPrior was " +\
                             "not recognizable as a prior. Be sure to " +\
                             "use the priors in ares/inference/Priors.py")
        if (type(numpars) in numerical_types):
            if numpars > 1:
                self._numparams = numpars
            else:
                raise ValueError("A LinkedPrior was initialized " +\
                                 "with only one parameter. " +\
                                 "Is this really what you want?")
        else:
            raise ValueError("The type of the number of parameters " +\
                             "given to a LinkedPrior was not numerical.")

    @property
    def numparams(self):
        """
        Finds and returns the number of parameters which this prior describes.
        """
        return self._numparams

    def draw(self):
        """
        Draws value from shared_prior and assigns that value to all parameters.
        
        returns numpy.ndarray of values (all are equal by design)
        """
        return np.ones(self.numparams) * self.shared_prior.draw()

    def log_prior(self, value):
        """
        Evaluates and returns the log_prior at the given value.
        
        value can be 0D or 1D (if 1D, all values must be identical for
              log_prior != -np.inf)
        
        returns the log_prior at the given value (ignoring delta functions)
        """
        if type(value) in numerical_types:
            return self.shared_prior.log_prior(value)
        elif type(value) in list_types:
            if (len(value) == self.numparams):
                for ival in range(len(value)):
                    if value[ival] != value[0]:
                        return -np.inf
                return self.shared_prior.log_prior(value[0])
            else:
                raise ValueError("The length of the point given to a " +\
                                 "LinkedPrior was not the same as the " +\
                                 "LinkedPrior's number of parameters.")
        else:
            raise ValueError("The point provided to a LinkedPrior " +\
                             "was not of a numerical type or a list type.")


class SequentialPrior(_Prior):
    """
    Class representing a prior on parameters which must be in a specific order.
    """
    def __init__(self, shared_prior=None, numpars=2):
        """
        shared_prior the prior from which values will be drawn before they are
                     sorted (must be univariate) (defaults to Unif(0,1))
        numpars the number of parameters which this SequentialPrior describes
        """
        if shared_prior is None:
            self.shared_prior = UniformPrior(0., 1.)
        elif isinstance(shared_prior, _Prior):
            if shared_prior.numparams == 1:
                self.shared_prior = shared_prior
            else:
                raise NotImplementedError("The shared_prior provided " +\
                                          "to a SequentialPrior was " +\
                                          "multivariate (I don't know " +\
                                          "how to deal with this!).")
        else:
            raise ValueError("The shared_prior given to a SequentialPrior " +\
                             "was not recognizable as a prior. Be sure to " +\
                             "use the priors in ares/inference/Priors.py")
        if (type(numpars) in numerical_types):
            if int(numpars) > 1:
                self._numparams = int(numpars)
            else:
                raise ValueError("A SequentialPrior was initialized " +\
                                 "with only one parameter. " +\
                                 "Is this really what you want?")
        else:
            raise ValueError("The type of the number of parameters " +\
                             "given to a SequentialPrior was not numerical.")

    @property
    def numparams(self):
        """
        Finds and returns the number of parameters which are described by
        this SequentialPrior.
        """
        return self._numparams
    
    def draw(self):
        """
        Draws values from shared_prior and sorts them.
        
        returns numpy.ndarray of values (sorted by design)
        """
        unsorted = [self.shared_prior.draw() for i in range(self.numparams)]
        return np.sort(np.array(unsorted))

    def log_prior(self, point):
        """
        Evaluates and returns the log_prior at the given point. Point must be
        a numpy.ndarray (or other list-type) and if they are sorted, the
        log_prior returns -inf.
        """
        if type(point) in list_types:
            if len(point) == self.numparams:
                if all([point[ip] <= point[ip+1]\
                        for ip in range(len(point)-1)]):
                    result = np.log(factorial(self.numparams))
                    for ipar in range(self.numparams):
                        result += self.shared_prior.log_prior(point[ipar])
                else:
                    return -np.inf
            else:
                raise ValueError("The length of the point provided to a " +\
                                 "SequentialPrior was not the same as the " +\
                                 "SequentialPrior's number of parameters")
        else:
            raise ValueError("The point given to a SequentialPrior " +\
                             "was not of a list type.")

class GriddedPrior(_Prior):
    """
    A class representing an arbitrary dimensional (well, up to 32 dimensions)
    probability distribution (of finite support!).
    """
    def __init__(self, variables, pdf=None):
        """
        Initializes a new GriddedPrior using the given variables.
        
        variables list of variable ranges (i.e. len(variables) == ndim
                  and variables[i] is the range of the ith variable)
        pdf numpy.ndarray with same ndim as number of parameters and with
              the ith axis having the same length as the ith variables range
        """
        if type(variables) in list_types:
            self._N = len(variables)
            self.vars = [variables[i] for i in range(len(variables))]
            self.shape =\
                tuple([len(variables[i]) for i in range(self.numparams)])
            self.size = np.prod(self.shape)
            if pdf is None:
                self.pdf = np.ones(self.shape)
            elif type(pdf) in list_types:
                arrpdf = np.array(pdf)
                if arrpdf.shape == self.shape:
                    self.pdf = arrpdf
                else:
                    raise ValueError("The pdf given to a GriddedPrior " +\
                                     "were not of the expected shape. It " +\
                                     "should be an N-dimensional array with" +\
                                     " each dimension given by the length " +\
                                     "of the corresponding variable's " +\
                                     "range. Its values should be " +\
                                     "proportional to the pdf.")
            else:
                raise ValueError("The pdf given to a GriddedPrior were " +\
                                 "not of a list type. It should be " +\
                                 "an N-dimensional array with each " +\
                                 "dimension given by the length of the " +\
                                 "corresponding variable's range. Its " +\
                                 "values should be proportional to the pdf.")
        else:
            raise ValueError("The variables given to a GriddedPrior were " +\
                             "not of a list type. It should be a " +\
                             "list/tuple of variable ranges.")
        self.pdf = self.pdf.flatten()
        self._make_cdf()

    @property
    def numparams(self):
        """
        Finds and returns the number of parameters which this prior describes.
        """
        return self._N

    def draw(self):
        """
        Draws and returns a point from this distribution.
        """
        rval = rand.rand()
        inv_cdf_index = self._inverse_cdf_by_packed_index(rval)
        return self._point_from_packed_index(inv_cdf_index)

    def log_prior(self, point):
        """
        Evaluates and returns the log of the pdf of this distribution at the
        given point.
        
        point numpy.ndarray of variable values describing the point
        
        returns the log of the pdf associated with the pixel containing point
        """
        index = self._packed_index_from_point(point)
        if index is None:
            return -np.inf
        return np.log(self.pdf[index])

    def _make_cdf(self):
        #
        # Constructs the cdf array.
        #
        running_sum = 0.
        self.cdf = []
        for i in range(len(self.pdf)):
            self.cdf.append(running_sum)
            running_sum += (self.pdf[i] * self._pixel_area(i))
        self.cdf = np.array(self.cdf) / self.cdf[-1]
        self.pdf = self.pdf / self.cdf[-1]

    def _unpack_index(self, index):
        #
        # Finds N-dimensional index corresponding to given index.
        #
        if index is None:
            return None
        inds_in_reverse = []
        running_product = self.shape[self._N - 1]
        inds_in_reverse.append(index % running_product)
        for k in range(1, self._N):
            rel_dim = self.shape[self._N - k - 1]
            inds_in_reverse.append((index / running_product) % rel_dim)
            running_product *= rel_dim
        return inds_in_reverse[-1::-1]

    def _pack_index(self, unpacked_indices):
        #
        # Finds single index which is the packed version
        # of unpacked_indices (which should be a list)
        #
        if unpacked_indices is None:
            return None
        cumulative_index = 0
        running_product = 1
        for i in range(self._N - 1, - 1, - 1):
            cumulative_index += (running_product * unpacked_indices[i])
            running_product *= self.shape[i]
        return cumulative_index

    def _unpacked_indices_from_point(self, point):
        #
        # Gets the unpacked indices which is associated with this point.
        #
        unpacked_indices = []
        for ivar in range(self._N):
            try:
                index = search_sorted(self.vars[ivar], point[ivar])
            except:
                return None
            unpacked_indices.append(index)
        return unpacked_indices

    def _packed_index_from_point(self, point):
        #
        # Finds the packed index associated with the given point.
        #
        return self._pack_index(self._unpacked_indices_from_point(point))
        
    def _point_from_packed_index(self, index):
        #
        # Finds the point associated with the given packed index
        #
        int_part = int(index + 0.5)
        unpacked_indices = self._unpack_index(int_part)
        point = [self.vars[i][unpacked_indices[i]] for i in range(self._N)]
        return  np.array(point) + self._continuous_offset(unpacked_indices)

    def _continuous_offset(self, unpacked_indices):
        #
        # Finds a vector offset to simulate a continuous distribution (even
        # though, internally pixels are being used
        #
        return np.array(\
            [self._single_var_offset(i, unpacked_indices[i], rand.rand())\
             for i in range(self._N)])

    def _single_var_offset(self, ivar, index, rval):
        #
        # Finds the offset for a single variable. rval should be Unif(0,1)
        #
        this_var_length = self.shape[ivar]
        if index == 0:
            return (0.5 *\
                    rval * (self.vars[ivar][1] - self.vars[ivar][0]))
        elif index == (this_var_length - 1):
            return ((-0.5) *\
                    rval * (self.vars[ivar][-1] - self.vars[ivar][-2]))
        else:
            return 0.5 * (self.vars[ivar][index]\
                          - (rval * self.vars[ivar][index - 1])\
                          - ((1 - rval) * self.vars[ivar][index + 1]))

    def _pixel_area(self, packed_index):
        #
        # Finds the area of the pixel described by the given index.
        #
        pixel_area = 1.
        unpacked_indices = self._unpack_index(packed_index)
        for ivar in range(self._N):
            this_index = unpacked_indices[ivar]
            if this_index == 0:
                pixel_area *= (0.5 * (self.vars[ivar][1] - self.vars[ivar][0]))
            elif this_index == len(self.vars[ivar]) - 1:
                pixel_area *= (0.5 *\
                               (self.vars[ivar][-1] - self.vars[ivar][-2]))
            else:
                pixel_area *= (0.5 * (self.vars[ivar][this_index + 1] -\
                                      self.vars[ivar][this_index - 1]))
        return pixel_area

    def _inverse_cdf_by_packed_index(self, value):
        #
        # Finds the index where the cdf has the given value.
        #
        return search_sorted(self.cdf, value)

