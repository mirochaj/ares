"""
File: $ARES/ares/inference/Priors.py

Author: Keith Tauscher
Affiliation: University of Colorado at Boulder
Created on: Thu Mar 17 12:48:00 MDT 2016
Updated on: Tue Feb 28 01:45:00 MDT 2017

Description: This file contains different useful priors. For examples, see
             below and the test in tests/test_priors.py.

The included univariate priors (and their initializations, equals signs
indicate optional parameters and their default values) are (pdf's apply only
in support):

(1) GammaPrior(shape, scale=1)
    (pdf) --- f(x) = (x/scale)^(shape-1) * e^(-x/scale) / scale / Gamma(shape)
    (support) --- x > 0
    (mean) --- shape * scale
    (variance) --- shape * scale^2

(2) BetaPrior(alpha, beta)
    (pdf) --- f(x) = x^(alpha - 1) * (1 - x)^(beta - 1) / Beta(alpha, beta)
    (support) --- 0 < x < 1
    (mean) --- alpha / (alpha+beta)
    (variance) (alpha * beta) / (alpha + beta)^2 / (alpha + beta + 1)

(3) PoissonPrior(scale)
    (pdf) --  f(k) = scale^k * e^(-scale) / k!
    (support) --- non-negative integer k
    (mean) --- scale
    (variance) --- scale

(4) ExponentialPrior(rate, shift=0)
    (pdf) --- f(x) = rate * e^(-rate * (x - shift))
    (support) --- x>0
    (mean) --- 1 / rate
    (variance) --- 1 / rate^2

(5) DoubleSidedExponentialPrior(mean, variance)
    (pdf) --- f(x) = e^(-|(x - mean) / sqrt(variance/2)|) / (sqrt(2*variance))
    (support) --- real x
    (mean) --- mean
    (variance) --- variance

(6) UniformPrior(low, high)
    (pdf) --- f(x) = 1 / (high - low)
    (support) --- low < x < high
    (mean) --- (low + high) / 2
    (variance) --- (high - low)^2 / 12

(7) GaussianPrior(mean, variance)
    (pdf) --- f(x) = e^(- (x - mean)^2 / (2 * variance)) / sqrt(2pi * variance)
    (support) --- -infty < x < infty
    (mean) --- mean
    (variance) --- variance

(8) TruncatedGaussianPrior(mean, variance, low, high)
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

(5) GriddedPrior(variables, prior=None)
    (pdf) user-defined through prior ndarray, if prior=None, assumed uniform
    (support) must be rectangular; defined through variable ranges in variables
    (mean) unknown a priori
    (variance) unknown a priori

(6) EllipticalPrior(mean, cov)
    (pdf) f(X)=1 when X is inside (X-mean)^T cov^-1 (X-mean)<= N+2
    (support) hyperellipsoid defined above
    (mean) mean
    (variance) cov
"""
import h5py
import numpy as np
import numpy.random as rand
import numpy.linalg as lalg
import scipy.linalg as slalg
from scipy.misc import factorial
from scipy.special import beta as beta_func
from scipy.special import gammaln as log_gamma
from scipy.special import erf, erfinv

int_types = [int, np.int16, np.int32, np.int64]
float_types = [float, np.float32, np.float64]
numerical_types = int_types + float_types
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
            if (range_max == range_max_0) or (range_min == 0):
                raise LookupError("For some reason, range_max-" +\
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
    raise NotImplementedError("Something went wrong! I got " +\
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
    self.to_string() --- string summary of this prior
    self.__eq__(other) --- checks for equality with another object
    self.fill_hdf5_group(group) --- fills given hdf5 group with data from prior
    
    In draw() and log_prior(), point is a configuration. It could be a
    single number for a univariate prior or a numpy.ndarray for a multivariate
    prior.
    """
    def __ne__(self, other):
        """
        This merely enforces that (a!=b) equals (not (a==b)) for all prior
        objects a and b.
        """
        return (not self.__eq__(other))
    
    def save(self, file_name):
        """
        Saves this prior in an hdf5 file using the prior's fill_hdf5_group
        function.
        
        file_name: name of hdf5 file to write
        """
        hdf5_file = h5py.File(file_name, 'w')
        self.fill_hdf5_group(hdf5_file)
        hdf5_file.close()

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
    
    def to_string(self):
        """
        Finds and returns the string representation of this GammaPrior.
        """
        return "Gamma(%.2g, %.2g)" % (self.shape, self.scale)
    
    def __eq__(self, other):
        """
        Checks for equality between other and this object. Returns True if
        if other is a GammaPrior with nearly the same shape and scale (up to
        dynamic range of 10^9) and False otherwise.
        """
        if isinstance(other, GammaPrior):
            shape_close =\
                np.isclose(self.shape, other.shape, rtol=1e-9, atol=0)
            scale_close =\
                np.isclose(self.scale, other.scale, rtol=1e-9, atol=0)
            return shape_close and scale_close
        else:
            return False
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data from this prior. Only things
        to save are shape, scale, and class name.
        
        group: hdf5 file group to fill
        """
        group.attrs['class'] = 'GammaPrior'
        group.attrs['shape'] = self.shape
        group.attrs['scale'] = self.scale

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
    
    def to_string(self):
        """
        Finds and returns a string representation of this BetaPrior.
        """
        return "Beta(%.2g, %.2g)" % (self.alpha, self.beta)
    
    def __eq__(self, other):
        """
        Checks for equality of this object with other. Returns True if other is
        a BetaPrior with nearly the same alpha and beta (down to 10^-9 level)
        and False otherwise.
        """
        if isinstance(other, BetaPrior):
            alpha_close =\
                np.isclose(self.alpha, other.alpha, rtol=1e-9, atol=0)
            beta_close = np.isclose(self.beta, other.beta, rtol=1e-9, atol=0)
            return alpha_close and beta_close
        else:
            return False
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 group with data from this prior. All that is to be
        saved is the class name, alpha, and beta.
        
        group: hdf5 file group to fill
        """
        group.attrs['class'] = 'BetaPrior'
        group.attrs['alpha'] = self.alpha
        group.attrs['beta'] = self.beta

class PoissonPrior(_Prior):
    """
    Prior with support on the nonnegative integers. It has only one parameter,
    the scale, which is both its mean and its variance.
    """
    def __init__(self, scale):
        """
        Initializes new PoissonPrior with given scale.
        
        scale: mean and variance of distribution (must be positive)
        """
        if type(scale) in numerical_types:
            if scale > 0:
                self.scale = (scale * 1.)
            else:
                raise ValueError("scale given to PoissonPrior was not " +\
                                 "positive.")
        else:
            raise ValueError("scale given to PoissonPrior was not a number.")
    
    @property
    def numparams(self):
        """
        Poisson pdf is univariate so numparams always returns 1.
        """
        return 1
    
    def draw(self):
        """
        Draws and returns a value from this distribution using numpy.random.
        """
        return rand.poisson(lam=self.scale)
    
    def log_prior(self, value):
        """
        Evaluates and returns the log of this prior when the variable is value.
        
        value: numerical value of the variable
        """
        if type(value) in int_types:
            if value >= 0:
                return (value * np.log(self.scale)) - self.scale -\
                    log_gamma(value + 1)
            else:
                return -np.inf
        else:
            raise TypeError("value given to PoissonPrior was not an integer.")

    def to_string(self):
        """
        Finds and returns a string version of this PoissonPrior.
        """
        return "Poisson(%.4g)" % (self.scale,)
    
    def __eq__(self, other):
        """
        Checks for equality of this prior with other. Returns True if other is
        a PoissonPrior with the same scale.
        """
        if isinstance(other, PoissonPrior):
            return np.isclose(self.scale, other.scale, rtol=1e-6, atol=1e-6)
        else:
            return False
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this prior. The only
        thing to save is the scale.
        
        group: hdf5 file group to fill
        """
        group.attrs['class'] = 'PoissonPrior'
        group.attrs['scale'] = self.scale
        

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

    def to_string(self):
        """
        Finds and returns a string version of this ExponentialPrior.
        """
        if self.shift != 0:
            return "Exp(%.2g, shift=%.2g)" %\
                (self.rate, self.shift)
        return "Exp(%.2g)" % (self.rate,)
    
    def __eq__(self, other):
        """
        Checks for equality of this prior with other. Returns True if other is
        an ExponentialPrior with the same rate (to 10^9 dynamic range) and
        shift (down to 1e-9) and False otherwise.
        """
        if isinstance(other, ExponentialPrior):
            rate_close = np.isclose(self.rate, other.rate, rtol=1e-9, atol=0)
            shift_close =\
                np.isclose(self.shift, other.shift, rtol=0, atol=1e-9)
            return rate_close and shift_close
        else:
            return False
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this prior. The only
        things to save are the class name, rate, and shift.
        
        group: hdf5 file group to fill
        """
        group.attrs['class'] = 'ExponentialPrior'
        group.attrs['rate'] = self.rate
        group.attrs['shift'] = self.shift


class DoubleSidedExponentialPrior(_Prior):
    """
    Prior on a single distribution with a double-sided exponential
    distribution. Double sided exponential distributions are "peak"ier than
    Gaussians.
    """
    def __init__(self, mean, variance):
        """
        Initializes a new DoubleSidedExponentialPrior with the given
        parameters.
        
        mean: mean, mode and median of the distribution
        variance: variance of distribution
        """
        if type(mean) in numerical_types:
            self.mean = (mean * 1.)
        else:
            raise ValueError('The mean parameter given to a ' +\
                             'DoubleSidedExponentialPrior was not of a ' +\
                             'numerical type.')
        if type(variance) in numerical_types:
            if variance > 0:
                self.variance = (1. * variance)
            else:
                raise ValueError("The variance given to a " +\
                                 "DoubleSidedExponentialPrior was not " +\
                                 "positive.")
        else:
            raise ValueError("The variance given to a " +\
                             "DoubleSidedExponentialPrior was not of " +\
                             "numerical type.")
        self._const_lp_term = (np.log(2) + np.log(self.variance)) / (-2)
    
    @property
    def numparams(self):
        """
        Exponential pdf is univariate so numparams always returns 1.
        """
        return 1
    
    @property
    def root_half_variance(self):
        if not hasattr(self, '_root_half_variance'):
            self._root_half_variance = np.sqrt(self.variance / 2.)
        return self._root_half_variance
    
    def draw(self):
        """
        Draws and returns a value from this distribution using numpy.random.
        """
        return rand.laplace(loc=self.mean, scale=self.root_half_variance)
    
    def log_prior(self, value):
        """
        Evaluates and returns the log of this prior when the variable is value.
        
        value numerical value of the variable
        """
        return self._const_lp_term -\
            (np.abs(value - self.mean) / self.root_half_variance)

    def to_string(self):
        """
        Finds and returns a string version of this DoubleSidedExponentialPrior.
        """
        return "DSExp(%.2g, %.2g)" % (self.mean, self.variance)
    
    def __eq__(self, other):
        """
        Checks for equality of this prior with other. Returns True if other is
        an DoubleSidedExponentialPrior with the same mean and sigma and False
        otherwise.
        """
        if isinstance(other, DoubleSidedExponentialPrior):
            return np.allclose([self.mean, self.variance],\
                [other.mean, other.variance], rtol=1e-6, atol=1e-9)
        else:
            return False
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this prior. The only
        things to save are the class name, rate, and shift.
        
        group: hdf5 file group to fill
        """
        group.attrs['class'] = 'DoubleSidedExponentialPrior'
        group.attrs['mean'] = self.rate
        group.attrs['variance'] = self.variance


class EllipticalPrior(_Prior):
    """
    Prior on a set of variables where the variables are equally likely to be
    at any point within an ellipsoid (defined by mean and cov). It is a uniform
    prior over an arbitrary ellipsoid.
    """
    def __init__(self, mean, cov):
        """
        Initializes this UniformPrior using properties of the ellipsoid
        defining it.
        
        mean the center of the ellipse defining this prior
        cov the covariance describing the ellipse defining this prior. A
            consequence of this definition is that, in order for a point, x, to
            be in the ellipse, (x-mean)^T*cov^-1*(x-mean) < N+2 must be
            satisfied
        """
        try:
            self.mean = np.array(mean)
        except:
            raise TypeError("mean given to EllipticalPrior could not be " +\
                            "cast as a numpy.ndarray.")
        try:
            self.cov = np.array(cov)
        except:
            raise TypeError("cov given to EllipticalPrior could not be " +\
                            "cast as a numpy.ndarray.")
        if (self.cov.shape != (2 * self.mean.shape)) or (self.mean.ndim != 1):
            raise ValueError("The shapes of the mean and cov given to " +\
                             "EllipticalPrior did not make sense. They " +\
                             "should fit the following pattern: " +\
                             "mean.shape=(rank,) and cov.shape=(rank,rank).")
        self._numparams = self.mean.shape[0]
        if self.numparams < 2:
            raise NotImplementedError("The EllipticalPrior doesn't take " +\
                                      "single variable random variates " +\
                                      "since, in the 1D case, it is the " +\
                                      "same as a simple uniform " +\
                                      "distribution but using the " +\
                                      "EllipticalPrior class would involve " +\
                                      "far too much computational overhead.")
        half_rank = self.numparams / 2.
        self.invcov = lalg.inv(self.cov)
        self.log_value = log_gamma(half_rank + 1) -\
            (half_rank * (np.log(np.pi) + np.log(self.numparams + 2))) -\
            (lalg.slogdet(self.cov)[1] / 2.)
        self.sqrtcov = slalg.sqrtm(self.cov)

    @property
    def numparams(self):
        """
        The number of parameters which are represented in this prior.
        """
        if not hasattr(self, '_numparams'):
            self._numparams = len(self.mean)
        return self._numparams

    def draw(self):
        """
        Draws a random vector from this uniform elliptical distribution. By the
        definition of this class, the point it draws is equally likely to lie
        anywhere inside the ellipse defining this prior.
        
        returns numpy.ndarray of containing random variates for each parameter
        """
        xi = _normed(rand.randn(self.numparams))
        # xi is now random directional unit vector
        radial_cdf = rand.rand()
        max_z_radius = np.sqrt(self.numparams + 2)
        fractional_radius = np.power(radial_cdf, 1. / self.numparams)
        deviation = max_z_radius * fractional_radius * np.dot(xi, self.sqrtcov)
        return self.mean + deviation
    
    def log_prior(self, value):
        """
        Evaluates the log of this prior at the given value.
        
        value the vector value of parameters at which to calculate the
              numerical value of this prior
        
        returns if value is inside ellipse, ln(V) where V is volume of
                                            ellipsoid
                if value is outside ellipse, -np.inf
        """
        centered_value = np.array(value) - self.mean
        matprod = np.dot(np.dot(centered_value, self.invcov), centered_value)
        if (matprod <= (self.numparams + 2)):
            return self.log_value
        else:
            return -np.inf

    def to_string(self):
        """
        Gives a simple string (of the form: "N-dim elliptical" where N is the
        number of parameters) summary of this prior.
        """
        return ('%i-dim elliptical' % (self.numparams,))
    
    def __eq__(self, other):
        """
        Checks for equality of this prior with other. Returns True if other is
        an EllipticalPrior of the same dimension with the same mean (down to
        10^-9 level) and covariance (down to dynamic range of 10^-12) and False
        otherwise.
        """
        if isinstance(other, EllipticalPrior):
            if self.numparams == other.numparams:
                mean_close =\
                    np.allclose(self.mean, other.mean, rtol=0, atol=1e-9)
                cov_close = np.allclose(self.cov, other.cov, rtol=1e-12, atol=0)
                return mean_close and cov_close
            else:
                return False
        else:
            return False
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data about this prior. The data to
        be saved includes the class name, mean, and covariance of this prior.
        
        group: hdf5 file group to fill
        """
        group.attrs['class'] = 'EllipticalPrior'
        group.create_dataset('mean', data=self.mean)
        group.create_dataset('covariance', data=self.cov)

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
    
    def to_string(self):
        """
        Finds and returns a string representation of this UniformPrior.
        """
        return "Unif(%.2g, %.2g)" % (self.low, self.high)
    
    def __eq__(self, other):
        """
        Checks for equality of this prior with other. Returns True if other is
        a UniformPrior with the same high and low (down to 1e-9 level) and
        False otherwise.
        """
        if isinstance(other, UniformPrior):
            low_close = np.isclose(self.low, other.low, rtol=0, atol=1e-9)
            high_close = np.isclose(self.high, other.high, rtol=0, atol=1e-9)
            return low_close and high_close
        else:
            return False
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data from this prior. All that
        needs to be saved is the class name and high and low values.
        
        group: hdf5 file group to fill
        """
        group.attrs['class'] = 'UniformPrior'
        group.attrs['low'] = self.low
        group.attrs['high'] = self.high

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
        if (self.lo is not None and value < self.lo) or\
                (self.hi is not None and value > self.hi):
            return -np.inf
        return self._cons_lp_term - ((value - self.mean) ** 2) / (2 * self.var)

    def to_string(self):
        if self.lo is None:
            low_string = "-inf"
        else:
            low_string = "%.1g" % (self.lo,)
        if self.hi is None:
            high_string = "inf"
        else:
            high_string = "%.1g" % (self.hi,)
        return "Normal(%.2g, %.2g) on [%s,%s]" %\
            (self.mean, self.var, low_string, high_string)
    
    def __eq__(self, other):
        """
        Checks for equality of this prior to other. Returns True if other is a
        TruncatedGaussianPrior with the same mean (down to 10^-9 level) and
        variance (down to 10^-12 dynamic range), and hi and lo (down to 10^-9
        level) and False otherwise.
        """
        if isinstance(other, TruncatedGaussianPrior):
            mean_close = np.isclose(self.mean, other.mean, rtol=0, atol=1e-9)
            var_close = np.isclose(self.var, other.var, rtol=1e-12, atol=0)
            if self.hi is None:
                hi_close = (other.hi is None)
            elif other.hi is not None:
                hi_close = np.isclose(self.hi, other.hi, rtol=0, atol=1e-9)
            else:
                # since self.hi is not None in this block, just return False
                return False
            if self.lo is None:
                lo_close = (other.lo is None)
            elif other.lo is not None:
                lo_close = np.isclose(self.lo, other.lo, rtol=0, atol=1e-9)
            else:
                return False
            return mean_close and var_close and hi_close and lo_close
        else:
            return False
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data from this prior. The low, 
        high, mean, and variance values need to be saved along with the class
        name.
        """
        group.attrs['class'] = 'TruncatedGaussianPrior'
        group.attrs['low'] = self.lo
        group.attrs['high'] = self.hi
        group.attrs['mean'] = self.mean
        group.attrs['variance'] = self.var


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
        self.logdetcov = lalg.slogdet(self.covariance)[1]
    
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
    
    def __add__(self, other):
        """
        Adds other to this Gaussian variate. The result of this operation is a
        Gaussian with a shifted mean but identical covariance.
        
        other: must be castable to the 1D array shape of the Gaussian variate
               described by this prior
        """
        return GaussianPrior(self.mean.A[0] + other, self.covariance.A)
    
    def __radd__(self, other):
        """
        Returns the same thing as __add__ (this makes addition commutative).
        """
        return self.__add__(other)
    
    def __mul__(self, other):
        """
        Multiplies the Gaussian random variate described by this prior by the
        given object.

        other: if other is a constant, the returned GaussianPrior is the same
                                       as this one with the mean multiplied by
                                       other and the covariance multiplied by
                                       other**2
               if other is a 1D numpy.ndarray, it must be of the same length
                                               as the dimension of this
                                               Gaussian. In this case, the
                                               returned Gaussian is the
                                               distribution of the dot product
                                               of the this Gaussian variate
                                               with other
               if other is a 2D numpy.ndarray, it must have shape
                                               (newparams, self.numparams)
                                               where newparams<=self.numparams
                                               The returned Gaussian is the
                                               distribution of other (matrix)
                                               multiplied with this Gaussian
                                               variate
        
        returns: GaussianPrior representing the multiplication of this Gaussian
                 variate by other
        """
        if type(other) in [list, tuple, np.ndarray]:
            other = np.array(other)
            if other.ndim == 1:
                if len(other) == self.numparams:
                    new_mean = np.dot(self.mean.A[0], other)
                    new_covariance =\
                        np.dot(np.dot(self.covariance.A, other), other)
                else:
                    raise ValueError("Cannot multiply Gaussian distributed " +\
                                     "random vector by a vector of " +\
                                     "different size.")
            elif other.ndim == 2:
                if other.shape[1] == self.numparams:
                    if other.shape[0] <= self.numparams:
                        # other is a matrix with self.numparams columns
                        new_mean = np.dot(other, self.mean.A[0])
                        new_covariance =\
                            np.dot(other, np.dot(self.covariance.A, other.T))
                    else:
                        raise ValueError("Cannot multiply Gaussian " +\
                                         "distributed random vector by " +\
                                         "matrix which will expand the " +\
                                         "number of parameters because the " +\
                                         "covariance matrix of the result " +\
                                         "would be singular.")
                else:
                    raise ValueError("Cannot multiply given matrix with " +\
                                     "Gaussian distributed random vector " +\
                                     "because the axis of its second " +\
                                     "dimension is not the same length as " +\
                                     "the random vector.")
            else:
                raise ValueError("Cannot multiply Gaussian distributed " +\
                                 "random vector by a tensor with more than " +\
                                 "3 indices.")
        else:
            # assume other is a constant
            new_mean = self.mean.A[0] * other
            new_covariance = self.covariance.A * (other ** 2)
        return GaussianPrior(new_mean, new_covariance)
        
    
    def __rmul__(self, other):
        """
        Returns the same thing as __mul__ (this makes multiplication
        commutative).
        """
        return self.__mul__(other)

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
        return -1. * ((self.logdetcov / 2.) + expon +\
            ((self.numparams * np.log(two_pi)) / 2.))

    def to_string(self):
        """
        Finds and returns the string representation of this GaussianPrior.
        """
        if self.numparams == 1:
            return "Normal(mean=%.3g,variance=%.3g)" %\
                (self.mean.A[0,0], self.covariance.A[0,0])
        else:
            return "%i-dim Normal" % (len(self.mean.A[0]),)
    
    def __eq__(self, other):
        """
        Checks for equality of this prior with other. Returns True if other is
        a GaussianPrior with the same mean (down to 10^-9 level) and variance
        (down to 10^-12 dynamic range) and False otherwise.
        """
        if isinstance(other, GaussianPrior):
            if self.numparams == other.numparams:
                mean_close =\
                    np.allclose(self.mean.A, other.mean.A, rtol=0, atol=1e-9)
                covariance_close = np.allclose(self.covariance.A,\
                    other.covariance.A, rtol=1e-12, atol=0)
            return mean_close and covariance_close
        else:
            return False
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data from this prior. The fact
        that this is a Gaussian is saved along with the mean and covariance.
        
        group: hdf5 file group to fill
        """
        group.attrs['class'] = 'GaussianPrior'
        group.create_dataset('mean', data=self.mean.A[0])
        group.create_dataset('covariance', data=self.covariance.A)


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

    def to_string(self):
        """
        Finds and returns a string representation of this ParallelepipedPrior.
        """
        return "Parallelepiped(%s, %s, %s)" %\
            (self.center, self.face_directions, self.distance)

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
    
    def __eq__(self, other):
        """
        Checks for equality of this prior with other. Returns True if other is
        a ParallelepipedPrior with the same center, face_directions, and
        distances (to a dynamic range of 10^-9) and False otherwise.
        """
        if isinstance(other, ParallelepipedPrior):
            center_close =\
                np.allclose(self.center, other.center, rtol=1e-9, atol=0)
            face_directions_close = np.allclose(self.face_directions.A,\
                other.face_directions.A, rtol=1e-9, atol=0)
            distances_close =\
                np.allclose(self.distances, other.distances, rtol=1e-9, atol=0)
            return center_close and face_directions_close and distances_close
        else:
            return False
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data from this prior. The class
        name of the prior is saved along with the center, face_directions, and
        distances.
        
        group: hdf5 file group to fill
        """
        group.attrs['class'] = 'ParallelepipedPrior'
        group.create_dataset('center', data=self.center)
        group.create_dataset('face_directions', data=self.face_directions)
        group.create_dataset('distances', data=self.distances)


class LinkedPrior(_Prior):
    """
    Class representing a prior which is shared by an arbitrary number of
    parameters. It piggybacks on another (univariate) prior (called the
    "shared_prior") by drawing from it and evaluating its log_prior while
    ensuring that the variables linked by this prior must be identical.
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

    def to_string(self):
        """
        Finds and returns a string representation of this LinkedPrior.
        """
        return "Linked(%s)" % (self.shared_prior.to_string(),)
    
    def __eq__(self, other):
        """
        Checks for equality of this prior with other. Returns True if other is
        a LinkedPrior with the same number of parameters and the same shared
        prior distribution and False otherwise.
        """
        if isinstance(other, LinkedPrior):
            numparams_equal = (self.numparams == other.numparams)
            shared_prior_equal = (self.shared_prior == other.shared_prior)
            return numparams_equal and shared_prior_equal
        return False
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data from this prior. The class
        name is saved alongside the component priors and the number of
        parameters.
        
        group: hdf5 file group to fill
        """
        group.attrs['class'] = 'LinkedPrior'
        group.attrs['numparams'] = self.numparams
        self.shared_prior.fill_hdf5_group(group.create_group('shared_prior'))


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
        return result

    def to_string(self):
        """
        Finds and returns a string representation of this SequentialPrior.
        """
        return "Sequential(%s)" % (self.shared_prior.to_string(),)
    
    def __eq__(self, other):
        """
        Checks for equality of this prior with other. Returns True if other is
        a SequentialPrior with the same number of parameters and the same
        shared prior distribution and False otherwise.
        """
        if isinstance(other, SequentialPrior):
            numparams_equal = (self.numparams == other.numparams)
            shared_prior_equal = (self.shared_prior == other.shared_prior)
            return numparams_equal and shared_prior_equal
        else:
            return False
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data from this prior. That data
        includes the class name, the number of parameters, and the shared
        prior.
        
        group: hdf5 file group to fill
        """
        group.attrs['class'] = 'SequentialPrior'
        group.attrs['numparams'] = self.numparams
        self.shared_prior.fill_hdf5_group(group.create_group('shared_prior'))
        

class GriddedPrior(_Prior):
    """
    A class representing an arbitrary dimensional (well, up to 32 dimensions)
    probability distribution (of finite support!).
    """
    def __init__(self, variables, pdf=None):
        """
        Initializes a new GriddedPrior using the given variables.
        
        variables list of variable ranges (i.e. len(variables) == ndim
                  and variables[i] is the set of the ith variables)
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
                                     "proportional to the pdf. The shape " +\
                                     ("was %s when it should have been %s" %\
                                     (arrpdf.shape, self.shape,)))
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


    def to_string(self):
        """
        Finds and returns a string representation of this GriddedPrior.
        """
        return "Gridded(user defined)"
    
    def __eq__(self, other):
        """
        Checks for equality of this prior with other. Returns True if other is
        a GriddedPrior with the same variable ranges and pdf and False
        otherwise.
        """
        if isinstance(other, GriddedPrior):
            if self.numparams == other.numparams:
                if self.shape == other.shape:
                    vars_close =\
                        np.allclose(self.vars, other.vars, rtol=0, atol=1e-9)
                    pdf_close =\
                        np.allclose(self.pdf, other.pdf, rtol=0, atol=1e-12)
                    return vars_close and pdf_close
                else:
                    return False
            else:
                return False
        else:
            return False

    def _make_cdf(self):
        #
        # Constructs the cdf array.
        #
        running_sum = 0.
        print 'initializing cdf'
        self.cdf = np.ndarray(len(self.pdf))
        print 'filling cdf'
        for i in range(len(self.pdf)):
            self.cdf[i] = running_sum
            running_sum += (self.pdf[i] * self._pixel_area(i))
        print 'renormalizing pdf and cdf'
        self.cdf = self.cdf / self.cdf[-1]
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
            except LookupError:
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
    
    def fill_hdf5_group(self, group):
        """
        Fills the given hdf5 file group with data from this prior. The class
        name, variables list, and pdf values.
        
        group: hdf5 file group to fill
        """
        group.attrs['class'] = 'GriddedPrior'
        group.attrs['numparams'] = self.numparams
        for ivar in xrange(len(self.vars)):
            group.attrs['variable_%i' % (ivar,)] = self.vars[ivar]
        group.create_dataset('pdf', data=self.pdf)

def load_prior_from_hdf5_group(group):
    """
    Loads a prior from the given hdf5 group.
    
    group: the hdf5 file group from which to load the prior
    
    returns: Prior object of the correct type
    """
    try:
        class_name = group.attrs['class']
    except KeyError:
        raise ValueError("group given does not appear to contain a prior.")
    if class_name == 'GammaPrior':
        shape = group.attrs['shape']
        scale = group.attrs['scale']
        return GammaPrior(shape, scale=scale)
    elif class_name == 'BetaPrior':
        alpha = group.attrs['alpha']
        beta = group.attrs['beta']
        return BetaPrior(alpha, beta)
    elif class_name == 'PoissonPrior':
        scale = group.attrs['scale']
        return PoissonPrior(scale)
    elif class_name == 'ExponentialPrior':
        rate = group.attrs['rate']
        shift = group.attrs['shift']
        return ExponentialPrior(rate, shift=shift)
    elif class_name == 'DoubleSidedExponentialPrior':
        mean = group.attrs['mean']
        variance = group.attrs['variance']
        return DoubleSidedExponentialPrior(mean, variance)
    elif class_name == 'EllipticalPrior':
        mean = group['mean'].value
        covariance = group['covariance'].value
        return EllipticalPrior(mean, covariance)
    elif class_name == 'UniformPrior':
        low = group.attrs['low']
        high = group.attrs['high']
        return UniformPrior(low=low, high=high)
    elif class_name == 'TruncatedGaussianPrior':
        mean = group.attrs['mean']
        variance = group.attrs['variance']
        low = group.attrs['low']
        high = group.attrs['high']
        return TruncatedGaussianPrior(mean, variance, low=low, high=high)
    elif class_name == 'GaussianPrior':
        mean = group['mean'].value
        covariance = group['covariance'].value
        return GaussianPrior(mean, covariance)
    elif class_name == 'ParallelepipedPrior':
        center = group['center'].value
        face_directions = group['face_directions'].value
        distances = group['distances'].value
        return ParallelepipedPrior(center, face_directions, distances)
    elif class_name == 'LinkedPrior':
        numparams = group.attrs['numparams']
        shared_prior = load_prior_from_hdf5_group(group['shared_prior'])
        return LinkedPrior(shared_prior, numparams)
    elif class_name == 'SequentialPrior':
        numparams = group.attrs['numparams']
        shared_prior = load_prior_from_hdf5_group(group['shared_prior'])
        return SequentialPrior(shared_prior=shared_prior, numpars=numparams)
    elif class_name == 'GriddedPrior':
        variables = []
        ivar = 0
        while ('variable_%i' % (ivar,)) in group.attrs:
            variables.append(group.attrs['variable_%i' % (ivar,)])
            ivar += 1
        pdf = group['pdf'].value
        return GriddedPrior(variables=variables, pdf=pdf)
    else:
        raise ValueError("The class of the prior was not recognized.")

