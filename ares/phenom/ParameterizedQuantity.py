"""

ParameterizedHaloProperty.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Tue Jan 19 09:44:21 PST 2016

Description:

"""

import gc
import numpy as np
from types import FunctionType
from ..util import ParameterFile
from ..util.ParameterFile import get_pq_pars
from ..util.Misc import numeric_types

try:
    from scipy.special import erf
except ImportError:
    pass

def tanh_astep(M, lo, hi, logM0, logdM):
    # NOTE: lo = value at the low-mass end
    return (lo - hi) * 0.5 * (np.tanh((logM0 - np.log10(M)) / logdM) + 1.) + hi
def tanh_rstep(M, lo, hi, logM0, logdM):
    # NOTE: lo = value at the low-mass end
    return hi * lo * 0.5 * (np.tanh((logM0 - np.log10(M)) / logdM) + 1.) + hi

func_options = {
    "pl": "p[0] * (x / p[1])**p[2]",
    "pl_10": "10**(p[0]) * (x / p[1])**p[2]",
    "exp": "p[0] * exp((x / p[1])**p[2])",
    "exp-m": "p[0] * exp(-(x / p[1])**p[2])",
    "exp_flip": "p[0] * exp(-(x / p[1])**p[2])",
    "dpl": "p[0] / ((x / p[1])**-p[2] + (x / p[1])**-p[3])",
    "dpl_arbnorm": "p[0] * (p[4]) / ((x / p[1])**-p[2] + (x / p[1])**-p[3])",
    "pwpl": "p[0] * (x / p[4])**p[1] if x <= p[4] else p[2] * (x / p[4])**p[3]",
    "plexp": "p[0] * (x / p[1])**p[2] * np.exp(-x / p[3])",
    "lognormal": "p[0] * np.exp(-(logx - p[1])**2 / 2. / p[2]**2)",
    "astep": "p0 if x <= p1 else p2",
    "rstep": "p0 * p2 if x <= p1 else p2",
    "plsum": "p[0] * (x / p[1])**p[2] + p[3] * (x / p[4])**p5",
    "ramp": "p0 if x <= p1, p2 if x >= p3, linear in between",
    "p_linear": "(p[3] - p[2])/(p[1] - p[0]) * (x - p[1]) + p[3]",
}

Np_max = 20

optional_kwargs = "pq_val_ceil", "pq_val_floor", "pq_var_ceil", "pq_var_floor"

class BasePQ(object):
    def __init__(self, **kwargs):
        self.args = []
        for i in range(Np_max):
            name = "pq_func_par{}".format(i)
            if name not in kwargs:
                continue

            self.args.append(kwargs[name])

        self.x = kwargs["pq_func_var"]

        self.xlim = (-np.inf, np.inf)
        self.xfill = None
        if "pq_func_var_lim" in kwargs:
            if kwargs["pq_func_var_lim"] is not None:
                self.xlim = kwargs["pq_func_var_lim"]
                self.xfill = kwargs["pq_func_var_fill"]

        for key in optional_kwargs:
            if key not in kwargs:
                setattr(self, key[3:], None)
            else:
                setattr(self, key[3:], kwargs[key])

        if "pq_func_var2" in kwargs:
            self.t = kwargs["pq_func_var2"]

            self.tlim = (-np.inf, np.inf)
            self.tfill = None
            if "pq_func_var2_lim" in kwargs:
                if kwargs["pq_func_var2_lim"] is not None:
                    self.tlim = kwargs["pq_func_var2_lim"]
                    self.tfill = kwargs["pq_func_var2_fill"]

    def get_time_var(self, **kwargs):
        if (self.t == "z") and ("z" in kwargs):
            t = kwargs[self.t]
        elif (self.t == "1+z") and ("z" in kwargs):
            t = 1. + kwargs["z"]
        elif (self.t == 'a') and ("z" in kwargs):
            t = 1. / (1. + kwargs["z"])
        else:
            raise KeyError(f"Time variable {self.t} not available given input kwargs.")

        return t

class PowerLaw(BasePQ):
    def __call__(self, **kwargs):
        if self.x == "1+z":
            x = 1. + kwargs["z"]
        else:
            x = kwargs[self.x]

        if type(x) in [int, float, np.float64]:
            if not (self.xlim[0] <= x <= self.xlim[1]):
                x = self.xfill
            ok = 1.
        else:
            ok = np.logical_and(self.xlim[0] <= x, x <= self.xlim[1])
            if self.xfill is not None:
                x[~ok] = self.xfill
                ok = np.ones_like(x)

        return ok * self.args[0] * np.power(x / self.args[1],
            self.args[2])

class PowerLaw10(BasePQ):
    def __call__(self, **kwargs):
        if self.x == "1+z":
            x = 1. + kwargs["z"]
        else:
            x = kwargs[self.x]

        if not (self.xlim[0] <= x <= self.xlim[1]):
            return self.xfill

        return 10**(self.args[0]) * (x / self.args[1])**self.args[2]

class PowerLawEvolvingNorm(BasePQ):
    def __call__(self, **kwargs):
        if self.x == "1+z":
            x = 1. + kwargs["z"]
        else:
            x = kwargs[self.x]

        if self.t == "1+z":
            t = 1. + kwargs["z"]
        else:
            t = kwargs[self.t]

        p0 = self.args[0] * (t / self.args[3])**self.args[4]

        return p0 * (x / self.args[1])**self.args[2]

class PowerLawEvolvingSlope(BasePQ):
    def __call__(self, **kwargs):
        if self.x == "1+z":
            x = 1. + kwargs["z"]
        else:
            x = kwargs[self.x]

        if self.t == "1+z":
            t = 1. + kwargs["z"]
        else:
            t = kwargs[self.t]

        p2 = self.args[2] * (t / self.args[3])**self.args[4]

        return self.args[0] * (x / self.args[1])**p2

class PowerLawEvolvingNormSlope(BasePQ):
    def __call__(self, **kwargs):
        if self.x == "1+z":
            x = 1. + kwargs["z"]
        else:
            x = kwargs[self.x]

        if self.t == "1+z":
            t = 1. + kwargs["z"]
        else:
            t = kwargs[self.t]

        p0 = self.args[0] * (t / self.args[3])**self.args[4]
        p2 = self.args[2] * (t / self.args[3])**self.args[5]

        return p0 * (x / self.args[1])**p2

class PowerLawEvolvingSlopeWithGradient(BasePQ):
    def __call__(self, **kwargs):
        if self.x == "1+z":
            x = 1. + kwargs["z"]
        else:
            x = kwargs[self.x]

        if self.t == "1+z":
            t = 1. + kwargs["z"]
        else:
            t = kwargs[self.t]

        p2 = self.args[2] * (t / self.args[3])**self.args[4] \
            * (x / self.args[5])**self.args[6]

        return self.args[0] * (x / self.args[1])**p2

class Erf(BasePQ):
    def __call__(self, **kwargs):
        if self.x == "1+z":
            x = 1. + kwargs["z"]
        else:
            x = kwargs[self.x]

        lo = self.args[0]
        hi = self.args[1]
        step = hi - lo
        x50 = self.args[2]
        sigma = self.args[3]

        return lo \
            + step * 0.5 * (1. + erf((np.log10(x) - x50) / np.sqrt(2) / sigma))

class ErfEvolvingAsB13(BasePQ):
    def __call__(self, **kwargs):
        if self.x == "1+z":
            x = 1. + kwargs["z"]
        else:
            x = kwargs[self.x]

        # Need scale factor
        a = 1. / (1. + kwargs['z'])

        # Basic idea here is to have parameters that dictate
        # low-z, medium-z, and high-z behaviour, e.g.,
        # log10(f_star,10) = p[0] + p[5] * (1 - a) \
        #                  + p[9] * np.log(1 + z) + p[13] * z

        lo = self.args[0] + self.args[4] * (1 - a) \
              + self.args[8] * np.log(1 + kwargs['z']) \
              + self.args[12] * kwargs['z']
        hi = self.args[1] + self.args[5] * (1 - a) \
              + self.args[9] * np.log(1 + kwargs['z']) \
              + self.args[13] * kwargs['z']

        lo = np.maximum(0, lo)
        hi = np.minimum(1, hi)

        step = hi - lo

        x50 = self.args[2] + self.args[6] * (1 - a) \
              + self.args[10] * np.log(1 + kwargs['z']) \
              + self.args[14] * kwargs['z']

        sigma = self.args[3] + self.args[7] * (1 - a) \
              + self.args[11] * np.log(1 + kwargs['z']) \
              + self.args[15] * kwargs['z']

        return lo \
            + step * 0.5 * (1. + erf((np.log10(x) - x50) / np.sqrt(2) / sigma))

class ErfEvolvingMidpointSlope(BasePQ):
    def __call__(self, **kwargs):
        if self.x == "1+z":
            x = 1. + kwargs["z"]
        else:
            x = kwargs[self.x]

        if self.t == "1+z":
            t = 1. + kwargs["z"]
        else:
            t = kwargs[self.t]

        p0 = self.args[0] + self.args[2] * (t - self.args[4])
        p1 = self.args[1] + self.args[3] * (t - self.args[4])

        return 0.5 * (1. + erf((np.log10(x) - p0) / np.sqrt(2) / p1))


class Exponential(BasePQ):
    def __call__(self, **kwargs):
        if self.x == "1+z":
            x = 1. + kwargs["z"]
        else:
            x = kwargs[self.x]
        return self.args[0] * np.exp((x / self.args[1])**self.args[2])

class ExponentialInverse(BasePQ):
    def __call__(self, **kwargs):
        if self.x == "1+z":
            x = 1. + kwargs["z"]
        else:
            x = kwargs[self.x]

        return self.args[0] * np.exp(-(x / self.args[1])**self.args[2])

class ExponentialInverseComplement(BasePQ):
    def __call__(self, **kwargs):
        if self.x == "1+z":
            x = 1. + kwargs["z"]
        else:
            x = kwargs[self.x]

        return 1 - self.args[0] * np.exp(-(x / self.args[1])**self.args[2])

class ExponentialInverseComplementEvolvingTurnover(BasePQ):
    def __call__(self, **kwargs):
        if self.x == "1+z":
            x = 1. + kwargs["z"]
        else:
            x = kwargs[self.x]

        if self.t == "1+z":
            t = 1. + kwargs["z"]
        else:
            t = kwargs[self.t]

        p1 = self.args[1] * (t / self.args[3])**self.args[4]

        return 1 - self.args[0] * np.exp(-(x / p1)**self.args[2])


class Normal(BasePQ):
    def __call__(self, **kwargs):
        if self.x == "1+z":
            x = 1. + kwargs["z"]
        else:
            x = kwargs[self.x]
        return self.args[0] * np.exp(-(x - self.args[1])**2
            / 2. / self.args[2]**2)

class LogNormal(BasePQ):
    def __call__(self, **kwargs):
        if self.x == "1+z":
            x = 1. + kwargs["z"]
        else:
            x = kwargs[self.x]
        logx = np.log10(x)
        return self.args[0] * np.exp(-(logx - self.args[1])**2
            / 2. / self.args[2]**2)

class PiecewisePowerLaw(BasePQ):
    def __call__(self, **kwargs):
        if self.x == "1+z":
            x = 1. + kwargs["z"]
        else:
            x = kwargs[self.x]

        lo = x < self.args[4]
        hi = x >= self.args[4]

        y = lo * self.args[0] * (x / self.args[4])**self.args[1] \
          + hi * self.args[2] * (x / self.args[4])**self.args[3]

        return y

class Ramp(BasePQ):
    def __call__(self, **kwargs):
        if self.x == "1+z":
            x = 1. + kwargs["z"]
        else:
            x = kwargs[self.x]

        # ramp slope
        m = (self.args[2] - self.args[0]) / (self.args[3] - self.args[1])

        lo = x <= self.args[1]
        hi = x >= self.args[3]
        mi = np.logical_and(x > self.args[1], x < self.args[3])

        y = lo * self.args[0] \
          + hi * self.args[2] + mi * (self.args[0] + m * (x - self.args[1]))

        return y

class LogRamp(BasePQ):
    def __call__(self, **kwargs):
        if self.x == "1+z":
            x = 1. + kwargs["z"]
        else:
            x = kwargs[self.x]

        logx = np.log10(x)

        # ramp slope
        alph = np.log10(self.args[2] / self.args[0]) \
             / (self.args[3] - self.args[1])

        lo = logx <= self.args[1]
        hi = logx >= self.args[3]
        mi = np.logical_and(logx > self.args[1], logx < self.args[3])

        fmid = self.args[0] * (x / 10**self.args[1])**alph

        y = lo * self.args[0] + hi * self.args[2] + mi * fmid

        return y

class TanhAbs(BasePQ):
    def __call__(self, **kwargs):
        if self.x == "1+z":
            x = 1. + kwargs["z"]
        else:
            x = kwargs[self.x]

        step = (self.args[0] - self.args[1]) * 0.5
        y = self.args[1] \
          + step * (np.tanh((self.args[2] - x) / self.args[3]) + 1.)
        return y

class TanhRel(BasePQ):
    def __call__(self, **kwargs):
        if self.x == "1+z":
            x = 1. + kwargs["z"]
        else:
            x = kwargs[self.x]

        y = self.args[1] \
          + self.args[1] * self.args[0] * 0.5 \
          * (np.tanh((self.args[2] - x) / self.args[3]) + 1.)

        return y

class LogTanhAbs(BasePQ):
    def __call__(self, **kwargs):
        if self.x == "1+z":
            x = 1. + kwargs["z"]
        else:
            x = kwargs[self.x]

        logx = np.log10(x)

        step = (self.args[0] - self.args[1]) * 0.5
        y = self.args[1] \
          + step * (np.tanh((self.args[2] - logx) / self.args[3]) + 1.)

        return y

class LogTanhAbsEvolvingMidpoint(BasePQ):
    def __call__(self, **kwargs):
        if self.x == "1+z":
            x = 1. + kwargs["z"]
        else:
            x = kwargs[self.x]

        logx = np.log10(x)

        step = (self.args[0] - self.args[1]) * 0.5

        if self.t == "1+z":
            mid = self.args[2] \
                + self.args[4] * ((1. + kwargs["z"]) / self.args[5])
        else:
            raise NotImplemented("help")

        y = self.args[1] \
          + step * (np.tanh((mid - logx) / self.args[3]) + 1.)

        return y

class LogTanhAbsEvolvingMidpointFloorCeiling(BasePQ):
    def __call__(self, **kwargs):
        if self.x == '1+z':
            x = 1. + kwargs['z']
        else:
            x = kwargs[self.x]

        logx = np.log10(x)

        hi = self.args[0] + self.args[6] * ((1. + kwargs['z']) / self.args[5])
        lo = self.args[1] + self.args[7] * ((1. + kwargs['z']) / self.args[5])

        hi = np.minimum(hi, 1.)
        lo = np.maximum(lo, 0.)

        step = (hi - lo) * 0.5

        if self.t == '1+z':
            mid = self.args[2] \
                + self.args[4] * ((1. + kwargs['z']) / self.args[5])
        else:
            raise NotImplemented('help')

        y = lo \
          + step * (np.tanh((mid - logx) / self.args[3]) + 1.)

        return y

class LogSigmoidEvolvingFloorCeilingWidth(BasePQ):
    def __call__(self, **kwargs):
        if self.x == '1+z':
            x = 1. + kwargs['z']
        else:
            x = kwargs[self.x]

        logx = np.log10(x)

        lo = self.args[0] + self.args[5] * ((1. + kwargs['z']) / self.args[4]) \
            + self.args[9] * ((1. + kwargs['z']) / self.args[4])**2
        hi = self.args[1] + self.args[6] * ((1. + kwargs['z']) / self.args[4]) \
            + self.args[10] * ((1. + kwargs['z']) / self.args[4])**2
        mid= self.args[2] + self.args[7] * ((1. + kwargs['z']) / self.args[4]) \
            + self.args[11] * ((1. + kwargs['z']) / self.args[4])**2
        w  = self.args[3] + self.args[8] * ((1. + kwargs['z']) / self.args[4]) \
            + self.args[12] * ((1. + kwargs['z']) / self.args[4])**2

        sigma = 1. / (1. + np.exp(-(logx - mid) / w))

        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, 1)

        y = lo + (hi - lo) * (1. - sigma)

        return y

class LogTanhAbsEvolvingMidpointFloorCeilingWidth(BasePQ):
    def __call__(self, **kwargs):
        if self.x == '1+z':
            x = 1. + kwargs['z']
        else:
            x = kwargs[self.x]

        logx = np.log10(x)

        lo = self.args[0] + self.args[5] * ((1. + kwargs['z']) / self.args[4])
        hi = self.args[1] + self.args[6] * ((1. + kwargs['z']) / self.args[4])

        mid= self.args[2] + self.args[7] * ((1. + kwargs['z']) / self.args[4])
        w  = self.args[3] + self.args[8] * ((1. + kwargs['z']) / self.args[4])

        hi = hi#np.minimum(hi, 1.)
        lo = np.maximum(lo, 0.)
        mid = np.maximum(mid, 0)
        w = np.maximum(w, 0)

        step = (hi - lo)

        # tanh(x) goes from -1 to 1 as x goes from -inf to inf.
        # So, for logx < mid
        y = lo + step * 0.5 * (np.tanh((mid - logx) / w) + 1.)

        return y

class LogTanhAbsEvolvingMidpointFloorCeilingWidthFlex(BasePQ):
    def __call__(self, **kwargs):
        if self.x == '1+z':
            x = 1. + kwargs['z']
        else:
            x = kwargs[self.x]

        logx = np.log10(x)

        hi = self.args[0] + self.args[5] * ((1. + kwargs['z']) / self.args[4]) \
            + self.args[9] * ((1. + kwargs['z']) / self.args[4])**2
        lo = self.args[1] + self.args[6] * ((1. + kwargs['z']) / self.args[4]) \
            + self.args[10] * ((1. + kwargs['z']) / self.args[4])**2
        mid= self.args[2] + self.args[7] * ((1. + kwargs['z']) / self.args[4]) \
            + self.args[11] * ((1. + kwargs['z']) / self.args[4])**2
        w  = self.args[3] + self.args[8] * ((1. + kwargs['z']) / self.args[4]) \
            + self.args[12] * ((1. + kwargs['z']) / self.args[4])**2

        hi = np.minimum(hi, 1.)
        lo = np.maximum(lo, 0.)
        w = np.maximum(w, 0)

        step = (hi - lo) * 0.5

        y = lo + step * (np.tanh((mid - logx) / w) + 1.)

        return y

class LogTanhAbsEvolvingWidth(BasePQ):
    def __call__(self, **kwargs):
        if self.x == "1+z":
            x = 1. + kwargs["z"]
        else:
            x = kwargs[self.x]

        logx = np.log10(x)

        step = (self.args[0] - self.args[1]) * 0.5

        if self.t == "1+z":
            w = self.args[3] \
                + self.args[4] * ((1. + kwargs["z"]) / self.args[5])
        else:
            raise NotImplemented("help")

        y = self.args[1] \
          + step * (np.tanh((self.args[2] - logx) / w) + 1.)

        return y

class LogTanhRel(BasePQ):
    def __call__(self, **kwargs):
        if self.x == "1+z":
            x = 1. + kwargs["z"]
        else:
            x = kwargs[self.x]

        logx = np.log10(x)

        y = self.args[1] \
          + self.args[1] * self.args[0] * 0.5 \
          * (np.tanh((self.args[2] - logx) / self.args[3]) + 1.)

        return y

class StepRel(BasePQ):
    def __call__(self, **kwargs):
        if self.x == "1+z":
            x = 1. + kwargs["z"]
        else:
            x = kwargs[self.x]

        lo = x <= self.args[2]
        hi = x > self.args[2]

        y = lo * self.args[0] * self.args[1] + hi * self.args[1]

        return y

class StepAbs(BasePQ):
    def __call__(self, **kwargs):
        if self.x == "1+z":
            x = 1. + kwargs["z"]
        else:
            x = kwargs[self.x]

        lo = x <= self.args[2]
        hi = x > self.args[2]

        y = lo * self.args[0] + hi * self.args[1]

        return y

class DoublePowerLawPeakNorm(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]

        # This is to conserve memory.
        y  = (x / self.args[1])**-self.args[2]
        y += (x / self.args[1])**-self.args[3]
        np.divide(1., y, out=y)
        y *= 2. * self.args[0]

        return y

class DoublePowerLaw(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]

        normcorr = (((self.args[4] / self.args[1])**-self.args[2] \
                 +   (self.args[4] / self.args[1])**-self.args[3]))

        # This is to conserve memory.
        y  = (x / self.args[1])**-self.args[2]
        y += (x / self.args[1])**-self.args[3]
        np.divide(1., y, out=y)
        y *= normcorr * self.args[0]

        return y

class DoublePowerLawExtended(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]

        normcorr = (((self.args[4] / self.args[1])**-self.args[2] \
                 +   (self.args[4] / self.args[1])**-self.args[3]))

        # This is to conserve memory.
        y  = (x / self.args[1])**-self.args[2]
        y += (x / self.args[1])**-self.args[3]
        np.divide(1., y, out=y)
        y *= normcorr * self.args[0]

        y *= (1. + (x / self.args[5])**self.args[6])**self.args[7]

        return y

class DoublePowerLawExtendedEvolvingNorm(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]

        # Normalization evolves
        normcorr = (((self.args[4] / self.args[1])**-self.args[2] \
                 +   (self.args[4] / self.args[1])**-self.args[3]))

        # This is to conserve memory.
        y  = (x / self.args[1])**-self.args[2]
        y += (x / self.args[1])**-self.args[3]
        np.divide(1., y, out=y)

        if self.t == "1+z":
            y *= normcorr * self.args[0] \
                * ((1. + kwargs["z"]) / self.args[5])**self.args[6]
        else:
            y *= normcorr * self.args[0] \
                * (kwargs[self.t] / self.args[5])**self.args[6]

        y *= (1. + (x / self.args[7])**self.args[8])**self.args[9]

        return y

class DoublePowerLawExtendedEvolvingNormPeakX(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]

        if self.t == "1+z":
            t = 1. + kwargs["z"]
        else:
            t = kwargs[self.t]

        # This is the peak mass
        p1 = self.args[1] * (t / self.args[5])**self.args[7]

        normcorr = (((self.args[4] / p1)**-self.args[2] \
                 +   (self.args[4] / p1)**-self.args[3]))

        # This is to conserve memory.
        xx = x / p1
        y  = np.power(xx, -self.args[2])
        y += np.power(xx, -self.args[3])
        y = np.power(y, -1.)#np.divide(1., y, out=y)

        if self.t == "1+z":
            y *= normcorr * self.args[0] \
                * ((1. + kwargs["z"]) / self.args[5])**self.args[6]
        else:
            y *= normcorr * self.args[0] \
                * (kwargs[self.t] / self.args[5])**self.args[6]

        # Need to add evolution for S(M) parameters here.
        piv = self.args[8] * (t / self.args[5])**-self.args[11]
        gam3 = self.args[9] * (t / self.args[5])**-self.args[12]
        gam4 = self.args[10] * (t / self.args[5])**-self.args[13]

        y *= (1. + (x / piv)**gam3)**gam4

        return y

class DoublePowerLawEvolvingNorm(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]

        # Normalization evolves
        normcorr = (((self.args[4] / self.args[1])**-self.args[2] \
                 +   (self.args[4] / self.args[1])**-self.args[3]))

        # This is to conserve memory.
        y  = np.power(x / self.args[1], -self.args[2])
        y += np.power(x / self.args[1], -self.args[3])

        y = np.power(y, -1.)
        #np.divide(1., y, out=y)

        if self.t == "1+z":
            y *= normcorr * self.args[0] \
                * ((1. + kwargs["z"]) / self.args[5])**self.args[6]
        else:
            y *= normcorr * self.args[0] \
                * (kwargs[self.t] / self.args[5])**self.args[6]

        return y

class DoublePowerLawEvolvingPeak(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]

        if self.t == "1+z":
            t = 1. + kwargs["z"]
        else:
            t = kwargs[self.t]

        # This is the peak mass
        p1 = self.args[1] * (t / self.args[5])**self.args[6]

        # Normalization evolves
        normcorr = (((self.args[4] / p1)**-self.args[2] \
                 +   (self.args[4] / p1)**-self.args[3]))

        # This is to conserve memory.
        y  = (x / p1)**-self.args[2]
        y += (x / p1)**-self.args[3]
        np.divide(1., y, out=y)

        y *= normcorr * self.args[0]
        return y


class DoublePowerLawEvolvingNormPeak(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]

        if self.t == "1+z":
            t = 1. + kwargs["z"]
        else:
            t = kwargs[self.t]

        # This is the peak mass
        p1 = self.args[1] * (t / self.args[5])**self.args[7]

        normcorr = (((self.args[4] / p1)**-self.args[2] \
                 +   (self.args[4] / p1)**-self.args[3]))

        # This is to conserve memory.
        xx = x / p1
        y  = np.power(xx, -self.args[2])
        y += np.power(xx, -self.args[3])
        y = np.power(y, -1.)#np.divide(1., y, out=y)

        if self.t == "1+z":
            y *= normcorr * self.args[0] \
                * ((1. + kwargs["z"]) / self.args[5])**self.args[6]
        else:
            y *= normcorr * self.args[0] \
                * (kwargs[self.t] / self.args[5])**self.args[6]

        return y

class DoublePowerLawEvolvingNormPeakSlope(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]

        #if self.t == "1+z":
        #    t = 1. + kwargs["z"]
        #else:
        #    t = kwargs[self.t]

        t = self.get_time_var(**kwargs)

        # This is the peak mass
        p1 = self.args[1] * (t / self.args[5])**self.args[7]

        normcorr = (((self.args[4] / p1)**-self.args[2] \
                 +   (self.args[4] / p1)**-self.args[3]))

        s1 = self.args[2] * (t / self.args[5])**self.args[8]
        s2 = self.args[3] * (t / self.args[5])**self.args[9]

        # This is to conserve memory.
        xx = x / p1
        y  = xx**-s1
        y += xx**-s2
        np.divide(1., y, out=y)

        if self.t == 'a':
            raise NotImplemented('help')
        else:
            y *= normcorr * self.args[0] \
                * (t / self.args[5])**self.args[6]


        return y

class DoublePowerLawEvolvingNormPeakSlopeFlex(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]

        if self.t == "1+z":
            t = 1. + kwargs["z"]
        else:
            t = kwargs[self.t]

        # This is the peak mass
        p1 = 10**(np.log10(self.args[1]) + self.args[7] * (t / self.args[5]) \
           + self.args[11] * (t / self.args[5])**2)

        normcorr = (((self.args[4] / p1)**-self.args[2] \
                 +   (self.args[4] / p1)**-self.args[3]))

        s1 = self.args[2] + self.args[8] * (t / self.args[5]) \
           + + self.args[12] * (t / self.args[5])**2
        s2 = self.args[3] + self.args[9] * (t / self.args[5]) \
           + + self.args[13] * (t / self.args[5])**2


        # This is to conserve memory.
        xx = x / p1
        y  = xx**-s1
        y += xx**-s2
        np.divide(1., y, out=y)

        if self.t == "1+z":
            y *= 10**(np.log10(normcorr * self.args[0]) \
               + self.args[6] * ((1. + kwargs["z"]) / self.args[5]) \
               + self.args[10] * ((1. + kwargs["z"]) / self.args[5])**2)
        else:
            raise NotImplemented('help')
            y *= normcorr * self.args[0] \
               + self.args[6] * (kwargs[self.t] / self.args[5]) \
               + self.args[10] * (kwargs[self.t] / self.args[5])**2

        return y

class DoublePowerLawEvolvingNormPeakSlopeFloor(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]

        if self.t == "1+z":
            t = 1. + kwargs["z"]
        else:
            t = kwargs[self.t]

        # This is the peak mass
        p1 = self.args[1] * (t / self.args[5])**self.args[7]

        normcorr = (((self.args[4] / p1)**-self.args[2] \
                 +   (self.args[4] / p1)**-self.args[3]))

        s1 = self.args[2] * (t / self.args[5])**self.args[8]
        s2 = self.args[3] * (t / self.args[5])**self.args[9]

        # This is to conserve memory.
        xx = x / p1
        y  = xx**-s1
        y += xx**-s2
        np.divide(1., y, out=y)

        if self.t == "1+z":
            y *= normcorr * self.args[0] \
                * (t / self.args[5])**self.args[6]
        else:
            y *= normcorr * self.args[0] \
                * (t / self.args[5])**self.args[6]


        floor = self.args[10] * (t / self.args[5])**self.args[11]

        return np.maximum(y, floor)

class DoublePowerLawEvolvingAsB13(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]

        # Need scale factor
        a = 1. / (1. + kwargs['z'])

        # Basic idea here is to have parameters that dictate
        # low-z, medium-z, and high-z behaviour, e.g.,
        # log10(f_star,10) = p[0] + p[5] * (1 - a) \
        #                  + p[9] * np.log(1 + z) + p[13] * z

        logp0 = np.log10(self.args[0]) + self.args[5] * (1 - a) \
              + self.args[9] * np.log(1 + kwargs['z']) \
              + self.args[13] * kwargs['z']
        p0 = 10**logp0

        logp1 = np.log10(self.args[1]) + self.args[6] * (1 - a) \
              + self.args[10] * np.log(1 + kwargs['z']) \
              + self.args[14] * kwargs['z']

        p1 = 10**logp1

        normcorr = (((self.args[4] / p1)**-self.args[2] \
                 +   (self.args[4] / p1)**-self.args[3]))

        s1 = self.args[2] + self.args[7] * (1 - a) \
              + self.args[11] * np.log(1 + kwargs['z']) \
              + self.args[15] * kwargs['z']

        s2 = self.args[3] + self.args[8] * (1 - a) \
              + self.args[12] * np.log(1 + kwargs['z']) \
              + self.args[16] * kwargs['z']

        # This is to conserve memory.
        xx = x / p1
        y  = xx**-s1
        y += xx**-s2
        np.divide(1., y, out=y)

        y *= normcorr * p0

        return y

class Okamoto(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]

        y = (1. + (2.**(self.args[0] / 3.) - 1.) \
          * (x / self.args[1])**-self.args[0])**(-3. / self.args[0])

        return y

class OkamotoEvolving(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]

        if self.t == "1+z":
            t = 1. + kwargs["z"]
        else:
            t = kwargs[self.t]

        p0 = self.args[0] * (t / self.args[2])**self.args[3]
        p1 = self.args[1] * (t / self.args[2])**self.args[4]

        y = (1. + (2.**(p0 / 3.) - 1.) * (x / p1)**-p0)**(-3. / p0)

        return y

class Schechter(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]

        if self.x.lower() in ["mags", "muv", "mag"]:
            y = 0.4 * np.log(10.) * 10**self.args[0] \
                * (10**(0.4 * (self.args[1] - x)))**(self.args[2] + 1.) \
                * np.exp(-10**(0.4 * (self.args[1] - x)))
        else:
            y = 10**self.args[0] * (x / self.args[1])**self.args[2] \
              * np.exp(-(x / self.args[1])) / self.args[1]

        return y

class SchechterEvolving(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]

        if self.t == "1+z":
            t = 1. + kwargs["z"]
        else:
            t = kwargs[self.t]

        p0 = 10**(self.args[0] + self.args[4] * (t - self.args[3]))
        p1 = self.args[1] + self.args[5] * (t - self.args[3])
        p2 = self.args[2] + self.args[6] * (t - self.args[3])

        if self.x.lower() in ["mags", "muv", "mag"]:
            y = 0.4 * np.log(10.) * p0 \
                * (10**(0.4 * (p1 - x)))**(p2 + 1.) \
                * np.exp(-10**(0.4 * (p1 - x)))
        else:
            y = p0 * (x / p1)**p2 * np.exp(-(x / p1)) / p1

        return y

class Linear(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]
        y = self.args[0] + self.args[2] * (x - self.args[1])
        return y

class LinearEvolvingNorm(BasePQ):
    def __call__(self, **kwargs):
        if self.x == "1+z":
            x = 1. + kwargs["z"]
        else:
            x = kwargs[self.x]

        if self.t == "1+z":
            t = 1. + kwargs["z"]
        else:
            t = kwargs[self.t]

        p0 = self.args[0] + self.args[4] * (t - self.args[3])

        x = kwargs[self.x]
        y = p0 + self.args[2] * (x - self.args[1])
        return y

class LogLinear(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]
        logy = self.args[0] + self.args[2] * (x - self.args[1])
        y = 10**logy
        return y

class LinLog(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]
        y = self.args[0] + self.args[2] * (np.log10(x) - self.args[1])
        return y

class LinLogEvolvingNorm(BasePQ):
    def __call__(self, **kwargs):
        if self.x == "1+z":
            x = 1. + kwargs["z"]
        else:
            x = kwargs[self.x]

        if self.t == "1+z":
            t = 1. + kwargs["z"]
        else:
            t = kwargs[self.t]

        p0 = self.args[0] * (t / self.args[3])**self.args[4]

        y = p0 + self.args[2] * (np.log10(x) - self.args[1])
        return y

class LogLinearEvolvingNorm(BasePQ):
    def __call__(self, **kwargs):
        if self.x == "1+z":
            x = 1. + kwargs["z"]
        else:
            x = kwargs[self.x]

        if self.t == "1+z":
            t = 1. + kwargs["z"]
        else:
            t = kwargs[self.t]

        p0 = self.args[0] * (t / self.args[3])**self.args[4]

        x = kwargs[self.x]
        logy = self.args[0] + self.args[2] * (x - self.args[1])
        y = 10**logy
        return y




class PointsLinear(BasePQ):
    def __call__(self, **kwargs):
        x = kwargs[self.x]
        m = (self.args[3] - self.args[2]) / (self.args[1] - self.args[0])
        y = m*(np.log10(x) - self.args[1]) + self.args[3]

        return y

class ParameterizedQuantity(object):
    def __init__(self, **kwargs):
        if kwargs["pq_func"] == "pl":
            self.func = PowerLaw(**kwargs)
        elif kwargs["pq_func"] == "pl_10":
            self.func = PowerLaw10(**kwargs)
        elif kwargs["pq_func"] == "pl_evolN":
            self.func = PowerLawEvolvingNorm(**kwargs)
        elif kwargs["pq_func"] == "pl_evolS":
            self.func = PowerLawEvolvingSlope(**kwargs)
        elif kwargs["pq_func"] == "pl_evolNS":
            self.func = PowerLawEvolvingNormSlope(**kwargs)    
        elif kwargs["pq_func"] == "pl_evolS2":
            self.func = PowerLawEvolvingSlopeWithGradient(**kwargs)
        elif kwargs["pq_func"] == "erf":
            self.func = Erf(**kwargs)
        elif kwargs["pq_func"] == "erf_evolB13":
            self.func = ErfEvolvingAsB13(**kwargs)
        elif kwargs["pq_func"] in ["dpl", "dpl_arbnorm"]:
            self.func = DoublePowerLaw(**kwargs)
        elif kwargs["pq_func"] == "dplx":
            self.func = DoublePowerLawExtended(**kwargs)
        elif kwargs["pq_func"] == "dplx_evolN":
            self.func = DoublePowerLawExtendedEvolvingNorm(**kwargs)
        elif kwargs["pq_func"] == "dplx_evolNPX":
            self.func = DoublePowerLawExtendedEvolvingNormPeakX(**kwargs)
        elif kwargs["pq_func"] in ["dpl_normP"]:
            self.func = DoublePowerLawPeakNorm(**kwargs)
        elif kwargs["pq_func"] == "dpl_evolN":
            self.func = DoublePowerLawEvolvingNorm(**kwargs)
        elif kwargs["pq_func"] == "dpl_evolP":
            self.func = DoublePowerLawEvolvingPeak(**kwargs)
        elif kwargs["pq_func"] == "dpl_evolNP":
            self.func = DoublePowerLawEvolvingNormPeak(**kwargs)
        elif kwargs["pq_func"] == "dpl_evolNPS":
            self.func = DoublePowerLawEvolvingNormPeakSlope(**kwargs)
        elif kwargs["pq_func"] == "dpl_evolNPSflex":
            self.func = DoublePowerLawEvolvingNormPeakSlopeFlex(**kwargs)
        elif kwargs["pq_func"] == "dpl_evolNPSF":
            self.func = DoublePowerLawEvolvingNormPeakSlopeFloor(**kwargs)
        elif kwargs["pq_func"] == "dpl_evolB13":
            self.func = DoublePowerLawEvolvingAsB13(**kwargs)
        elif kwargs["pq_func"] == "exp":
            self.func = Exponential(**kwargs)
        elif kwargs["pq_func"] in ["normal", "gaussian"]:
            self.func = Normal(**kwargs)
        elif kwargs["pq_func"] == "lognormal":
            self.func = LogNormal(**kwargs)
        elif kwargs["pq_func"] == "exp-":
            self.func = ExponentialInverse(**kwargs)
        elif kwargs['pq_func'] == 'exp-comp':
            self.func = ExponentialInverseComplement(**kwargs)
        elif kwargs['pq_func'] == 'exp-comp_evolT':
            self.func = ExponentialInverseComplementEvolvingTurnover(**kwargs)
        elif kwargs["pq_func"] == "pwpl":
            self.func = PiecewisePowerLaw(**kwargs)
        elif kwargs["pq_func"] == "ramp":
            self.func = Ramp(**kwargs)
        elif kwargs["pq_func"] == "logramp":
            self.func = LogRamp(**kwargs)
        elif kwargs["pq_func"] == "tanh_abs":
            self.func = TanhAbs(**kwargs)
        elif kwargs["pq_func"] == "tanh_rel":
            self.func = TanhRel(**kwargs)
        elif kwargs["pq_func"] == "logtanh_abs":
            self.func = LogTanhAbs(**kwargs)
        elif kwargs["pq_func"] == "logtanh_abs_evolM":
            self.func = LogTanhAbsEvolvingMidpoint(**kwargs)
        elif kwargs['pq_func'] == 'logtanh_abs_evolMFC':
            self.func = LogTanhAbsEvolvingMidpointFloorCeiling(**kwargs)
        elif kwargs['pq_func'] == 'logtanh_abs_evolMFCW':
            self.func = LogTanhAbsEvolvingMidpointFloorCeilingWidth(**kwargs)
        elif kwargs['pq_func'] == 'logtanh_abs_evolMFCWflex':
            self.func = LogTanhAbsEvolvingMidpointFloorCeilingWidthFlex(**kwargs)
        elif kwargs['pq_func'] == 'logtanh_abs_evolW':
            self.func = LogTanhAbsEvolvingWidth(**kwargs)
        elif kwargs["pq_func"] == "logtanh_rel":
            self.func = LogTanhRel(**kwargs)
        elif kwargs["pq_func"] == 'logsigmoid_abs_evol_FCW':
            self.func = LogSigmoidEvolvingFloorCeilingWidth(**kwargs)
        elif kwargs["pq_func"] == "step_abs":
            self.func = StepAbs(**kwargs)
        elif kwargs["pq_func"] == "step_rel":
            self.func = StepRel(**kwargs)
        elif kwargs["pq_func"] == "okamoto":
            self.func = Okamoto(**kwargs)
        elif kwargs["pq_func"] == "okamoto_evol":
            self.func = OkamotoEvolving(**kwargs)
        elif kwargs["pq_func"] in ["schechter", "plexp"]:
            self.func = Schechter(**kwargs)
        elif kwargs["pq_func"] in ["schechter_evol"]:
            self.func = SchechterEvolving(**kwargs)
        elif kwargs["pq_func"] in ["linear"]:
            self.func = Linear(**kwargs)
        elif kwargs["pq_func"] in ["linear_evolN"]:
            self.func = LinearEvolvingNorm(**kwargs)
        elif kwargs["pq_func"] in ["loglin"]:
            self.func = LogLinear(**kwargs)
        elif kwargs["pq_func"] in ["linlog"]:
            self.func = LinLog(**kwargs)
        elif kwargs["pq_func"] in ["linlog_evolN"]:
            self.func = LinLogEvolvingNorm(**kwargs)
        elif kwargs["pq_func"] in ["loglin_evolN"]:
            raise NotImplemented('help')
        elif kwargs["pq_func"] in ["p_linear"]:
            self.func = PointsLinear(**kwargs)
        else:
            raise NotImplemented("help")

    def __call__(self, **kwargs):

        # Patch up kwargs. Make sure inputs are arrays and that they lie
        # within the specified range (if there is one).
        kw = {}
        for key in kwargs:
            var = np.atleast_1d(kwargs[key])

            if key != self.func.x:
                kw[key] = var
                continue

            # Should have these options for var2 also
            if self.func.var_ceil is not None:
                if type(self.func.var_ceil) in numeric_types:
                    var = np.minimum(var, self.func.var_ceil)
            if self.func.var_floor is not None:
                if type(self.func.var_floor) in numeric_types:
                    var = np.maximum(var, self.func.var_floor)

            kw[key] = var

        y = self.func.__call__(**kw)

        if self.func.val_ceil is not None:
            if type(self.func.val_ceil) in numeric_types:
                y = np.minimum(y, self.func.val_ceil)
        if self.func.val_floor is not None:
            if type(self.func.val_floor) in numeric_types:
                y = np.maximum(y, self.func.val_floor)

        return y


def get_function_from_par(par, pf):
    """
    Returns a function representation of input parameter `par`.

    For example, the user supplies the parameter `pop_dust_yield`. This
    routien figures out if that's a number, a function, or a string
    indicating a ParameterizedQuantity, and creates a callable function
    no matter what.
    """

    t = type(pf[par])

    if t in numeric_types:
        func = lambda **kwargs: pf[par]
    elif t == FunctionType:
        func = lambda **kwargs: pf[par](**kwargs)
    elif isinstance(pf[par], str) and pf[par].startswith('pq'):
        pars = get_pq_pars(pf[par], pf)
        ob = ParameterizedQuantity(**pars)
        func = lambda **kwargs: ob.__call__(**kwargs)
    else:
        raise NotImplementedError(f"Unrecognized option for `{par}`.")

    if f'{par}_inv' in pf:
        if pf[f'{par}_inv']:
            func = lambda **kwargs: 1. - func(**kwargs)

    return func
