"""

TurningPoints.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Apr 28 10:14:06 MDT 2014

Description:

"""

import numpy as np
from ..util import ParameterFile
from scipy.misc import derivative
from scipy.optimize import minimize
from ..physics.Constants import nu_0_mhz
from ..util.Math import central_difference
from scipy.interpolate import splrep, splev
from ..util.SetDefaultParameterValues import SetAllDefaults

class TurningPoints(object):
    def __init__(self, inline=False, **kwargs):
        self.pf = ParameterFile(**kwargs)

        self.delay = self.pf['delay_extrema']

        # Keep track of turning points we've passed
        self.TPs = []
        #if self.pf['initial_redshift'] < 70:
        #    self.TPs.append('A')

        self.turning_points = {}
        self.found_TP, self.z_TP = False, -99999
        self.Npts = 0

    @property
    def step(self):
        if not hasattr(self, '_step'):
            self._step = 0

        return self._step

    def which_extremum(self, z, dTb, zp, dTbdz):
        # Compute second derivative
        zpp, dTb2dz2 = central_difference(zp, dTbdz)

        # Must strictly all elements be of the same sign and concavity?
        # Leads to misidentification of turning points on rare occasions
        # when a single point has the wrong concavity.
        #negative = np.all(dTb < 0)
        negative = len(dTb[dTb < 0]) > len(dTb[dTb > 0])
        positive = not negative
        #concave_up = np.all(dTb2dz2 > 0)
        concave_up = len(dTb2dz2[dTb2dz2 > 0]) > len(dTb2dz2[dTb2dz2 < 0])
        concave_down = not concave_up

        # Based on sign of brightness temperature and concavity,
        # determine which turning point we've found.
        #print max(z), negative, concave_up, self.TPs.keys()
        if negative and concave_up and (max(z) > 60 and 'B' not in self.TPs):
            return 'A'
        elif negative and concave_down:
            # If WF coupling happens early enough there will not be a
            # turning point A or B, and thus we're probably looking at the
            # end of EoR "turning point" if C has already been found.
            if 'B' in self.TPs:
                return 'Bp'
            elif ('B' not in self.TPs):
                return 'B'
            elif 'C' in self.TPs:
                # Preferably this is zero within tolerances?
                assert dTb < 1e-2
                return 'E'

        elif negative and concave_up \
            and (('A' in self.TPs) or (max(z) < 60) or ('B' in self.TPs)):
            if ('C' not in self.TPs):
                return 'C'
            else:
                return 'Cp'
        elif positive and concave_down:
            if ('D' not in self.TPs):
                return 'D'
            else:
                return 'Dp'
        else:
            return 'unknown'

    def is_stopping_point(self, z, dTb):
        """
        Determine if we have just passed a turning point.

        Parameters
        ----------
        z : np.ndarray
            Redshift (descending order).
        dTb : np.ndarray
            Corresponding brightness temperatures.

        """

        # Hack: don't check for turning points right at beginning
        # Also: need at least 3 points to check for a turning point
        if self.step < 3 or (z[-1] > 1e3):
            self._step += 1
            return False

        # Sometimes get discontinuities when RT gets flipped on.
        # This is kludgey...sorry.
        if min(z) > (self.pf['initial_redshift'] - self.pf['delay_tracking']):
            return False

        self._step += 1

        if not self.found_TP:

            # Grab last three redshift and temperature points
            dz3pt = np.diff(z[-1:-4:-1])
            dT3pt = np.diff(dTb[-1:-4:-1])

            # If changes in temperatures have different signs, we found
            # a turning point. Come back in self.delay steps to interpolate
            # and determine its position more accurately
            if len(np.unique(np.sign(dT3pt))) == 1:
                pass
            else:
                self.found_TP = True

                # Set a redshift just below current point.
                # This is when we'll come back to precisely determine where
                # the extremum is (when we have enough points to interpolate)
                self.z_TP = z[-2]

            # Check for zero-crossing too
            if not np.all(np.sign(dTb[-1:-3:-1]) < 0) and \
                'ZC' not in self.turning_points:
                zTP = np.interp(0.0, dTb[-1:-3:-1], z[-1:-3:-1])
                nu = nu_0_mhz / (1. + np.array(z))
                nn, dTbdnu = central_difference(np.array(nu[-1:-10:-1]),
                    np.array(dTb[-1:-10:-1]))

                slope = np.interp(nu_0_mhz / (1. + zTP), nn, dTbdnu)

                self.turning_points['ZC'] = (zTP, slope, None)

                if self.pf['stop'] == 'ZC':
                    return True

        # If we found a turning point, hone in on its position
        # (z[-1] < self.z_TP) and
        if self.found_TP and (self.Npts > 5):

            # Redshift and temperature points bracketing crudely-determined
            # extremum position
            k = -2 * self.delay - 1
            zbracketed = np.array(z[k:-1])
            Tbracketed = np.array(dTb[k:-1])

            # Interpolate derivative to find extremum more precisely
            zz, dT = central_difference(zbracketed, Tbracketed)
            # In order of increasing redshift

            TP = self.which_extremum(zbracketed, Tbracketed, zz, dT)
            self.TPs.append(TP)

            # Crude guess at turning pt. position
            zTP_guess = zz[np.argmin(np.abs(dT))]
            TTP_guess = np.interp(zTP_guess, z[k:-1], dTb[k:-1])

            #zTP_new_guess, TTP_new_guess = \
            #    self.guess_from_signal(TP, z[k:-1], dTb[k:-1])

            # Spline interpolation to get "final" extremum redshift
            for ll in [3, 2, 1]:

                if ll > 1:
                    Bspl_fit1 = splrep(z[k:-1][-1::-1], dTb[k:-1][-1::-1], k=ll)

                    if ('B' in TP) or ('D' in TP):
                        dTb_fit = lambda zz: -splev(zz, Bspl_fit1)
                    else:
                        dTb_fit = lambda zz: splev(zz, Bspl_fit1)
                else:
                    dTb_fit = lambda zz: np.interp(zz, z[k:-1][-1::-1],
                        dTb[k:-1][-1::-1])

                    #print 'linear', z[k:-1][-1::-1], dTb[k:-1][-1::-1]

                zTP = float(minimize(dTb_fit, zTP_guess, tol=1e-4).x[0])
                TTP = float(dTb_fit(zTP))

                if ('B' in TP) or ('D' in TP):
                    TTP *= -1.

                # Contingencies....
                if self.is_crazy(TP, zTP, TTP):

                    if ll == 1:
                        self.turning_points[TP] = (-np.inf, -np.inf, -np.inf)
                    else:
                        continue

                else:
                    # Compute curvature at turning point (mK**2 / MHz**2)
                    nuTP = nu_0_mhz / (1. + zTP)
                    d2 = float(derivative(lambda zz: splev(zz, Bspl_fit1),
                        x0=float(zTP), n=2, dx=1e-4, order=5) * nu_0_mhz**2 / nuTP**4)

                    self.turning_points[TP] = (zTP, TTP, d2)

                    break

            # Reset null state
            self.found_TP = False
            self.z_TP = -99999
            self.Npts = 0

            if self.pf['stop'] in self.TPs:
                return True

        elif self.found_TP:
            self.Npts += 1

        return False

    #def guess_from_derivative(self, tp, zarr, Tarr, dTarr):
    #    return zz[np.argmin(np.abs(dTarr))], \
    #        np.interp(zTP_guess, z[k:-1], dTb[k:-1])

    def is_crazy(self, tp, z, T):
        # Check that redshift is within bounds of simulation
        if (z < self.pf['final_redshift']) or (z > self.pf['initial_redshift']):
            return True

        return False

        if tp == 'B':
            if T < -50:
                return True
            else:
                return False
        elif tp == 'C':
            if T < -500:
                return True
            else:
                return False
        elif tp == 'D':
            if T > 50:
                return True
            else:
                return False

        return False

    def guess_from_signal(self, tp, zarr, Tarr):
        """
        If turning point is unphysical, something went wrong.
        Fall back to simpler estimate.
        """

        if tp == 'B':
            TTP_guess = np.min(np.abs(Tarr))
            zTP_guess = np.interp(TTP_guess, Tarr[-1::-1], zarr[-1::-1])
        elif tp == 'C':
            TTP_guess = np.min(Tarr)
            zTP_guess = np.interp(TTP_guess, Tarr[-1::-1], zarr[-1::-1])
        elif tp == 'D':
            TTP_guess = np.max(Tarr)
            zTP_guess = np.interp(TTP_guess, Tarr, zarr)
        else:
            return None, None

        return zTP_guess, TTP_guess



        # Check curvature
