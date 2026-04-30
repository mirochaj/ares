"""

InterpolationTables.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Jan 14 13:02:46 2013

Description:

"""

import re
import numpy as np
from ..util.Math import interp1d
from ..util.Math import LinearNDInterpolator
from scipy.interpolate import RectBivariateSpline

spline_err = "interp_tab == cubic and there are infs in our table!"
spline_err += "\nSpline interpolation is global, so any infs will render *all*"
spline_err += "\ninterpolated values NaN. Try linear interpolation, or narrow"
spline_err += "\nbounds of interpolation table."

err_msg = (lambda s: ("{0!s}\n\n# Table: {1!s}".format(spline_err, s)))

class LookupTable(object):
    def __init__(self, pf, name, logN, table, logx=None, t=None):
        """
        Initialize lookup table object for interpolation in N-D.

        Parameters
        ----------
        pf : dict

        """
        self.pf = pf
        self.name = name
        self.logN = np.array(logN)
        self.table = table

        if '_' in name:
            self.basename = name[0:name.find('_')]
        else:
            self.basename = name

        self.logNmin = np.min(self.logN, axis = 1)
        self.logNmax = np.max(self.logN, axis = 1)

        # Possible extra dimensions
        self.logx = logx
        self.t = t

        # Table properties
        self.shape = self.table.shape
        self.size = np.prod(self.shape)

        self.Nd = len(logN)
        self.evolving = False#int(np.any(pf['spectrum_evolving']))
        self.adv_secondary_ionization = int(pf['secondary_ionization'] > 1)
        self.Ed = self.adv_secondary_ionization + self.evolving

        self.D = self.Nd + self.Ed

        # Initialize
        self._init()

    def __call__(self, logN, logx=None, t=None):
        """
        Perform interpolation in self.D dimensions.

        Parameters
        ----------
        logN : np.ndarray
            Contains column densities at which to evaluate integral.
            Shape = (N_cells, N_absorbers)
        """

        # If we are beyond bounds of integral table, fix
        for i in range(self.Nd):
            logN[...,i][logN[...,i] < self.logNmin[i]] = self.logNmin[i]
            logN[...,i][logN[...,i] > self.logNmax[i]] = self.logNmax[i]

        if self.adv_secondary_ionization and logx is not None:
            logx[logx < self.logxmin] = self.logxmin
            logx[logx > self.logxmax] = self.logxmax

        # Compute result
        if self.D == 1:
            logresult = self.interp(logN[...,0])
        elif self.D == 2:
            ax2 = self._extra_axis(logx, t)
            logresult = np.zeros_like(logN[...,0])
            for i in range(logN.shape[0]):
                if self.adv_secondary_ionization and \
                    not (re.search('Wiggle', self.basename) or
                    re.search('Hat', self.basename)):
                    logresult[i] = self.interp(logN[i,0]).squeeze()
                else:
                    logresult[i] = self.interp(logN[i,0], ax2[i]).squeeze()
        else:
            logresult = \
                np.array([self.interp(tuple(Ncol)) for Ncol in logN])

        return logresult

    def _extra_axis(self, logx, t):
        """
        Non-column-density dimension in lookup table.

        Is it ionized fraction or time?
        """
        if logx is not None:
            return logx

        return t

    def _init(self):
        """ Create interpolation table. """

        if self.pf['interp_tab'] == 'cubic' and self.D <= 2 and \
            not np.all(np.isfinite(self.table)):
            raise ValueError(err_msg(self.basename))

        # Set up interpolation tables
        if self.D == 1:
            self.interp = \
                interp1d(self.logN[0], self.table,
                    kind=self.pf['interp_tab'])
        elif self.D == 2:

            if self.adv_secondary_ionization:
                ax2 = self.logx
                self.logxmin = np.min(self.logx)
                self.logxmax = np.max(self.logx)
            elif self.evolving:
                ax2 = self.t

            if self.basename in ['logPhi', 'logTau'] \
                and self.adv_secondary_ionization:
                self.interp = interp1d(self.logN[0], self.table)
            else:
                self.interp = \
                    RectBivariateSpline(self.logN[0], ax2, self.table)
        else:
            if self.Ed:
                raise NotImplemented('Haven\'t implemented time and secondary ionization option yet.')

            self.interp = LinearNDInterpolator(self.logN, self.table)
