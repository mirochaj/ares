"""
williams2018.py

Williams et al. 2018, ApJS, 236, 33W

https://arxiv.org/abs/1802.05272
https://ui.adsabs.harvard.edu/abs/2018ApJS..236...33W/abstract

"""

import numpy as np

magbins = np.arange(-22.75, -16.75, 0.5)

data = \
{
 0.5: {'M': magbins,
       'phi': np.array([-14.59, -10.42, -7.74, -6.02, -4.88, -4.13, -3.61,
            -3.25, -2.98, -2.76, -2.59, -2.44]),
       'err': None,
      },
 0.8: {'M': magbins[0:-1], # final magbin has 0 error, excised here by hand
       'phi': np.array([-12.02, -8.96, -6.98, -5.68, -4.78, -4.11, -3.58, -3.17,
            -2.86, -2.63, -2.45]),
       'err': np.array([1.34, 1.1, 0.85, 0.62, 0.43, 0.27, 0.16, 0.08, 0.03,
            0.01, 0.01]),
      },
 1.25: {'M': magbins,
       'phi': np.array([-10.38, -7.91, -6.27, -5.15, -4.37, -3.78, -3.34,
            -3.0, -2.75, -2.54, -2.37, -2.23]),
       'err': np.array([0.88, 0.69, 0.50, 0.34, 0.21, 0.11, 0.04, 0.03, 0.05,
            0.06, 0.07, 0.09]),
      },
 1.75: {'M': magbins,
       'phi': np.array([-9.06, -6.92, -5.35, -4.26, -3.54, -3.07, -2.76,
            -2.54, -2.39, -2.27, -2.16, -2.06]),
       'err': np.array([0.21, 0.17, 0.15, 0.15, 0.16, 0.17, 0.17, 0.16,
            0.14, 0.12, 0.10, 0.08]),
      },
 2.25: {'M': magbins,
       'phi': np.array([-6.55, -5.34, -4.49, -3.88, -3.43, -3.08, -2.8, -2.59,
            -2.41, -2.27, -2.14, -2.03]),
       'err': np.array([0.70, 0.53, 0.38, 0.27, 0.18, 0.12, 0.09, 0.10, 0.12,
            0.15, 0.17, 0.21]),
      },
 2.75: {'M': magbins,
       'phi': np.array([-6.27, -5.15, -4.35, -3.77, -3.33, -3.01, -2.76, -2.56,
            -2.39, -2.23, -2.09, -1.95]),
       'err': np.array([0.30, 0.24, 0.19, 0.17, 0.17, 0.17, 0.17, 0.17, 0.17,
            0.17, 0.18, 0.20]),
      },
 3.75: {'M': magbins,
       'phi': np.array([-6.44, -5.16, -4.28, -3.66, -3.23, -2.92, -2.69,  -2.51,
            -2.36, -2.22, -2.10, -1.99]),
       'err': np.array([0.17, 0.11, 0.07, 0.05, 0.05, 0.05, 0.05, 0.06, 0.07,
            0.08, 0.10, 0.12]),
      },


}
units = {'lf': 'log10'}

def get_Reff(z, Ms, quiescent=False, cosm=None):
    """
    Return effective (half-light) radius [kpc].

    .. note :: This is Equations 23-25 in Williams et al. (2018).

    Parameters
    ----------
    z : int, float
        Redshift of interest
    Ms : int, float, np.ndarray
        Stellar mass(es) in Msun.
    quiescent : bool
        If True, uses different function specific to quiescent galaxies.

    Returns
    -------
    Half-light radius in kpc.

    """

    if quiescent:
        B_H = 3.8e-4 * np.exp(np.log10(Ms)*0.71) - 0.11

        if Ms >= 10**9.75:
            Beta_H = 1.38e12 * np.exp(-2.87 * np.log10(Ms)) - 1.21
        else:
            Beta_H = -0.19
    else:
        B_H = 0.23 * np.log10(Ms) - 1.61
        Beta_H = -0.08 * np.log10(Ms) + 0.25

    H = cosm.HubbleParameter(z)
    H0 = cosm.HubbleParameter(0)

    R = B_H * (H / H0)**Beta_H

    return 10**R
