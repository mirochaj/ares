"""

test_21cm_parameterized.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Aug  6 08:54:15 MDT 2014

Description: 21-cm signal in absence of astrophysical sources.

"""

import ares
import numpy as np

def test():

    # Create instance of Hydrogen class
    hydr = ares.physics.Hydrogen(approx_thermal_history='piecewise')

    # Analytic approximation to thermal history
    Tk = lambda z: hydr.cosm.Tgas(z)

    # Spin temperature (arguments: z, Tk, Ja, xHII, ne)
    Ts = lambda z: hydr.SpinTemperature(z, Tk(z), 0.0, 0.0, 0.0)

    # Brightness temperature (arguments: z, Ts, xavg optional)
    dTb = lambda z: hydr.get_21cm_dTb(z, Ts(z))

    # Define redshift interval of interest
    z = np.linspace(10, 1e3, 500)

    # Get CosmoRec recombination history
    CR = hydr.cosm.get_inits_rec()

    # Assume neutral medium for simplicity
    Ts_CR = hydr.SpinTemperature(CR['z'], CR['Tk'], 0.0, 0.0, 0.0)
    dTb_CR = hydr.get_21cm_dTb(CR['z'], Ts_CR)

if __name__ == '__main__':
    test()
