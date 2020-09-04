"""

test_util_stats.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Tue 24 Mar 2020 22:11:31 EDT

Description: 

"""

import ares
import numpy as np
from ares.util.Stats import GaussND
from scipy.interpolate import interp1d

def test():
    
    # First, test my dumb wrapper around interp1d
    x = np.linspace(0, 4 * np.pi, 20)
    y = np.sin(x)
    
    func1 = interp1d(x, y, kind='cubic')
    func2 = ares.util.Math.interp1d_wrapper(x, y, kind='cubic')
    
    x2 = np.linspace(0, 4 * np.pi, 50)
    
    f1 = func1(x2)
    f2 = func1(x2)
    
    assert np.array_equal(f1, f2)
    
    # Next, test LinaerNDInterpolator
    _x = _y = np.linspace(0, 5, 100)
    xx, yy = np.meshgrid(_x, _y)
    
    z = np.sin(xx) + np.cos(yy)
    
    func3 = ares.util.Math.LinearNDInterpolator([_x, _y], z)
    
    z0 = func3(np.array([0.5, 1.3]))

if __name__ == '__main__':
    test()
