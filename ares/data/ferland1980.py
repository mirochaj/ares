import os
import numpy as np

info = \
{
 'reference': 'Ferland 1980',
 'data': 'Table 1'
}

def _load():
    E = np.array([1.00, 0.25, 0.25, 0.11, 0.11, 0.0625, 0.0625, 0.04, 0.04,0.0278, 0.0278, 0.0204, 0.0204,0.0156, 0.0156, 0.0123, 0.0123,0.0100, 0.0100, 0.0083, 0.0083, 0.0069])
    T10 = np.array([2.11e-44, 2.48e-39, 1.37e-40, 1.15e-39, 4.26e-40, 9.04e-40, 5.93e-40, 8.51e-40, 6.90e-40, 8.50e-40, 7.56e-40, 8.66e-40, 8.06e-40, 8.87e-40, 8.47e-40, 9.11e-40, 8.82e-40, 9.34e-40, 9.14e-40, 9.58e-40, 9.42e-40, 9.80e-40])
    T20 = np.array([3.29e-42, 1.06e-39, 2.32e-40, 6.78e-40, 4.23e-40, 6.31e-40, 5.21e-40, 6.41e-40, 5.84e-40, 6.65e-40, 6.31e-40, 6.90e-40, 6.69e-40, 7.16e-40, 7.02e-40, 7.41e-40, 7.31e-40, 7.64e-40, 7.57e-40, 7.87e-40, 7.81e-40, 8.08e-40])

    return E, T10, T20
