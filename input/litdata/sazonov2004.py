"""
Sazonov, S. Yu., Ostriker, J. P., & Sunyaev, R. A. 2004, MNRAS, 347, 144
"""

import numpy as np

# Parameters for the Sazonov & Ostriker AGN template
_Alpha = 0.24
_Beta = 1.60
_Gamma = 1.06
_E_1 = 83e3
_K = 0.0041
_E_0 = (_Beta - _Alpha) * _E_1
_A = np.exp(2e3 / _E_1) * 2e3**_Alpha
_B = ((_E_0**(_Beta - _Alpha)) \
    * np.exp(-(_Beta - _Alpha))) / \
    (1.0 + (_K * _E_0**(_Beta - _Gamma)))
    
# Normalization constants to make the SOS04 spectrum continuous.
_SX_Normalization = 1.0
_UV_Normalization = _SX_Normalization * ((_A * 2e3**-_Alpha) * \
    np.exp(-2e3 / _E_1)) / ((1.2 * 2e3**-1.7) * np.exp(2000.0 / 2000.))
_IR_Normalization = _UV_Normalization * ((1.2 * 10**-1.7) \
    * np.exp(10.0 / 2e3)) / (1.2 * 159 * 10**-0.6)
_HX_Normalization = _SX_Normalization * (_A * _E_0**-_Alpha * \
    np.exp(-_E_0 / _E_1)) / (_A * _B * (1.0 + _K * _E_0**(_Beta - _Gamma)) * \
    _E_0**-_Beta)
    
def Spectrum(E, t=0.0, **kwargs):
    """
    Broadband quasar template spectrum.
    
    References
    ----------
    Sazonov, S., Ostriker, J.P., & Sunyaev, R.A. 2004, MNRAS, 347, 144.
    """
    
    op = (E < 10)
    uv = (E >= 10) & (E < 2e3) 
    xs = (E >= 2e3) & (E < _E_0)
    xh = (E >= _E_0) & (E < 4e5)
    
    if type(E) in [int, float]:
        if op:
            F = _IR_Normalization * 1.2 * 159 * E**-0.6
        elif uv:
            F = _UV_Normalization * 1.2 * E**-1.7 * np.exp(E / 2000.0)
        elif xs:
            F = _SX_Normalization * _A * E**-_Alpha * np.exp(-E / _E_1)
        elif xh:
            F = _HX_Normalization * _A * _B * (1.0 + _K * \
                E**(_Beta - _Gamma)) * E**-_Beta
        else: 
            F = 0
            
    else:        
        F = np.zeros_like(E)
        F += op * _IR_Normalization * 1.2 * 159 * E**-0.6
        F += uv * _UV_Normalization * 1.2 * E**-1.7 * np.exp(E / 2000.0)
        F += xs * _SX_Normalization * _A * E**-_Alpha * np.exp(-E / _E_1)
        F += xh * _HX_Normalization * _A * _B * (1.0 + _K * \
                E**(_Beta - _Gamma)) * E**-_Beta
    
    return E * F
        
