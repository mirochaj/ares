"""
Kroupa P., 2001, MNRAS, 322, 231
"""

import numpy as np

m_0 = (0.01, 0.08)
m_1 = (0.08, 0.5)
m_2 = (0.5, 1.0)
m_3 = (1.0, np.inf)

imf_pars = \
{
 'alpha_0': 0.3,
 'alpha_1': 1.3,
 'alpha_2': 2.3,
 'alpha_3': 3.3,  
}

imf_err = \
{
 'alpha_0': 0.7,
 'alpha_1': 0.5,
 'alpha_2': 0.3,
 'alpha_3': 0.7,
}

class InitialMassFunction(object):
    def __init__(self, **kwargs):
        
        if not kwargs:
            kwargs = imf_pars
        else:
            kw = imf_pars.copy()
            kw.update(kwargs)
            kwargs = kw

        self.kwargs = kwargs

        self._norm()

    def __call__(self, m):

        if m_0[0] <= m < m_0[1]:
            return self._n0 * m**-self.kwargs['alpha_0']
        elif m_1[0] <= m < m_1[1]:
            return self._n1 * m**-self.kwargs['alpha_1']
        elif m_2[0] <= m < m_2[1]:
            return self._n2 * m**-self.kwargs['alpha_2']
        elif m_3[0] <= m < m_3[1]:
            return self._n3 * m**-self.kwargs['alpha_3']

    def _norm(self):
        self._n0 = 1.0
        self._n1 = self._n0 * m_0[1]**-self.kwargs['alpha_0'] \
            / m_1[0]**-self.kwargs['alpha_1']
        self._n2 = self._n1 * m_1[1]**-self.kwargs['alpha_1'] \
            / m_2[0]**-self.kwargs['alpha_2'] 
        self._n3 = self._n2 * m_2[1]**-self.kwargs['alpha_2'] \
            / m_3[0]**-self.kwargs['alpha_3']
        

    
    
    