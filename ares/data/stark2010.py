"""

stark2010.py

"""

import numpy as np

info = \
{
 'reference': 'Stark et al., 2010, MNRAS, 408, 1628',
 'data': 'Figure 13',  
 'label': 'Stark+ (2010)',
}

redshifts = np.array([7., 8.])
_mags1 = np.arange(-22, -18, 0.5)
_mags2 = np.arange(-22, -18, 1)


# Fig. 13a
x_13a = (0.0885, 0.0658, 0.0695, 0.1035, 0.1729, 0.3024, 0.4655, 0.6021)
e_13a = (0.1752-0.0885,0.0885), (0.1083-0.0658, 0.0658-0.0339), \
       (0.0944-0.0695, 0.0695-0.0448), (0.1372-0.1035,0.1035-0.0717), \
       (0.2153-0.1729, 0.1729-0.141), (0.382-0.3024,0.3024-0.2316), \
       (0.6265-0.4655, 0.4655-0.3096), (0.8037-0.6021,0.6021-0.3968)

# Fig 13b
# blue = 3.5 <= z <= 4.5
m_st_blue = np.arange(-22, -18, 1)
x_st_blue = (0.053, 0.063, 0.151, 0.4526)
e_st_blue = (0.096-0.053,0.053), (0.0856-0.063, 0.063-0.0391),\
            (0.1815-0.151,0.151-0.1218), (0.5615-0.4526, 0.4526-0.3477)

# red = 4.5 <= z <= 6.0
m_st_red = np.arange(-22, -18, 1)
x_st_red = (0.1968, 0.1016, 0.2174, 0.5549)
e_st_red = (0.3256-0.1968, 0.1968-0.0694), (0.1348-0.1016, 0.1016-0.0684),\
           (0.2638-0.2174,0.2174-0.1723), (0.7221-0.5549, 0.5549-0.389)


data_all = {'mags': _mags1, 'x_LAE': np.array(x_13a), 'err': np.array(e_13a)}

data_split = \
{
 'low': {'mags': _mags2, 'x_LAE': np.array(x_st_blue), 'err': np.array(e_st_blue)},
 'high': {'mags': _mags2, 'x_LAE': np.array(x_st_blue), 'err': np.array(e_st_blue)},
 'z': {'low': (3.5, 4.5), 'high': (4.5, 6.0)}
}









