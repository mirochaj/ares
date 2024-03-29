"""

schaerer2002.py

Schaerer (2002), A&A 382, 82

"""

import numpy as np

_col3 = ['M', 'log(L/Lsun)', 'logTeff', 'Q(H)', 'Q(He)', 'Q(He+)', 'Q(H2)',
         'Q(He)/Q(H)',  'Q(He+)/Q(H)']
_tab3 = \
[[1000., 7.444, 5.026, 1.607E+51, 1.137E+51, 2.829E+50, 1.727E+51, 0.708E+00, 0.176E+00],
[500.  , 7.106, 5.029, 7.380E+50, 5.223E+50, 1.299E+50, 7.933E+50, 0.708E+00, 0.176E+00],
[400.  , 6.984, 5.028, 5.573E+50, 3.944E+50, 9.808E+49, 5.990E+50, 0.708E+00, 0.176E+00],
[300.  , 6.819, 5.007, 4.029E+50, 2.717E+50, 5.740E+49, 4.373E+50, 0.674E+00, 0.142E+00],
[200.  , 6.574, 4.999, 2.292E+50, 1.546E+50, 3.265E+49, 2.487E+50, 0.674E+00, 0.142E+00],
[120.  , 6.243, 4.981, 1.069E+50, 7.213E+49, 1.524E+49, 1.161E+50, 0.674E+00, 0.142E+00],
[80.   , 5.947, 4.970, 5.938E+49, 3.737E+49, 3.826E+48, 6.565E+49, 0.629E+00, 0.644E-01],
[60.   , 5.715, 4.943, 3.481E+49, 2.190E+49, 2.243E+48, 3.848E+49, 0.629E+00, 0.644E-01],
[40.   , 5.420, 4.900, 1.873E+49, 1.093E+49, 1.442E+47, 2.123E+49, 0.584E+00, 0.770E-02],
[25.   , 4.890, 4.850, 5.446E+48, 2.966E+48, 5.063E+44, 6.419E+48, 0.545E+00, 0.930E-04],
[15.   , 4.324, 4.759, 1.398E+48, 6.878E+47, 2.037E+43, 1.760E+48, 0.492E+00, 0.146E-04],
[9.    , 3.709, 4.622, 1.794E+47, 4.303E+46, 1.301E+41, 3.785E+47, 0.240E+00, 0.725E-06],
[5.    , 2.870, 4.440, 1.097E+45, 8.629E+41, 7.605E+36, 3.760E+46, 0.787E-03, 0.693E-08]]

_col4 = ['M', 'lifetime', 'Q(H)', 'Q(He)', 'Q(He+)', 'Q(H2)',
         'Q(He)/Q(H)',  'Q(He+)/Q(H)']
_tab4 = \
[[1000., -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
[500.00, 1.899E+06, 6.802E+50, 3.858E+50, 5.793E+49, 7.811E+50, 0.567E+00, 0.852E-01],
[400.00 , 1.974E+06, 5.247E+50, 3.260E+50, 5.567E+49, 5.865E+50, 0.621E+00, 0.106E+00],
[300.00 , 2.047E+06, 3.754E+50, 2.372E+50, 4.190E+49, 4.182E+50, 0.632E+00, 0.112E+00],
[200.00 , 2.204E+06, 2.624E+50, 1.628E+50, 1.487E+49, 2.918E+50, 0.621E+00, 0.567E-01],
[120.00 , 2.521E+06, 1.391E+50, 7.772E+49, 5.009E+48, 1.608E+50, 0.559E+00, 0.360E-01],
[80.00  , 3.012E+06, 7.730E+49, 4.317E+49, 1.741E+48, 8.889E+49, 0.558E+00, 0.225E-01],
[60.00  , 3.464E+06, 4.795E+49, 2.617E+49, 5.136E+47, 5.570E+49, 0.546E+00, 0.107E-01],
[40.00  , 3.864E+06, 2.469E+49, 1.316E+49, 8.798E+46, 2.903E+49, 0.533E+00, 0.356E-02],
[25.00  , 6.459E+06, 7.583E+48, 3.779E+48, 3.643E+44, 9.387E+48, 0.498E+00, 0.480E-04],
[15.00  , 1.040E+07, 1.861E+48, 8.289E+47, 1.527E+43, 2.526E+48, 0.445E+00, 0.820E-05],
[9.00   , 2.022E+07, 2.807E+47, 7.662E+46, 3.550E+41, 5.576E+47, 0.273E+00, 0.126E-05],
[5.00   , 6.190E+07, 1.848E+45, 1.461E+42, 1.270E+37, 6.281E+46, 0.791E-03, 0.687E-08]]

# tab5 (mass loss included)
_tab5 = \
[[1000., 2.430E+06, 1.863E+51, 1.342E+51, 3.896E+50, 2.013E+51, 0.721E+00, 0.209E+00],
[500.  , 2.450E+06, 7.719E+50, 5.431E+50, 1.433E+50, 8.345E+50, 0.704E+00, 0.186E+00],
[300.  , 2.152E+06, 4.299E+50, 3.002E+50, 7.679E+49, 4.766E+50, 0.698E+00, 0.179E+00],
[220.  , 2.624E+06, 2.835E+50, 1.961E+50, 4.755E+49, 3.138E+50, 0.692E+00, 0.168E+00],
[200.  , 2.628E+06, 2.745E+50, 1.788E+50, 2.766E+49, 3.028E+50, 0.651E+00, 0.101E+00],
[150.  , 2.947E+06, 1.747E+50, 1.156E+50, 2.066E+49, 1.917E+50, 0.662E+00, 0.118E+00],
[100.  , 3.392E+06, 9.398E+49, 6.118E+49, 9.434E+48, 1.036E+50, 0.651E+00, 0.100E+00],
[80.   , 3.722E+06, 6.673E+49, 4.155E+49, 4.095E+48, 7.466E+49, 0.623E+00, 0.614E-01]]

tab3 = np.array(_tab3)
tab4 = np.array(_tab4)
tab5 = np.array(_tab5)

def _load(**kwargs):

    if kwargs['source_model'] == 'zams':
        tab = tab3
        kmax = 6
        raise ValueError('No \'lifetime\' here!')
    elif kwargs['source_model'] == 'tavg_nms':
        tab = tab4
        kmax = 5
    else:
        tab = tab5
        kmax = 5

    masses = tab[:,0]

    M = kwargs['source_mass']
    k = np.argmin(np.abs(M - masses))

    if abs(masses[k] - M) < 1e-3:
        y = [tab[k,kmax]]
        y.extend(tab[k,kmax-3:kmax])
        y = np.array(y)
    else:
        raise NotImplemented('must interpolate!')

    return y, 10**tab3[k,2], tab[k,1]
