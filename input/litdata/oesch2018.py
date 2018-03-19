"""
Oesch et al., 2017, arxiv

Table 4 and volume estimate from text.
"""

info = \
{
 'reference': 'Oesch et al., 2017, arXiv',
 'data': 'Table 4', 
}

redshifts = [10.]

wavelength = 1600. # I think?

ULIM = -1e10

data = {}
data['lf'] = \
{
 10.0: {'M': [-22.25, -21.25, -20.25, -19.25, -18.25,-17.25],
       'phi': [0.017e-4, 0.01e-4, 0.1e-4, 0.34e-4, 1.9e-4, 6.3e-4],
       'err': [ULIM, (0.022e-4, 0.008e-4), (0.1e-4, 0.05e-4),
            (0.45e-4, 0.22e-4), (2.5e-4, 1.2e-4), (14.9e-4, 5.2e-4)],
      },               
}

units = {'lf': 1.}



