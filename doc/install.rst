Installation
============

ARES depends on:

* `numpy <http://numpy.scipy.org/>`_
* `scipy <http://www.scipy.org/>`_ 
* `h5py <http://www.h5py.org/>`_
* `matplotlib <http://matplotlib.sourceforge.net>`_ (for built-in analysis routines)

and optionally:

* `python-progressbar <https://code.google.com/p/python-progressbar/>`_
* `ndspace <https://bitbucket.org/mirochaj/ndspace>`_ (for creation/manipulation of N-D model grids)
* `hmf <https://github.com/steven-murray/hmf>`_ (halo mass function calculator written by Stephen Murray)

To download the code, do::

    hg clone https://bitbucket.org/mirochaj/ares ares
    cd ares
    python setup.py install
    
A few lookup tables will be downloaded to ares/input automatically.    


