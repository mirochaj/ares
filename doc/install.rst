Installation
============

ARES depends on:

* `numpy <http://numpy.scipy.org/>`_
* `scipy <http://www.scipy.org/>`_ 
* `matplotlib <http://matplotlib.sourceforge.net>`_ (for built-in analysis routines)

and optionally:

* `python-progressbar <https://code.google.com/p/python-progressbar/>`_
* `hmf <https://github.com/steven-murray/hmf>`_ (halo mass function calculator written by Stephen Murray)
* `mpi4py <http://mpi4py.scipy.org>`_
* `h5py <http://www.h5py.org/>`_

To download the code, do::

    hg clone https://bitbucket.org/mirochaj/ares ares
    cd ares
    python setup.py install
    
A few lookup tables will be downloaded to ``$ARES/input`` automatically.    


