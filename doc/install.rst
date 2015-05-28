Installation
============
*ares* depends on:

* `numpy <http://numpy.scipy.org/>`_
* `scipy <http://www.scipy.org/>`_ 
* `matplotlib <http://matplotlib.sourceforge.net>`_ (for built-in analysis routines)

and optionally:

* `python-progressbar <https://code.google.com/p/python-progressbar/>`_
* `hmf <http://hmf.readthedocs.org/en/latest/>`_ (halo mass function calculator written by Stephen Murray)
* `mpi4py <http://mpi4py.scipy.org>`_
* `h5py <http://www.h5py.org/>`_

To download the code, type::

    hg clone https://bitbucket.org/mirochaj/ares ares
    cd ares
    python setup.py install
    
It would be in your best interest to set an environment variable which points
to the *ares* install directory, e.g. (in bash) ::

    export ARES=/users/<yourusername>/ares    
    
A few lookup tables will be downloaded to ``$ARES/input`` automatically.    

