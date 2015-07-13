Installation
============
*ares* depends on:

* `numpy <http://numpy.scipy.org/>`_
* `scipy <http://www.scipy.org/>`_ 
* `matplotlib <http://matplotlib.sourceforge.net>`_

and optionally:

* `python-progressbar <https://code.google.com/p/python-progressbar/>`_
* `hmf <http://hmf.readthedocs.org/en/latest/>`_ (halo mass function calculator written by Stephen Murray)
* `mpi4py <http://mpi4py.scipy.org>`_
* `h5py <http://www.h5py.org/>`_

If you have mercurial installed, you can clone *ares* and its entire revision history via: ::

    hg clone https://bitbucket.org/mirochaj/ares ares
    cd ares
    python setup.py install
    
If you do not have mercurial installed, and would rather just grab a tarball of the most recent version, select the `Download repository <https://bitbucket.org/mirochaj/ares/downloads>`_ option on bitbucket.
    
Once you've got the code, you'll need to set an environment variable which
points to the *ares* install directory, e.g. (in bash) ::

    export ARES=/users/<yourusername>/ares    
    
A few lookup tables will be downloaded to ``$ARES/input`` automatically.    

Help
----
If you encounter problems with installation or running simple scripts, first check the :doc:`troubleshooting` page in the documentation to see if you're dealing with a common problem. If you don't find your problem listed there, please let me know!