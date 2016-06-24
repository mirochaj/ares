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
* `setuptools <https://pypi.python.org/pypi/setuptools>`_
* `mpmath <http://mpmath.googlecode.com/svn-history/r1229/trunk/doc/build/setup.html>`_
* `shapely <https://pypi.python.org/pypi/Shapely>`_
* `descartes <https://pypi.python.org/pypi/descartes>`_

If you have mercurial installed, you can clone *ares* and its entire revision history via: ::

    hg clone https://bitbucket.org/mirochaj/ares ares
    cd ares
    python setup.py install
    
If you do not have mercurial installed, and would rather just grab a tarball
of the most recent version, select the `Download repository
<https://bitbucket.org/mirochaj/ares/downloads>`_ option on bitbucket.

You'll need to set an environment variable which points to the *ares* install directory, e.g. (in bash) ::

    export ARES=/users/<yourusername>/ares

*ares* will look in ``$ARES/input`` for lookup tables of various kinds. To download said lookup tables, run ::

    python remote.py
    
This might take a few minutes. If something goes wrong with the download, you can run    ::

    python remote.py fresh
    
to get fresh copies of everything.

*ares* branches
---------------
*ares* has two main branches. The first, ``default``, is meant to be stable, and will only be updated with critical bug fixes or upon arrival at major development milestones. The "bleeding edge" lives in the ``ares-dev`` branch, and while you are more likely to find bugs in ``ares-dev``, you will also find the newest features. 

By default after you clone *ares* you'll be using the ``default`` branch. To switch, simply type:  ::

    hg update ares-dev
    
To switch back, ::

    hg update default
    
For a nice discussion of the pros and cons of different branching techniques in mercurial, `this article is a nice place to start <http://stevelosh.com/blog/2009/08/a-guide-to-branching-in-mercurial/>`_. 

Don't have Python already?
--------------------------
If you do *not* already have Python installed, you might consider downloading `yt <http://yt-project.org/>`_, which has a convenient installation script that will download and install Python and many commonly-used Python packages for you.

Help
----
If you encounter problems with installation or running simple scripts, first check the :doc:`troubleshooting` page in the documentation to see if you're dealing with a common problem. If you don't find your problem listed there, please let me know!
