Installation
============
*ARES* depends on:

* `numpy <http://numpy.scipy.org/>`_
* `scipy <http://www.scipy.org/>`_
* `matplotlib <http://matplotlib.sourceforge.net>`_
* `h5py <http://www.h5py.org/>`_

and optionally:

* `progressbar2 <http://progressbar-2.readthedocs.io/en/latest/>`_
* `hmf <http://hmf.readthedocs.org/en/latest/>`_
* `emcee <http://dan.iel.fm/emcee/current/>`_
* `distpy <https://bitbucket.org/ktausch/distpy>`_
* `mpi4py <http://mpi4py.scipy.org>`_
* `pymp <https://github.com/classner/pymp>`_
* `setuptools <https://pypi.python.org/pypi/setuptools>`_
* `mpmath <http://mpmath.googlecode.com/svn-history/r1229/trunk/doc/build/setup.html>`_
* `shapely <https://pypi.python.org/pypi/Shapely>`_
* `descartes <https://pypi.python.org/pypi/descartes>`_

If you have `git` installed, you can clone *ARES* and its entire revision history via: ::

    git clone https://github.com/mirochaj/ares.git
    cd ares
    python setup.py install

*ARES* will look in ``$ARES/input`` for lookup tables of various kinds. To download said lookup tables, run ::

    python remote.py

This might take a few minutes. If something goes wrong with the download, you can run    ::

    python remote.py fresh

to get fresh copies of everything. If you're concerned that a download may have been interrupted and/or the file appears to be corrupted (strange I/O errors may indicate this), you can also just download fresh copies of the particular file you want to replace. For example, to grab a fresh initial conditions file, simply do ::

    python remote.py fresh inits



*ARES* versions
---------------
The first stable release of *ARES* was used in `Mirocha et al. (2015) <http://adsabs.harvard.edu/abs/2015ApJ...813...11M>`_, and is tagged as `v0.1` in the revision history. The tag `v0.2` is associated with `Mirocha, Furlanetto, & Sun (2017) <http://adsabs.harvard.edu/abs/2017MNRAS.464.1365M>`_. Note that these tags are just shortcuts to specific revisions. You can switch between them just like you would switch between branches, e.g.,

::

    git update v0.2

If you're unsure which version is best for you, see the :doc:`history`.

Don't have Python already?
--------------------------
If you do *not* already have Python installed, you might consider downloading `yt <http://yt-project.org/>`_, which has a convenient installation script that will download and install Python and many commonly-used Python packages for you. `Anaconda <https://www.continuum.io/downloads>`_ is also good for this.

Help
----
If you encounter problems with installation or running simple scripts, first check the :doc:`troubleshooting` page in the documentation to see if you're dealing with a common problem. If you don't find your problem listed there, please let me know!
