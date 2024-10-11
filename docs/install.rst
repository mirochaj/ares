Installation
============
ARES depends on:

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

If you have ``git`` installed, you can clone ARES and its entire revision history via: ::

    git clone https://github.com/mirochaj/ares.git
    cd ares
    pip install -e .

ARES will look in ``$HOME/.ares`` for lookup tables of various kinds. To download said lookup tables, run ::

    ares download all

This might take a few minutes. If something goes wrong with the download, you can run    ::

    ares download --fresh

to get fresh copies of everything. If you're concerned that a download may have been interrupted and/or the file appears to be corrupted (strange I/O errors may indicate this), you can also just download fresh copies of the particular file you want to replace. For example, to grab a fresh initial conditions file, simply do ::

    ares download planck --fresh


Trouble with external datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The file downloads described above have been known to fail on occasion. There are a variety of reasons for this:

- Intermittent network connectivity might mean only one download fails while the rest proceed no problem. In this case, running with the ``--fresh`` flag should do the trick.
- Over time, some of these files may be moved to a new site, and so the hardcoded links in ARES will point to the wrong place. If you copy-paste the link into your browser and there is no file to be found, please let me know. Better yet, if you can find the new home of this file, go ahead and submit a pull request with the updated path (which you should find in ``ares.util.cli`` in the ``aux_data`` dictionary).
- There are also some potentially-OS dependent failure modes. For example, some of the files downloaded are ``.zip`` files or tarballs, and so there is an unpacking step that may actually be to blame for the failure. In the future, it's probably worth handling these errors separately, but in the meantime, please check if the error is a red herring by verifying whether or not the file has been downloaded, and if it has, try to unpack it yourself by hand.


Downloading BPASS versions >= 2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you want to use newer versions of BPASS, you'll have to download those files by hand from the Google Drive folders where they are hosted, which you can navigate to from `here <https://bpass.auckland.ac.nz/9.html>`_. Then, unpack in ``$HOME/.ares/bpass_v2``.

Help
----
If you encounter problems with installation or running simple scripts, first check the :doc:`troubleshooting` page in the documentation to see if you're dealing with a common problem. If you don't find your problem listed there, please let me know!
