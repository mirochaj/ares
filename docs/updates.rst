*ARES* Development: Staying Up To Date
======================================
Things are changing fast! To keep up with advancements, a working knowledge of `mercurial <https://mercurial.selenic.com/>`_  will be very useful. If you're reading this, you may already be familiar with mercurial to some degree, as its ``clone`` command can be used to checkout a copy of the most-up-to-date version (the ''tip'' of development) from bitbucket. For example (as in :doc:`install`),  ::

    git clone https://github.com/mirochaj/ares.git ares
    cd ares
    python setup.py install

If you don't plan on making changes to the source code, but would like to make sure you have the most up-to-date version of *ARES*, you'll want to use the ``git pull`` command semi-regularly, i.e., simply type ::

    git pull

from anywhere within the *ARES* folder. Then, to re-install *ARES*: ::

    python setup.py install

If you plan on making changes to *ARES*, you should fork it so that your line of development can run in parallel with the ''main line'' of development. Once you've forked, you should clone a copy just as we did above. For example (note the hyperlink change), ::

    git clone https://github.com/<yourusername>/ares.git ares-<yourname>
    cd ares-<yourname>
    python setup.py install

There are many good tutorials online, but in the following sections we'll go through the commands you'll likely be using all the time.


Checking the Status of your Fork
--------------------------------
You'll typically want to know if, for example, you have changed any files recently and if so, what changes you have made. To do this, type::

    git status

This will print out a list of files in your fork that have either been modified (indicated with ``M``), added (``A``), removed (``R``), or files that are not currently being tracked (``?``). If nothing is returned, it means that you have not made any changes to the code locally, i.e., you have no ''outstanding changes.''

If, however, some files have been changed and you'd like to see just exactly what changes were made, use the ``diff`` command. For example, if when you type ``git status`` you see something like::

    modified: tests/test_solver_chem_h.py

follow-up with::

    git diff tests/test_solver_chem_h.py

and you'll see a modified version of the file with ``+`` symbols indicating additions and ``-`` signs indicating removals. If there have been lots of changes, you may want to pipe the output of ``git diff`` to, e.g., the UNIX program ``less``::

    git diff tests/test_solver_chem_h.py | less

and use ``u`` and ``d`` to navigate up and down in the output.

Making Changes and Pushing them Upstream
----------------------------------------
If you convince yourself that the changes you've made are *good* changes, you should absolutely save them and beam them back up to the cloud. Your changes will either be:

- Modifications to a pre-existing file.
- Creation of an entirely new file.

If you've added new files to *ARES*, they should be listed under the "Untracked files" header of the ``git status`` print-out. To start tracking them, you need to add them to the repository. For example, if we made a new file ``tests/test_new_feature.py``, we would do::

    git add tests/test_new_feature.py

Upon typing ``git status`` again, that file should now have an ``A`` indicator to its left.

If you've modified pre-existing files, they will be marked ``M`` by ``git status``. Once you're happy with your changes, you must *commit* them, i.e.::

    git commit -a -m "Made some changes."

The ``-m`` indicates that what follows in quotes is the ''commit message'' describing what you've done. Commit messages should be descriptive but brief, i.e., try to limit yourself to a sentence (or maybe two), tops. You can see examples of this in the `ares commit history <https://github.com/mirochaj/ares/commits/main>`_.

Note that your changes are still *local*, meaning the *ARES* repository on bitbucket is unaware of them. To remedy that, go ahead and ``push``::

    git push

You'll once again be prompted for your credentials, and then (hopefully) told how many files were updated etc.


Contributing your Changes to the main repository
------------------------------------------------
If you've made changes, you should let us know! The most formal way of doing so is to issue a pull request (PR), which alerts the administrators of *ARES* to review your changes and pull them into the main line of *ARES* development.

Dealing with Conflicts
----------------------
Will cross this bridge when we come to it!
