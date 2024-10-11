External datasets
=================

Trouble with external datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The file downloads described above have been known to fail on occasion. There are a variety of reasons for this:

- Intermittent network connectivity might mean only one download fails while the rest proceed no problem. In this case, running with the ``--fresh`` flag should do the trick.
- Over time, some of these files may be moved to a new site, and so the hardcoded links in ARES will point to the wrong place. If you copy-paste the link into your browser and there is no file to be found, please let me know. Better yet, if you can find the new home of this file, go ahead and submit a pull request with the updated path (which you should find in ``ares.util.cli`` in the ``aux_data`` dictionary).
- There are also some potentially-OS dependent failure modes. For example, some of the files downloaded are ``.zip`` files or tarballs, and so there is an unpacking step that may actually be to blame for the failure. In the future, it's probably worth handling these errors separately, but in the meantime, please check if the error is a red herring by verifying whether or not the file has been downloaded, and if it has, try to unpack it yourself by hand.

Downloading BPASS versions >= 2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you want to use newer versions of BPASS, you'll have to download those files by hand from the Google Drive folders where they are hosted, which you can navigate to from `here <https://bpass.auckland.ac.nz/9.html>`_. Then, unpack in ``$HOME/.ares/bpass_v2``.
