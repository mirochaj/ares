"""

download_gsm.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Dec 15 20:13:01 MST 2014

Description: Download Global Sky Model (10 MHz - 100 GHz) from 
de Oliveira-Costa et al. (2008).

"""

import urllib, tarfile, os

gsm_fn = 'gsm.tar.gz'
gsm_prefix = 'http://xte.mit.edu/angelica/gsm/'

if not os.path.exists(gsm_fn):
    print "\nDownloading %s/%s..." % (gsm_prefix, gsm_fn)
    urllib.urlretrieve('%s/%s' % (gsm_prefix, gsm_fn), gsm_fn)

tar = tarfile.open(gsm_fn)
tar.extractall()
tar.close()

