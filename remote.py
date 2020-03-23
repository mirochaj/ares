#!/usr/bin/env python

import os, re, sys, tarfile, zipfile
try:
    from urllib.request import urlretrieve # this option only true with Python3
except:
    from urllib import urlretrieve

options = sys.argv[1:]

# Auxiliary data downloads
# Format: [URL, file 1, file 2, ..., file to run when done]

_bpass_v1_links = ['sed_bpass_z{!s}_tar.gz'.format(Z) \
    for Z in ['001', '004', '008', '020', '040']]

aux_data = \
{
 'hmf': ['https://www.dropbox.com/s/7ui8qqvl6eutyk2/hmf.tar.gz?dl=1',
     'hmf.tar.gz',
    None],
 'inits': ['https://www.dropbox.com/s/c6kwge10c8ibtqn/inits.tar.gz?dl=1',
     'inits.tar.gz',
    None],    
 'optical_depth': ['https://www.dropbox.com/s/dqeeetvaea4ap8q/tau.tar.gz?dl=1',
     'tau.tar.gz',
    None],
 'secondary_electrons': ['https://www.dropbox.com/s/jidsccfnhizm7q2/elec_interp.tar.gz?dl=1',
     'elec_interp.tar.gz',
     'read_FJS10.py'],
 'starburst99': ['http://www.stsci.edu/science/starburst99/data',
    'data.tar.gz',
    None],
 'bpass_v1': ['http://bpass.auckland.ac.nz/2/files'] + _bpass_v1_links + [None],
 'bpass_v1_stars': ['http://bpass.auckland.ac.nz/1/files',
    'starsmodels_tar.gz',
    None],
 #'bpass_v2': ['https://drive.google.com/file/d/'] + \
 #    ['bpassv2-imf{}-300tar.gz'.format(IMF) for IMF in [100, 135]] + \
 #     [None],    
 #'behroozi2013': ['http://www.peterbehroozi.com/uploads/6/5/4/8/6548418/',
 #   'sfh_z0_z8.tar.gz', 'observational-data.tar.gz', None]
 'edges': ['http://loco.lab.asu.edu/download',
    '790/figure1_plotdata.csv',
    '792/figure2_plotdata.csv', 
    None],
 'nircam': ['https://jwst-docs.stsci.edu/files/73022379/73022381/1/1486438006000',
     'nircam_throughputs_22April2016_v4.tar.gz',
     None],
 'wfc3': ['http://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/wfc3/performance/throughputs/_documents/',
    'IR.zip',
     None],
 'wfc': ['https://www.dropbox.com/s/zv8qomgka9fkiek/wfc.tar.gz?dl=1',
     'wfc.tar.gz',
    None],
 'irac': ['https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/calibrationfiles/spectralresponse/',
    '080924ch1trans_full.txt',
    '080924ch2trans_full.txt',
    None],
 #'wfc': ['http://www.stsci.edu/hst/acs/analysis/throughputs/tables',
 #   'wfc_F435W.dat',
 #   'wfc_F606W.dat',
 #   'wfc_F775W.dat',
 #   'wfc_F814W.dat',
 #   'wfc_F850LP.dat',
 #   None],
 'planck': ['https://pla.esac.esa.int/pla/aio',
    'product-action?COSMOLOGY.FILE_ID=COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE_R3.00.zip',
    'product-action?COSMOLOGY.FILE_ID=COM_CosmoParams_base-plikHM-zre6p5_R3.01.zip',
    'product-action?COSMOLOGY.FILE_ID=COM_CosmoParams_base-plikHM_R3.01.zip',
    None]
}

if not os.path.exists('input'):
    os.mkdir('input')

os.chdir('input')

files = []
if (len(options) > 0) and ('clean' not in options):
    if 'minimal' in options:
        to_download = ['inits', 'secondary_electrons', 'hmf', 'wfc', 'wfc3']
        files = [None, None, None, None, None]
    elif 'clean' in options:
        to_download = aux_data.keys()
        files = [None] * len(to_download)
    else:
        ct = 0
        to_download = []
        for key in options:
            if key == 'fresh':
                continue
                
            if re.search(':', key):
                pre, post = key.split(':')
                to_download.append(pre)
                files.append(int(post))
            else:
                to_download.append(key)
                files.append(None)
                
            if 'basic' in options:
                break
                
            ct += 1    
                
        if to_download == [] and 'fresh' in options:
            to_download = aux_data.keys()
            files = [None] * len(to_download)
else:
    to_download = list(aux_data.keys())
    to_download.remove('cosmo_params')
    files = [None] * len(to_download)
        
for i, direc in enumerate(to_download):
                
    if not os.path.exists(direc):
        os.mkdir(direc)
    
    os.chdir(direc)
    
    web = aux_data[direc][0]
    
    if files[i] is None:
        fns = aux_data[direc][1:-1]
    else:
        fns = [aux_data[direc][1:-1][files[i]]]
        
    for fn in fns:
         
        if '/' in fn:
            _fn = fn[fn.rfind('/')+1:]#fn.split('/') 
        else:
            _fn = fn    
            
        if os.path.exists(fn):
            if ('fresh' in options) or ('clean' in options):
                os.remove(fn)
            else:
                continue
            
        # 'clean' just deletes files, doesn't download new ones
        if 'clean' in options:
            continue
    
        if 'dropbox' in web:
            print("Downloading {0!s} to {1!s}...".format(web, fn))
                        
            try:
                urlretrieve('{0!s}'.format(web), fn)
            except:
                print("WARNING: Error downloading {0!s}".format(web))
                continue
        else:    
            print("Downloading {0!s}/{1!s}...".format(web, fn))
            
            try:
                urlretrieve('{0!s}/{1!s}'.format(web, fn), _fn)
            except:
                print("WARNING: Error downloading {0!s}/{1!s}".format(web, fn))
                continue
        
        # If it's a zip, unzip and move on.
        if re.search('.zip', _fn) and (not re.search('tar', _fn)):
            zip_ref = zipfile.ZipFile(_fn, 'r')
            zip_ref.extractall()
            zip_ref.close()            
            continue
        
        # If it's not a tarball, move on
        if (not re.search('tar', _fn)) and (not re.search('tgz', _fn)):
            continue
            
        # Otherwise, unpack it
        try:
            tar = tarfile.open(_fn)
            tar.extractall()
            tar.close()
        except:
            print("WARNING: Error unpacking {0!s}".format(_fn))
        
        if direc != 'cosmo_params': 
            continue
            
        _files = os.listdir(os.curdir)
        for _file in _files:
            if _file=='COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE_R3.00':
                try:
                    os.chdir('COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE_R3.00')
                    tarfiles=os.listdir(os.curdir)
                    for pack_file in tarfiles:
                        tar = tarfile.open(pack_file)
                        tar.extractall()
                        tar.close()
                except:
                    print('Could not unpack the planck chains')
    
    # Run a script [optional]
    if aux_data[direc][-1] is not None:
        try:
            execfile(aux_data[direc][-1])
        except:
            print("WARNING: Error running {!s}".format(aux_data[direc][-1]))

    os.chdir('..')

