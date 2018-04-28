#!/usr/bin/env python

import os, re, sys, tarfile
try:
    from urllib.request import urlretrieve # this option only true with Python3
except:
    from urllib import urlretrieve

options = sys.argv[1:]

ares_link = 'https://bitbucket.org/mirochaj/ares'

# Auxiliary data downloads
# Format: [URL, file 1, file 2, ..., file to run when done]

#_bpassv2_links = \
#[
# ''
#]
#
aux_data = \
{
 'hmf': ['{!s}/downloads'.format(ares_link),
    'hmf_ST_logM_1200_4-16_z_1141_3.0-60.0.npz',
    None],
 'inits': ['{!s}/downloads'.format(ares_link), 
     'initial_conditions.npz',
     None],    
 'optical_depth': ['{!s}/downloads'.format(ares_link),
    'optical_depth_He_200x429_z_5-60_logE_2.3-4.5.npz',
    'optical_depth_He_400x862_z_5-60_logE_2.3-4.5.npz',
    'optical_depth_He_1000x2158_z_5-60_logE_2.3-4.5.npz',
    None],
 'secondary_electrons': ['{!s}/downloads'.format(ares_link),
    'elec_interp.tar.gz',
    'read_FJS10.py'],
 'starburst99': ['http://www.stsci.edu/science/starburst99/data',
    'data.tar.gz',
    None],
 #'hm12': ['http://www.ucolick.org/~pmadau/CUBA/Media',
 #   'UVB.out', 
 #   'emissivity.out', 
 #   None],
 'bpass_v1': ['http://bpass.auckland.ac.nz/2/files'] + \
    ['sed_bpass_z{!s}_tar.gz'.format(Z) for Z in ['001', '004', '008', '020', '040']] + \
    [None],
 #'bpass_v2': ['https://drive.google.com/file/d/'] + \
 #    ['bpassv2-imf{}-300tar.gz'.format(IMF) for IMF in [100, 135]] + \
 #     [None],    
 #'behroozi2013': ['http://www.peterbehroozi.com/uploads/6/5/4/8/6548418/',
 #   'sfh_z0_z8.tar.gz', 'observational-data.tar.gz', None]
 'edges': ['http://loco.lab.asu.edu/download/792/',
    'figure2_plotdata.csv', 
    None]
}

if not os.path.exists('input'):
    os.mkdir('input')

os.chdir('input')

files = []
if (len(options) > 0) and ('clean' not in options):
    if 'minimal' in options:
        to_download = ['inits', 'secondary_electrons', 'hmf']
        files = [None, None, None]
    elif 'clean' in options:
        to_download = aux_data.keys()
        files = [None] * len(to_download)
    else:
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
                
        if to_download == [] and 'fresh' in options:
            to_download = aux_data.keys()
            files = [None] * len(to_download)        
else:
    to_download = aux_data.keys()
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
            
        if os.path.exists(fn):
            if ('fresh' in options) or ('clean' in options):
                os.remove(fn)
            else:
                continue
            
        # 'clean' just deletes files, doesn't download new ones
        if 'clean' in options:
            continue
    
        print("Downloading {0!s}/{1!s}...".format(web, fn))
        
        try:
            urlretrieve('{0!s}/{1!s}'.format(web, fn), fn)
        except:
            print("WARNING: Error downloading {0!s}/{1!s}".format(web, fn))
            continue
        
        # If it's not a tarball, move on
        if not re.search('tar', fn):
            continue
            
        # Otherwise, unpack it
        try:
            tar = tarfile.open(fn)
            tar.extractall()
            tar.close()
        except:
            print("WARNING: Error unpacking {0!s}/{1!s}".format(web, fn))
    
    # Run a script [optional]
    if aux_data[direc][-1] is not None:
        try:
            execfile(aux_data[direc][-1])
        except:
            print("WARNING: Error running {!s}".format(aux_data[direc][-1]))

    os.chdir('..')

