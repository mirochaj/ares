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
_bpass_tests = 'https://www.dropbox.com/s/bq10l5f6gzqqvu7/sed_degraded.tar.gz?dl=1'

aux_data = \
{
 'halos': ['https://www.dropbox.com/s/8df7rsskr616lx5/halos.tar.gz?dl=1',
     'halos.tar.gz',
    None],
 'inits': ['https://www.dropbox.com/s/c6kwge10c8ibtqn/inits.tar.gz?dl=1',
     'inits.tar.gz',
    None],
 'optical_depth': ['https://www.dropbox.com/s/ol6240qzm4w7t7d/tau.tar.gz?dl=1',
     'tau.tar.gz',
    None],
 'secondary_electrons': ['https://www.dropbox.com/s/jidsccfnhizm7q2/elec_interp.tar.gz?dl=1',
     'elec_interp.tar.gz',
     'read_FJS10.py'],
 'starburst99': ['http://www.stsci.edu/science/starburst99/data',
    'data.tar.gz',
    None],
 'bpass_v1': ['http://bpass.auckland.ac.nz/2/files'] + _bpass_v1_links \
     + [_bpass_tests] + [None],
 'bpass_v1_stars': ['http://bpass.auckland.ac.nz/1/files',
    'starsmodels_tar.gz',
    None],
 #'bpass_v2': ['https://drive.google.com/file/d/'] + \
 #    ['bpassv2-imf{}-300tar.gz'.format(IMF) for IMF in [100, 135]] + \
 #     [None],
 #'behroozi2013': ['http://www.peterbehroozi.com/uploads/6/5/4/8/6548418/',
#    'sfh_z0_z8.tar.gz', 'observational-data.tar.gz', None],
 'umachine-data': ['http://halos.as.arizona.edu/UniverseMachine/DR1',
    'umachine-dr1-obs-only.tar.gz', None],
 'edges': ['http://loco.lab.asu.edu/download',
    '790/figure1_plotdata.csv',
    '792/figure2_plotdata.csv',
    None],
 'nircam': ['https://jwst-docs.stsci.edu/files/97978094/97978135/1/1596073152953',
     'nircam_throughputs_22April2016_v4.tar.gz',
     None],
 'wfc3': ['https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/wfc3/performance/throughputs/_documents/',
    'IR.zip',
     None],
 'wfc': ['https://www.dropbox.com/s/zv8qomgka9fkiek/wfc.tar.gz?dl=1',
     'wfc.tar.gz',
    None],
 'irac': ['https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/calibrationfiles/spectralresponse/',
    '080924ch1trans_full.txt',
    '080924ch2trans_full.txt',
    None],
 'roman': ['https://roman.gsfc.nasa.gov/science/RRI/',
    'Roman_effarea_20201130.xlsx',
    None],
 'rubin': ['https://s3df.slac.stanford.edu/data/rubin/sim-data/rubin_sim_data',
    'throughputs_aug_2021.tgz',
    None],
 'spherex': ['https://github.com/SPHEREx/Public-products/archive/refs/heads',
    'master.zip',
    None],
 'wise': ['https://www.astro.ucla.edu/~wright/WISE',
    'RSR-W1.txt', 'RSR-W2.txt',
    None],
 '2mass': ['http://svo2.cab.inta-csic.es/svo/theory/fps3/',
    'getdata.php?format=ascii&id=2MASS/2MASS.J',
    'getdata.php?format=ascii&id=2MASS/2MASS.H',
    'getdata.php?format=ascii&id=2MASS/2MASS.K',
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

extra = ['nircam', 'irac', 'roman', 'edges', 'bpass_v1_stars']
needed_for_tests = ['inits', 'secondary_electrons', 'halos', 'wfc', 'wfc3',
    'planck', 'bpass_v1', 'optical_depth']
needed_for_tests_fn = ['inits.tar.gz', 'elec_interp.tar.gz', 'halos.tar.gz',
    'IR.zip', 'wfc.tar.gz', aux_data['planck'][1], 'sed_degraded.tar.gz',
    'tau.tar.gz']

files = []
if (len(options) > 0) and ('clean' not in options):
    if ('minimal' in options) or ('test' in options):
        to_download = needed_for_tests
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

            ct += 1

        if to_download == [] and 'fresh' in options:
            to_download = aux_data.keys()
            files = [None] * len(to_download)
else:
    to_download = []
    for key in aux_data.keys():
        if key in extras:
            continue
        to_download.append(key)

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

    for i, fn in enumerate(fns):

        if fn.startswith('https'):
            _web = fn
            _fn = fn[fn.rfind('/')+1:fn.rfind('?')]
        else:
            _web = web
            _fn = fn

        if ('minimal' in options) or ('test' in options):
            if _fn not in needed_for_tests_fn:
                print("File {} not needed for minimal build.".format(_fn))
                continue

        if '/' in _fn:
            _fn_ = _fn[_fn.rfind('/')+1:]
        else:
            _fn_ = _fn

        if os.path.exists(_fn_) and ('test' not in options):
            if ('fresh' in options) or ('clean' in options):
                os.remove(_fn_)
            else:
                continue

        # 'clean' just deletes files, doesn't download new ones
        if 'clean' in options:
            continue

        if 'dropbox' in _web:
            print("# Downloading {0!s} to {1!s}...".format(_web, _fn_))

            if 'test' in options:
                continue

            try:
                urlretrieve('{0!s}'.format(_web), _fn_)
            except:
                print("WARNING: Error downloading {0!s}".format(_web))
                continue
        else:
            print("# Downloading {0!s}/{1!s}...".format(_web, _fn_))

            if 'test' in options:
                continue

            try:
                urlretrieve('{0!s}/{1!s}'.format(_web, fn), _fn_)
            except:
                print("# WARNING: Error downloading {0!s}/{1!s}".format(_web, _fn_))
                continue

        # If it's a zip, unzip and move on.
        if re.search('.zip', _fn_) and (not re.search('tar', _fn_)):
            zip_ref = zipfile.ZipFile(_fn_, 'r')
            zip_ref.extractall()
            zip_ref.close()
            continue

        # If it's not a tarball, move on
        if (not re.search('tar', _fn_)) and (not re.search('tgz', _fn_)):
            continue

        # Otherwise, unpack it
        try:
            tar = tarfile.open(_fn_)
            tar.extractall()
            tar.close()
        except:
            print("# WARNING: Error unpacking {0!s}".format(_fn_))

        if direc != 'planck':
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
                    print('# Could not unpack the planck chains')

    # Run a script [optional]
    if aux_data[direc][-1] is not None:
        try:
            if sys.version_info[0] < 3:
                execfile(aux_data[direc][-1])
            else:
                exec(open(aux_data[direc][-1]).read())
        except:
            print("# WARNING: Error running {!s}".format(aux_data[direc][-1]))

    os.chdir('..')
