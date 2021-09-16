#!/bin/bash

me=$(whoami)

echo ''
echo '################################ tau ################################'

if [ $me = 'jordanmirocha' ]
then
  echo \# Hey, J.M., right this way.
else
  printf '# Are you sure you want to proceed [yes/no]: '
  read -r areyousure

  if [ $areyousure = 'yes' ]
  then
    echo \# OK, hope you know what you are doing.
  else
    exit 1
  fi
fi

printf '# Number of MPI tasks to use for tau calculation: '
read -r np

echo \# Generating 4 optical depth tables with $np processors...

if [ $np -eq 1 ]
then
  python generate_optical_depth_tables.py tau_redshift_bins=400 include_He=1
  python generate_optical_depth_tables.py tau_redshift_bins=400 include_He=0
  python generate_optical_depth_tables.py tau_redshift_bins=1000 include_He=1
  python generate_optical_depth_tables.py tau_redshift_bins=1000 include_He=0
else
  mpirun -np $np python generate_optical_depth_tables.py tau_redshift_bins=400 include_He=1
  mpirun -np $np python generate_optical_depth_tables.py tau_redshift_bins=400 include_He=0
  mpirun -np $np python generate_optical_depth_tables.py tau_redshift_bins=1000 include_He=1
  mpirun -np $np python generate_optical_depth_tables.py tau_redshift_bins=1000 include_He=0
fi

tar -czvf tau.tar.gz optical_depth_planck_TTTEEE_lowl_lowE_best_H_400x862_z_5-60_logE_2.3-4.5.hdf5 \
	optical_depth_planck_TTTEEE_lowl_lowE_best_He_400x862_z_5-60_logE_2.3-4.5.hdf5 \
	optical_depth_planck_TTTEEE_lowl_lowE_best_H_1000x2158_z_5-60_logE_2.3-4.5.hdf5 \
	optical_depth_planck_TTTEEE_lowl_lowE_best_He_1000x2158_z_5-60_logE_2.3-4.5.hdf5

# Copy to dropbox
echo Created tarball tau.tar.gz.

# Copy to dropbox
FILE=$DROPBOX/ares
if [ -d "$FILE" ]
then
  :
else
  mkdir $FILE
  echo "Created $FILE."
fi

if [ -d "$FILE/input" ]
then
  :
else
  mkdir $FILE/input
  echo "Created $FILE/input."
fi

if [ -d "$FILE/input/optical_depth" ]
then
  :
else
  mkdir $FILE/input/optical_depth
  echo "Created $FILE/input/optical_depth."
fi

cp tau.tar.gz $FILE/input/optical_depth
echo Copied tau.tar.gz to $FILE/input/optical_depth

echo '#####################################################################'
echo ''
