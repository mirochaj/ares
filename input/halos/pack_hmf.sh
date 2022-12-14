#!/bin/bash

me=$(whoami)

echo ''
echo '################################ HMF ################################'

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

printf '# Number of MPI tasks to use for HMF calculation: '
read -r np

echo \# Generating HMF tables using ST mass function with $np processors...

if [ $np -eq 1 ]
then
  python generate_hmf_tables.py halo_mf=ST
  python generate_hmf_tables.py halo_mf=PS halo_zmin=5 halo_zmax=30 halo_dz=1
  python generate_hmf_tables.py halo_mf=ST halo_dt=1 halo_tmin=30 halo_tmax=1000
else
  mpirun -np $np python generate_hmf_tables.py halo_mf=ST
  mpirun -np $np python generate_hmf_tables.py halo_mf=PS halo_zmin=5 halo_zmax=30 halo_dz=5
  mpirun -np $np python generate_hmf_tables.py halo_mf=ST halo_dt=1 halo_tmin=30 halo_tmax=1000
fi

python generate_halo_histories.py halo_mf_ST_planck_TTTEEE_lowl_lowE_best_logM_1400_4-18_t_971_30-1000.hdf5

tar -czvf halos.tar.gz \
	halo_mf_ST_planck_TTTEEE_lowl_lowE_best_logM_1400_4-18_z_1201_0-60.hdf5 \
	halo_mf_PS_planck_TTTEEE_lowl_lowE_best_logM_1400_4-18_z_6_5-30.hdf5 \
	halo_mf_ST_planck_TTTEEE_lowl_lowE_best_logM_1400_4-18_t_971_30-1000.hdf5 \
	halo_hist_ST_planck_TTTEEE_lowl_lowE_best_logM_1400_4-18_t_971_30-1000_xM_10_0.10.hdf5

echo Created tarball halos.tar.gz.

# Copy to dropbox
if [ -n "$DROPBOX" ];
then
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

  if [ -d "$FILE/input/hmf" ]
  then
    :
  else
    mkdir $FILE/input/halos
    echo "Created $FILE/input/halos."
  fi

  cp halos.tar.gz $FILE/input/halos
  echo Copied hmf.tar.gz to $FILE/input/halos

else
  echo Must manually upload halos.tar.gz to DROPBOX and update links
  echo in remote.py accordingly.
fi

echo '#####################################################################'
echo ''
