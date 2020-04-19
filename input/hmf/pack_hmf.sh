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
  python generate_hmf_tables.py hmf_model=ST
  python generate_hmf_tables.py hmf_model=ST hmf_dt=1 hmf_tmin=30 hmf_tmax=1000
else
  mpirun -np $np python generate_hmf_tables.py hmf_model=ST
  mpirun -np $np python generate_hmf_tables.py hmf_model=ST hmf_dt=1 hmf_tmin=30 hmf_tmax=1000
fi

python generate_halo_histories.py hmf_ST_planck_TTTEEE_lowl_lowE_best_logM_1400_4-18_t_971_30-1000.hdf5

tar -czvf hmf.tar.gz \
	hmf_ST_planck_TTTEEE_lowl_lowE_best_logM_1400_4-18_z_1201_0-60.hdf5 \
	hmf_ST_planck_TTTEEE_lowl_lowE_best_logM_1400_4-18_t_971_30-1000.hdf5 \
	hgh_ST_planck_TTTEEE_lowl_lowE_best_logM_1400_4-18_t_971_30-1000_xM_10_0.10.hdf5

echo Created tarball hmf.tar.gz.

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

if [ -d "$FILE/input/hmf" ]
then
  :
else 
  mkdir $FILE/input/hmf	
  echo "Created $FILE/input/hmf."
fi

cp hmf.tar.gz $FILE/input/hmf
echo Copied hmf.tar.gz to $FILE/input/hmf

echo '#####################################################################'
echo ''