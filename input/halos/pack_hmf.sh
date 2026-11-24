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

printf '# Would you like to generate new u(k) tables also? (yes/no) '
read -r run_uofk

echo \# Generating HMF tables using ST mass function with $np processors...

if [ $np -eq 1 ]
then
  python generate_hmf_tables.py halo_mf=ST halo_zmin=0 halo_zmax=60 halo_dz=0.05
  python generate_hmf_tables.py halo_mf=PS halo_zmin=5 halo_zmax=30 halo_dz=1
  python generate_hmf_tables.py halo_mf=ST halo_dt=1 halo_tmin=30 halo_tmax=1000 halo_dz=None
  python generate_hmf_tables.py halo_mf=Tinker10 halo_dt=100 halo_tmin=100 halo_tmax=13700 halo_dz=None
  python generate_hmf_tables.py halo_mf=Tinker10 halo_dt=100 halo_tmin=5000 halo_tmax=13700 halo_dz=None
else
  mpirun -np $np python generate_hmf_tables.py halo_mf=ST halo_zmin=0 halo_zmax=60 halo_dz=0.05
  mpirun -np $np python generate_hmf_tables.py halo_mf=PS halo_zmin=5 halo_zmax=30 halo_dz=5
  mpirun -np $np python generate_hmf_tables.py halo_mf=ST halo_dt=1 halo_tmin=30 halo_tmax=1000 halo_dz=None
  mpirun -np $np python generate_hmf_tables.py halo_mf=Tinker10 halo_dt=100 halo_tmin=100 halo_tmax=13700 halo_dz=None
  mpirun -np $np python generate_hmf_tables.py halo_mf=Tinker10 halo_dt=100 halo_tmin=5000 halo_tmax=13700 halo_dz=None
fi
python generate_halo_histories.py halo_mf_ST_planck_TTTEEE_lowl_lowE_best_logM_1400_4-18_t_971_30-1000.hdf5

if [ $run_uofk = 'yes' ]
then
  echo '#####################################################################'
  echo ''
  echo \# Generating new halo profile tables...
  if [ $np -eq 1 ]
  then
    python generate_prof_tables.py
  else
    mpirun -np $np python generate_prof_tables.py
  fi

  cp halo_prof*.hdf5 $HOME/.ares/halos
  echo Copied halo_prof*.hdf5 to $HOME/.ares/halos
  # Remove checkpoints
  rm -f halo_prof_nfw*checkpt*.pkl

else
  echo \# Will not generate new halo profile tables as per user input.
fi

##
# Make tarball: add sub-string containing today's date.
CURRENT_DATE=$(date +"%Y%m%d")

if [ $run_uofk = 'yes' ]
then
  tar -czvf halos_$CURRENT_DATE.tar.gz \
  	halo_mf_ST_planck_TTTEEE_lowl_lowE_best_logM_1400_4-18_z_1201_0-60.hdf5 \
  	halo_mf_PS_planck_TTTEEE_lowl_lowE_best_logM_1400_4-18_z_6_5-30.hdf5 \
  	halo_mf_ST_planck_TTTEEE_lowl_lowE_best_logM_1400_4-18_t_971_30-1000.hdf5 \
  	halo_hist_ST_planck_TTTEEE_lowl_lowE_best_logM_1400_4-18_t_971_30-1000_xM_10_0.10.hdf5 \
    halo_mf_Tinker10_planck_TTTEEE_lowl_lowE_best_logM_1400_4-18_t_137_100-13700.hdf5 \
    halo_prof_nfw_duffy_logM_1400_4-18_t_137_100-13700_lnk_-9.0-11.0_dlnk_0.050.hdf5
else
  tar -czvf halos_$CURRENT_DATE.tar.gz \
  	halo_mf_ST_planck_TTTEEE_lowl_lowE_best_logM_1400_4-18_z_1201_0-60.hdf5 \
  	halo_mf_PS_planck_TTTEEE_lowl_lowE_best_logM_1400_4-18_z_6_5-30.hdf5 \
  	halo_mf_ST_planck_TTTEEE_lowl_lowE_best_logM_1400_4-18_t_971_30-1000.hdf5 \
  	halo_hist_ST_planck_TTTEEE_lowl_lowE_best_logM_1400_4-18_t_971_30-1000_xM_10_0.10.hdf5 \
    halo_mf_Tinker10_planck_TTTEEE_lowl_lowE_best_logM_1400_4-18_t_137_100-13700.hdf5
fi

echo Created tarball halos_$CURRENT_DATE.tar.gz.

mkdir -p $HOME/.ares
mkdir -p $HOME/.ares/halos
cp halos_$CURRENT_DATE.tar.gz $HOME/.ares/halos
cp halo*.hdf5 $HOME/.ares/halos
echo Copied halos_$CURRENT_DATE.tar.gz to $HOME/.ares/halos
echo Copied halo*.hdf5 to $HOME/.ares/halos
  
echo '#####################################################################'
echo "# Must manually upload halos_$CURRENT_DATE.tar.gz files to Drive and update links"
echo '# in ares.util.cli (aux_data dictionary) accordingly.'
echo '#####################################################################'
echo ''
