#!/bin/bash

me=$(whoami)
degrade=100

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

echo \# Generating BPASS tables at $degrade x lower resolution.

python degrade_bpass_seds.py $degrade

tar -czvf sed_degraded.tar.gz \
	SEDS/sed.bpass.constant.nocont.sin.z020.deg$degrade \
	SEDS/sed.bpass.instant.nocont.sin.z020.deg$degrade

echo Created tarball sed_degraded.tar.gz.

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

if [ -d "$FILE/input/bpass_v1" ]
then
  :
else 
  mkdir $FILE/input/bpass_v1
  echo "Created $FILE/input/bpass_v1."
fi

cp sed_degraded.tar.gz $FILE/input/bpass_v1/
echo Copied sed_degraded.tar.gz to $FILE/input/bpass_v1/

echo '#####################################################################'
echo ''