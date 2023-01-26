#!/bin/bash

# cd scratch place
cd ~/Datasets

# Download zip dataset from Google Drive
filename='CheXpert_Processed.tar'
fileid='1xVOF8zwiu0unyFGCDSroXak1X3fb2qJL'
wget -c --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt

#Unzip
tar -xf ${filename}
# rm ${filename}
cd