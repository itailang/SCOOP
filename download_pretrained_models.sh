#!/usr/bin/env bash

# define a download function
function google_drive_download()
{
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

# download models
google_drive_download 14M0R0AMJhsyGZYHDIp3p8QStCkKCy_bO pretrained_models.zip
unzip pretrained_models.zip
rm pretrained_models.zip
