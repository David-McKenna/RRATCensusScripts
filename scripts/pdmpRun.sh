#!/bin/env bash

path=${1:-./}

# Remove fit files from failed runs
find "${path}" -type f -name "*pulseTOA*.fit" -size -4k -delete

# Find and run pdmp on all zapped archives in the directory tree
find "${path}" -maxdepth 3 -type f -name "*.zap.ar" -print0 | xargs -0 -P 16 -n 1 bash scripts/pdmpWrapper.sh
