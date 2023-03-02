#!/bin/env bash

# Generate variables
fil=${1:-NO_INPUT_FILE}
source=$(dirname "$(dirname "${fil}")")
file=$(basename "$1")
outputName=${source}/pdmpDump/${file/.zap.ar/.fit}

# Make sure inputs and outputs exists
if [ ! -d "./pdmpScan/${source}/" ]; then
	mkdir -p "./pdmpScan/$source/"
fi

if [ ! -d "./${source}/pdmpDump/" ]; then
	mkdir -p "./${source}/pdmpDump/"
fi

if [ ! -f "${fil}" ]; then
	echo "ERROR: File ${fil} not found."
	exit
fi

# Skip completed / flagged files
if [ -f "$outputName" ]; then
	exit
fi

if [ -f "$outputName""_flagged" ]; then
	exit
fi

# Do the atual work
echo "Beginning processing for ${fil} -> ${outputName}"
pdmp -f -g "${file/.zap.ar/.ps}/ps" -dr 1.0 -ds 0.001 -do 0 -bf test -b -pr 0 -ps 0 -ar 0 -as 0 -output-pdm-s/n "${fil}" -g png | grep -v "INTER" > "${outputName}"
mv "${file/.zap.ar/.ps}" "./${source}/pdmpDump/"
