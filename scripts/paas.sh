#!/bin/env bash

inp="${1}"
src=$(basename $inp | awk -F_ '{print $1}')

paas -w "./${src}.m" -s "./${src}.std" -j "./${src}.txt" "${inp}" -D -d /xwin -i
