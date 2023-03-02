#!/bin/env bash

par="${1}";

for cand in ${2:-../cands/*cand}; do
	echo $cand
	prefix=$(basename "${cand}"); 
	prefix=${prefix/.cand/};
	inp=$(ls -1a ../fils/"${prefix}"*P000.fil 2>/dev/null);
	echo $prefix $inp
	if [ ! -f "$inp" ] || [ -z "$inp" ]; then 
		echo "Filterbank for ../fils/${prefix}*P000.fil not found"; 
		continue;
	fi;
	python3 ../../scripts/pulseCand.py -f ../cands/"${prefix}".cand -i "${inp}" -p 0.5 -E "${par}" -o "${prefix}".sh & \
done

jobs -p
wait < <(jobs -p)

