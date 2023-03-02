#!/bin/env bash

for cand in *cand; do
	prefix=${cand/.cand/};
	fil=$(ls -1a ../fils/"$prefix"*P000.fil 2>/dev/null);
	if [ ! -f "$fil" ] || [ -z "$fil" ]; then 
		continue; 
	fi;
	echo "${fil}";
	python3.6 ../../scripts/quickplotcand.py -r -q -b 6 -t -k -i "${fil}" -f "./${prefix}.cand" -o "./${prefix}_plots/${prefix}";
done
