#!/bin/env bash

for src in ./J???????/ ./J?????????/; do 
	if [ ! -d "${src}/pdmpDump/" ]; then 
		echo "Unable to find directory $src/pdmpDump/"; 
		continue
	fi
	src=$(basename "${src}")
	pushd "${src}"/pdmpDump/ || exit

	if [ "$(ls -1a *.fit 2>/dev/null | wc -l)" -eq 0 ]; then
		echo "No fits found for ${src}/pdmp; continuing.";
		popd || exit
		continue
	fi

	cat ./*.fit | grep "Best DM" | awk '{print $4,$10}' > "${src}.dms"
	python3 ../../scripts/dmFit.py "./${src}.dms" ../dm.txt
	popd || exit

done
