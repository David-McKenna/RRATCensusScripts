#!/bin/env bash

for fil in ./J???????/ ./J?????????/; do  
	pushd "${fil}" || exit
	python3 ../scripts/extractMeta_tskyfix.py
	if [ -f 'PERIODIC' ]; then 
		python3 ../scripts/extractMeta_periodic.py
	fi
	popd || exit
done
