#!/bin/env bash

for psr in ./J???????/ ./J?????????/; do
	echo "${psr}"
	dm="$(cat "${psr}"/dm.txt | awk '{print $1}')"
	dmerr="$(cat "${psr}"/dm.txt | awk '{print $2}')"
	pars=$(grep -R -l "$(basename "${psr}")" ./pars/)
	sed -i '/DM /s/.*/DM '"${dm} ${dmerr}"'/' "${pars}"
done