#!/bin/env bash

for s in J*sh; do
	pref=${s/.sh/};
	src=$(echo "$pref" | awk -F_ '{print $1}');
	echo "$src" "$pref";
	pat -F -s ../model/"$src".std -m ../model/"$src".m -f tempo2 -A PIS "$pref"*.zap.ar > "${pref}.tim";
	vap -c fracmjd "${pref}*zap.ar" | grep "ar" > "${pref}.vap"
	python3 ../../scripts/dspsrCorrection.py "${pref}.tim";
 done
