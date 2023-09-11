#!/bin/env bash

for s in J*sh; do
	pref=${s/.sh/};
	src=$(echo "$pref" | awk -F_ '{print $1}');
	echo "$src" "$pref";
	pat -F -s ../model/"$src".std -m ../model/"$src".m -f tempo2 -A PIS "$pref"*.zap.ar > "${pref}.tim";
	vap -c fracmjd "${pref}*zap.ar" | grep "ar" > "${pref}.vap"
	python3 ../../scripts/dspsrCorrection.py "${pref}.tim";
	cat "${pref}.tim" | awk '{print $2}' | grep -v "^$" | while read freq; do
		if [ "${freq}" == "1" ]; then
			continue
		fi
		sed -i "s/${freq}/197.55859375/g" "${pref}_corrected.tim"
	done

 done
