#!/bin/env bash

candFolder=${1:-FIXME}

if [ "${candFolder}" == "FIXME" ]; then
        candFolder=$(find /mnt/ucc4_data2/data/David/ -maxdepth 1 -type d -name "rrat_20*" | sort | tail -n 1)
fi;

mkdir tmpLink/
rm ./tmpLink/*

for cands in "${candFolder}/"*_cands/; do
        if [[ ${cands} = *full_cand* ]]; then
                continue;
        fi; 


        src=$(basename "${cands}" | awk -F_ '{print $1}'); 
        if [ ! -d "./$src/" ]; then
                echo "No output folder avaialble for ${src}."
                continue;
        fi;

        ts=$(basename "${cands}" | awk -F_ '{print $2}');
        awkp="./$src/cands/awk.part";
        echo "$src" "$ts" "$awkp" "${cands}";
        cat "$cands"*cand | awk -f "$awkp" > "./$src/cands/$src""_$ts.cand";
        cat "./$src/cands/$src""_$ts.cand";
        printf "\n\n\n";

        filname=$(ls -1a "$candFolder"/../"$src""_""$ts"*P000.fil);
        if [ ! -f "$filname" ]; then
                echo "Unable to find ${filname}"
                continue;
        fi;
        bn=$(basename "$filname");
        echo "$bn" "$filname";
        ln -s "$filname" ./"$src"/fils/"$bn";
        pushd "$src"/cands/ || exit;
        bash ../../scripts/plotCands.sh;
        ln -s "../$src/cands/${src}_${ts}_plots/" ../../tmpLink/
        popd || exit;
done;

for cands in "$candFolder/"*_cands/; do \
        if [[ $cands = *full_cand* ]]; then \
                continue; \
        fi; \
        \
        src=$(basename "$cands" | awk -F_ '{print $1}'); \
        if [ ! -d "./$src/" ]; then
                continue; \
        fi; \
        ts=$(basename "$cands" | awk -F_ '{print $2}'); \
        echo "$src" "$ts" "$awkp" "$cands"; \
        nano "./${src}/cands/${src}_${ts}.cand"; \
done;

read -r -n 1 -p "Awaiting input before continuing"

startDate=$(date "+%Y-%m-%d")
dt=$(date "+%Y-%m")
for cands in "$candFolder/"*_cands/; do \
        if [[ $cands = *full_cand* ]]; then \
                continue; \
        fi; \
        \
        src=$(basename "$cands" | awk -F_ '{print $1}'); \
        if [ ! -d "./$src/" ]; then
                continue; \
        fi; \
        \
        pushd "$src"/workdir || exit; \
        if [ -f "$startDate" ]; then \
                echo "$src / $startDate already processed."
                popd || exit; \
                continue; \
        fi; \
	echo "Running work for ../../pars/*/${src}.par"
        bash ../../scripts/work.sh ../../pars/*/"$src".par; \
        for scr in *.sh; do bash "$scr"; done; \
        bash ../../scripts/pat.sh; \
        bash ../../scripts/cleanup.sh; \
        touch "$startDate"
	popd || exit;
        pushd "$src"/TOAs; \
        cat ./*corrected.tim > merged_"$dt".tim; \
        popd || exit; \
done

find ./J*/workdir/ -name "$startDate" -delete
find ./J*/ -type l -delete

rm -r tmpLink
