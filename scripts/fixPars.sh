#!/bin/env bash

for src in ./pars/*/*.par; do \
	cat "${src}" | grep -v "START" | grep -v "FINISH" > "${src/.par/_tmp.par}";
	mv "${src/.par/_tmp.par}" "${src}";
done
