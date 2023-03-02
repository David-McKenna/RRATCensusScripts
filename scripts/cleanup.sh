#!/bin/env bash

mv ./*tim ./*vap ../TOAs/
mv ./*ar ../pulses/
mv ./*fil ../fils/
mv ./*sh ../scripts/
find ../fils/ -type l -name "J*P000.fil" -delete
