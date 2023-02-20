inputPath="${1}"

# Call getNewCands.sh from the root working directory
# The input path expects a directory of directories that rougly follow
# 	./${sourceName}_..._cands
# Where ... does not have an _full suffix.
#
# Each candidates file in the directories are then parsed to filter candidates
# using the 'awk.part' file in each of the ${sourceName}/cands/ subdirectories
# of this directory.
#
#
# After filtering the resulting candidates (plots are generated for
# visual inspection in the cands/ subdirs), an ephemeris with the source name
# in ./pars/*/ is used to determine the amount of time to extract from the
# filterbank, and then dedisperse and resample the data with DSPSR and PSRCHIVE
# tools.
#
#
# A bug somewhere between DSPSR and PSRCHIVE whereby the start time is corrupted
# is worked around in the output TOAs using the true start time of the filterbank
bash scripts/getNewCands.sh ${inputPath}
