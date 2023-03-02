import argparse
import os
import tqdm

from psrqpy import Pulsar
import sigpyproc as spp

# Zapping imports
import numpy as np
from numpy.lib import stride_tricks
from itertools import groupby
from operator import itemgetter
import warnings
warnings.filterwarnings('ignore')

from genericFuncs import rfiZapping, grouping, parseCandidate


def main(args):

	# Sanity check parameters, expand paths to absolute paths
	if not os.path.exists(args.filterbank):
		raise RuntimeError(f"Unable to find filterbank at {args.filterbank}, exiting.")
	else:
		args.filterbank = os.path.abspath(args.filterbank)

	if not args.pulsar:
		args.pulsar = os.path.basename(args.filterbank).split('_')[0]

	if (args.ephemeris is not None) and not os.path.exists(args.ephemeris):
		print(f"Provided ephemeris {args.ephemeris} does not exist, attempting to continue...")
		args.ephemeris = None
	elif (args.ephemeris is not None):
		args.ephemeris = os.path.abspath(args.ephemeris)

	if (args.cand or args.candfile) and args.candtime:
		raise RuntimeError("Please only provide either a set of TOAs or a set of candidates, exiting.")

	if not (args.cand or args.candfile or args.candtime):
		raise RuntimeError("Please provide at least one source of TOAs to process, exiting.")

	if not (args.ephemeris or args.pulsar or (args.period and args.dm)):
		raise RuntimeError("Please provide ether a source name, ephemeris or DM/Period to process, exiting.")


	# Get information on the source
	if args.ephemeris:
		with open(args.ephemeris, 'r') as ref:
			eph = ref.readlines()
			period = [line for line in eph if 'P0' in line]
			if len(period) == 0:
				lines = [line for line in eph if 'F0' in line]
				if len(lines) == 0:
					period = 3.
				else:
					period = 1. / float([line for line in eph if 'F0' in line][0].split()[1])
			else:
				period = float(period[0].split()[1])

			dm = float([line for line in eph if ('DM' in line) and ('DM1' not in line) and ('DMEPOCH' not in line)][0].split()[1])

	elif (args.period and args.dm):
		period, dm = args.period, args.dm
	elif args.pulsar:
		try:
			pulsar = Pulsar(args.pulsar)
			period, dm = pulsar['DM'], pulsar['DM0']
		except Exception:
			raise RuntimeError(f"Failed to find {args.pulsar} in psrcat, exiting.")

	fixWidths = {'J2108+45': 4}
	if args.pulsar in fixWidths:
		print(f"{args.pulsar} requires period patching; {period}*{fixWidths[args.pulsar]}.")
		period *= fixWidths[args.pulsar]
	else:
		print(f"{args.pulsar}: No period patching needed")

	# Extract the TOAs we are going to be working with
	candidates = []
	if args.candtime is not None:
		candidates = args.candtime
	
	reader = spp.FilReader(args.filterbank)

	if args.cand:
		candidates.append(parseCandidate(args.cand)[2])
	if args.candfile:
		with open(args.candfile, 'r') as ref:
			candidates = candidates + [cand[1] * reader.header.tsamp for cand in parseCandidate(ref.readlines())]


	print(reader.header.tsamp)
	bins = int((1<<int(period / reader.header.tsamp).bit_length()) / 2)
	period = bins * reader.header.tsamp

	periodOffset = args.padding * period
	delayOffset = reader.header.getDMdelays(dm)[-1] * reader.header.tsamp
	extractLen = periodOffset * 2 + delayOffset + reader.header.tsamp * 0.9 # Add most of an extra sample to account for digifil sometimes dropping a sample
	extractSamples = int(extractLen / reader.header.tsamp)

	phaseRoll = 1 -(reader.header.getDMdelays(dm)[reader.header.nchans // 2] * reader.header.tsamp / (period))

	print(bins, period, periodOffset, extractLen, extractSamples)


	digifilCmd = []
	dspsrCmd = []
	zapCmd = []

#	if args.ephemeris:
#		ephemeris = f"-E {args.ephemeris}"
	if args.ephemeris or not args.pulsar:
		ephemeris = f"-D {dm} -c {period}"
	else:
		ephemeris = ""

	for cand in tqdm.tqdm(candidates):
		delayStr = f"-S {cand - periodOffset} -T {extractLen}"
		filterbank = args.filterbank

		if args.digifil:
			filterbank = os.path.join('/'.join(args.outfile.split('/')[:-1]),  args.filterbank.split('/')[-1].replace('.fil', f'_pulseTOA{cand}_8bit.fil'))
			digifilCmd.append(f"digifil {delayStr} -b 8 {args.filterbank} -o {filterbank}\n")
			delayStr = ""
			dspsrOut = filterbank.replace('.fil', '')
		else:
			dspsrOut = args.filterbank.replace('.fil', f'_pulseTOA_{cand:04d}_8bit')

		tstart = reader.readBlock(int((cand - periodOffset) / reader.header.tsamp), 1).header.tstart
		dspsrCmd.append(f"dspsr -cepoch {tstart} -p {phaseRoll} {delayStr} {ephemeris} {args.dspsrargs} -T {extractLen} -b {bins} {filterbank} -O {dspsrOut}\n")
#		dspsrCmd.append(f"dspsr {delayStr} {ephemeris} {args.dspsrargs} -T {extractLen} -b {int((2 * periodOffset) // reader.header.tsamp)} {filterbank} -O {dspsrOut}\n")
		dspsrOut += ".ar"

		if args.zap:
			bandpass = reader.readBlock(int((cand - periodOffset) / reader.header.tsamp), extractSamples).get_bandpass()
			zapStr = rfiZapping(bandpass, zappedChans = np.arange(3700, 3904), gui = False, chanGroup = 8, windowSize = 16, std = 3., bandpassPassed = True).replace("-zap_chans ", '" -Z "')[2:] + '"'
			zapCmd.append(f"paz -e zap.ar {dspsrOut} {zapStr}\n")



	for line in digifilCmd:
		print(line)
	for line in dspsrCmd:
		print(line)
	for line in zapCmd:
		print(line)

	if args.outfile is not None:
		with open(args.outfile, 'a+') as ref:

			if args.digifil:
				ref.writelines(reversed(digifilCmd))
				ref.writelines(['\n\n'])

			ref.writelines(reversed(dspsrCmd))

			if args.zap:
				ref.writelines(['\n\n'])
				ref.writelines(reversed(zapCmd))
		print(f"Written {len(dspsrCmd)} TOA extractions to {args.outfile}.")


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = "Generate digifil and dspsr commands to extract a pulse from a filterbank.")

	parser.add_argument("-c", dest = "cand", default = None, type = str, help = "Single candidate to handle")#
	parser.add_argument("-f", dest = "candfile", default = None, type = str, help = "File of candidates to handle")#
	parser.add_argument("-t", dest = "candtime", default = None, nargs = '?', type = float, help = "Times of candidates to handle (requires DM or ephemeris)")#

	parser.add_argument("-i", dest = "filterbank", required = True, type = str, help = "Filterbank to extract data from")#
	parser.add_argument("-8", dest = "digifil", default = True, action = 'store_false', help = "Disable digifil command output (for 8-bit data case)")#
	parser.add_argument("-z", dest = "zap", default = True, action = 'store_false', help = "Disable paz zap commands")

	parser.add_argument("-p", dest = "padding", default = 1., type = float, help = "Fraction of pulse length to pad before/after TOA")#
	parser.add_argument("-b", dest = "binMul", default = 4, type = int, help = "Target bins per time sample")#
	parser.add_argument("-a", dest = "dspsrargs", default = "-skz -K -k Ielfrhba", type = str, help = "Extra arguments to pass to DSPSR")#
#	parser.add_argument("-a", dest = "dspsrargs", default = "-skz -K -k Ielfrhba -turns 1 -nsub 32", type = str, help = "Extra arguments to pass to DSPSR")#

	parser.add_argument("-E", dest = "ephemeris", default = None, type = str, help = "Location of ephemeris to fold with")#
	parser.add_argument("-P", dest = "pulsar", default = None, type = str, help = "Pulsar to lookup in psrcat")#
	parser.add_argument("-r", dest = "period", default = 1.0, type = float, help = "Period to use for calculations")#
	parser.add_argument("-d", dest = "dm", default = None, type = float, help = "DM to use for calculations")#

	parser.add_argument("-o", dest = "outfile", default = None, type = str, help = "Write commands to given file location")


	main(parser.parse_args())




