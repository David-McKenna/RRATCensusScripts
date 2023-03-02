import sigpyproc as spp
import numpy as np
import argparse
import tqdm
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from numpy.lib import stride_tricks
from itertools import groupby
from operator import itemgetter


import warnings
warnings.filterwarnings('ignore')

from genericFuncs import exploreFolderTree, extractHeimdallCand, grouping, rfiZapping





def filterCands(cands, minSnr = 0., minDM = 0., maxDM = 0., minWidth = 0, maxWidth = 0, minBoxcars = 0, maxBoxcars = 0):

	snrIdx = 0
	if (minSnr != 0.):
		cands = filter(lambda cand: (cand[snrIdx] >= minSnr), cands)

	widthIdx = 3
	if (minWidth != 0):
		cands = filter(lambda cand: (cand[widthIdx] >= minWidth), cands)
	if (maxWidth != 0):
		cands = filter(lambda cand: (cand[widthIdx] <= maxWidth), cands)

	dmIdx = 5
	if (minDM != 0.):
		cands = filter(lambda cand: (cand[dmIdx] >= minDM), cands)
	if (maxDM != 0.):
		cands = filter(lambda cand: (cand[dmIdx] <= maxDM), cands)


	boxcarIdx = 6
	if (minBoxcars != 0):
		cands = filter(lambda cand: (cand[boxcarIdx] >= minBoxcars), cands)
	if (maxBoxcars != 0):
		cands = filter(lambda cand: (cand[boxcarIdx] <= maxBoxcars), cands)

	return list(cands)

def rollingAverage(series, size):
	ds = np.cumsum(series)
	return ds[size:] - ds[:-size] 

def extractPulse(cand, sppReader, bufferMul = 2, mulType = 'fraction', dedisperse = True, plot = False, plotnorm = False, overrideDM = None, prefix = '', zap = True, toFile = True, zappy = []):
	startIdx, endIdx = cand[7], cand[8]
	pulseLength = endIdx - startIdx
	
	if mulType == 'fraction':
		bufferVal = int(bufferMul * pulseLength)
		startIdx -= bufferVal
		endIdx += bufferVal + sppReader.header.getDMdelays(overrideDM or cand[5])[-1]
	elif mulType == 'absolute':
		bufferVal = int(bufferMul / sppReader.header.tsamp)
		startIdx -= bufferVal
		endIdx += bufferVal + sppReader.header.getDMdelays(overrideDM or cand[5])[-1]
	else:
		raise RuntimeError("Unknown mulType ({mulType}), only acceping 'fraction' or 'abolute', exiting.")

	startIdx = max(startIdx, 0)
	endIdx = min(endIdx, sppReader.header.nsamples - 1)
	dataBlock = sppReader.readBlock(startIdx, endIdx - startIdx)

	if zap:
		chans, __ = rfiZapping(dataBlock, heimdallReturn = False)
		dataBlock[chans, :] = np.nan

	for chans in zappy:
		dataBlock[chans, :] = np.nan


	if dedisperse:
		dataBlock = dataBlock.dedisperse(overrideDM or cand[5], True)

	if plot or plotnorm:
		ffactor = 32
		if 'J1931+4229' in sppReader.header.filename and not 'top' in sppReader.header.filename:
			ffactor = 16
		plotBlock = dataBlock.downsample(tfactor = max(int(pulseLength // 8), 1),ffactor = ffactor)
		plotting = []
		if plot:
			plotting.append(('', plotBlock))
		if plotnorm:
			plotting.append(('normalise', plotBlock.normalise()))

		for suffix, plotBlock in plotting:
			fig = plt.figure(figsize = (18, 18))
			gs = fig.add_gridspec(10,5)
			axtop1 = fig.add_subplot(gs[0, :-1])
			axtop2 = fig.add_subplot(gs[1, :-1])			
			axright = fig.add_subplot(gs[2:, -1])
			axall = fig.add_subplot(gs[2:, :-1])
			axinfo = fig.add_subplot(gs[0, -1])

			dataTopRaw = np.nansum(dataBlock, axis = 0)
			dataTopNorm = np.nansum(dataBlock.normalise(), axis = 0)
			rawRollingAvg = rollingAverage(dataTopRaw, pulseLength)
			normRollingAvg = rollingAverage(dataTopNorm, pulseLength)

			quarterBand = int(dataBlock.header.nchans // 8)
			dataTopSplit = np.vstack([rollingAverage(np.nansum(dataBlock[i * quarterBand: (i + 1) * quarterBand, :].normalise(), axis = 0), pulseLength) for i in range(8)]).T
			dataTopSplit /= np.nanmax(dataTopSplit, axis = 0)[None, :]

			rollingXArr = np.arange(dataTopRaw.size)[int(pulseLength / 2): -1 * (dataTopRaw.size - rawRollingAvg.size - int(pulseLength / 2))]
			dataTopRaw /= np.max(dataTopRaw)
			dataTopNorm /= np.max(dataTopNorm)


			dataRight = dataBlock.get_bandpass()
			vmx, vmn = np.nanpercentile(plotBlock, (90, 16))
			
			axall.imshow(plotBlock, aspect = 'auto', vmax = vmx, vmin = vmn, interpolation = 'none', extent = (0, dataBlock.shape[1] * dataBlock.header.tsamp, dataBlock.header.fbottom, dataBlock.header.ftop))
			axall.set_xlabel("Time (s)")
			axall.set_ylabel("Frequency (MHz)")

			axtop1.plot(dataTopRaw, alpha = 0.4)
			axtop1.plot(dataTopNorm, alpha = 0.4, c = 'g')
			axtop1.twinx().plot(rollingXArr, rawRollingAvg, alpha = 0.66, c= 'r')
			axtop1.twinx().plot(rollingXArr, normRollingAvg, alpha = 0.66, c= 'm')
			ylim = axtop1.get_ylim()
			axtop1.vlines([bufferVal, bufferVal + pulseLength], 0., 1.5)
			axtop1.set_ylim(ylim)
			axtop1.set_xlim([0, dataTopRaw.size - 1])

			axtop2.plot(rollingXArr, dataTopSplit, alpha = 0.33)
			axtop2.set_xlim([0, dataTopRaw.size - 1])

			axright.plot(dataRight, np.arange(dataRight.size))
			axright.set_ylim([0, dataRight.size - 1])
			axright.set_xlim([np.nanmin(dataRight)* 0.95, np.nanpercentile(dataRight, 83) * 2])
			axright.invert_yaxis()

			axtop1.set_title(f"{prefix}\nSNR: {cand[0]}    DM: {overrideDM or cand[5]}    Width: {dataBlock.header.tsamp * 2 ** cand[3]}s    MJD: {dataBlock.header.tstart + dataBlock.header.tsamp * cand[7] / (60 * 60 * 24)}     Samples: {cand[7]}-{cand[8]} ({cand[8]-cand[7]})")
			axinfo.annotate(f"{sppReader.header.filename.split('/')[-1]}\n{cand[0]}\n{cand[2]}s\n{cand[3]}/{cand[6]}\n{overrideDM or cand[5]}\n{cand[7]}-{cand[8]}\n{cand[9]}", (0.5, 0.5), xycoords='axes fraction', va='center', ha='center')
			axinfo.axis('off')
			plt.tight_layout()
			plt.savefig(f"{prefix}-SNR{cand[0]}-DM{overrideDM or cand[5]}-ts{cand[7]}-{cand[8]}{suffix}.png")
		plt.close('all')
	if toFile:
		dataBlock.toFile(f"{prefix}-SNR{cand[0]}-DM{overrideDM or cand[5]}-ts{cand[7]}-{cand[8]}.fil")
	return startIdx, endIdx, bufferVal, dataBlock.shape[1] - bufferVal



def main(args):
	args.forceflag = [np.arange(int(flags.split(':')[0]), int(flags.split(':')[1])) for flags in args.forceflag.split(',')]

	print(f"\nOpening Sigpyproc reader at {args.infile}")
	reader = spp.FilReader(args.infile)

	cands = []
	for file in [args.infolder]:
		cands = cands + extractHeimdallCand(file)

	candsCopy = cands.copy()
	cands = []
	for idx, cand in enumerate(candsCopy):
		cands.append(cand + [idx])

	
	numCands = len(cands)
	cands = filterCands(cands, minSnr = float(args.minSnr), minDM = float(args.dms[0]), maxDM = float(args.dms[1]), minWidth = int(args.width[0]), maxWidth = int(args.width[1]), minBoxcars = int(args.boxcars[0]), maxBoxcars = int(args.boxcars[1]))
	cands.sort(reverse = True, key = lambda x: x[0])

	print(f"\nFound {numCands} candidates. After filtering, we have {len(cands)} of interest. Begining processing...\n")

	prefixFolder = os.path.dirname(args.prefix)
	if not os.path.exists(prefixFolder):
		os.makedirs(prefixFolder)

	allseen = True
	for cand in cands:
		if os.path.exists(f"{args.prefix}-SNR{cand[0]}-DM{args.overrideDM or cand[5]}-ts{cand[7]}-{cand[8]}{'normalise' if args.plotnorm else ''}.png"):
			continue
		else:
			print(f"Missing {args.prefix}-SNR{cand[0]}-DM{args.overrideDM or cand[5]}-ts{cand[7]}-{cand[8]}.png")
			allseen = False
			break

	if allseen:
		print("All candidates have been plotted previously, exiting.")
		exit()

	pbar = tqdm.tqdm(cands)
	with open(f"{args.prefix}-pulseIndices.txt", 'a+') as fileRef:
		for cand in pbar:
			pbar.set_description(f"{cand[0]} @ {args.overrideDM or cand[5]}")
			filName = f"{args.prefix}-SNR{cand[0]}-DM{args.overrideDM or cand[5]}-ts{cand[7]}-{cand[8]}.fil"
			if args.overwrite or not (os.path.exists(filName) or os.path.exists(filName.replace('.fil', '.png'))):
				startIdx, endIdx, pulseStartIdx, pulseEndIdx = extractPulse(cand, reader, bufferMul = float(args.buffer), mulType = args.bufferType, dedisperse = args.dedisp, plot = args.plot, plotnorm = args.plotnorm, overrideDM = args.overrideDM, prefix = args.prefix, toFile = args.writeFil, zap = args.rfi, zappy = args.forceflag)
				fileRef.writelines([f"{filName} {startIdx} {endIdx} {pulseStartIdx} {pulseEndIdx}\n"])

	print(f"Finished extracting {len(cands)} pulses to {args.prefix}. Exiting.")

crabPulseLen = 0.034
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = "Extract pulses from a SigProc Filterbank based on Heimdall Pulse Candidates")

	parser.add_argument('-i', dest = 'infile', required = True, help = "Input filterbank location")
	parser.add_argument('-f', dest = 'infolder', required = True, help = "Input heimdall candidate file")
	parser.add_argument('-o', dest = 'prefix', default = 'out', help = "Output files prefix")
	parser.add_argument('-z', dest = 'dedisp', default = True, action = 'store_false', help = "Dedisperse pulse before ")
	parser.add_argument('-m', dest = "overrideDM", default = None, type = float, help = "Force dedisperse at a set DM")
	parser.add_argument('-p', dest = 'plot', default = False, action = 'store_true', help = "Plot pulses as well")
	parser.add_argument('-q', dest = 'plotnorm', default = False, action = 'store_true', help = "Normalise plot data")
	parser.add_argument('-b', dest = 'buffer', default = crabPulseLen, help = "Buffer before/after the pulse in the outputs with time in seconds or multiple of the pulse length")
	parser.add_argument('-t', dest = 'bufferType', default = 'absolute', action = 'store_const', const = 'fraction', help = "Swap the buffering from storing a constant amount of data to a fraction of the pulse length")
	parser.add_argument('-c', dest = 'checkPulse', default = True, action = 'store_false', help = "(Flag disables) If the pulse length if longer than the absolute buffer, expand it to the length of the pulse.")
	parser.add_argument('-s', dest = 'minSnr', default = 6., help = "Minimum SNR of a candidate to extract the pulse")
	parser.add_argument('-d', dest = 'dms', default = [0., 0.], nargs = 2, type = float, help = "Upper / lower limits of DMs to process")
	parser.add_argument('-w', dest = 'width', default = [0, 0], nargs = 2, type = int, help = "Log2 of the upper/lower bound of boxcar widths to process")
	parser.add_argument('-n', dest = 'boxcars', default = [0, 0], nargs = 2, type = int, help = "Upper / lower bound of the number of boxcars to process")
	parser.add_argument('-x', dest = 'overwrite', default = False, action = 'store_true', help = "Don't pass on a candidate if the file already exists")
	parser.add_argument('-k', dest = 'writeFil', default = True, action = 'store_false', help = "Don't save a copy of the data to a new filterbank")
	parser.add_argument('-r', dest = 'rfi', default = True, action = 'store_false', help = "Disable RFI flagging (recommended for digifil-d-8-bit data)")
	parser.add_argument('-l', dest = 'forceflag', default = '3650:3904', help = "Channels to always flag")

	main(parser.parse_args())