import os
import numpy as np
import sigpyproc as spp
from scipy import optimize as opt
import pickle
from dreambeam.rime.scenarios import on_pointing_axis_tracking
from datetime import timedelta
from astropy.time import Time
import matplotlib.pyplot as plt


from genericFuncs import powerl, tempSky, pointing, lofar_tinst_range, fracBand, get_lofar_aeff_max, calculateBrightness, generateJonesCorrection


def processFilterbank(data, lowerChan = 64, upperChannel = 3672, targetBandwidth = fracBand, channelisationFactor = 8, offpulseFrac = 0.33):
	channelsSplit = []
	freqs = []
	channelsPerSplit = abs(int(targetBandwidth / data.header.foff / channelisationFactor) * channelisationFactor)
	splits = abs(int((upperChannel - lowerChan) * data.header.foff / (channelsPerSplit * data.header.foff)))

	for i in range(splits):
		channelsSplit.append(data[i * channelsPerSplit + lowerChan: (i + 1) * channelsPerSplit + lowerChan])
		freqs.append(((i * channelsPerSplit + lowerChan) * data.header.foff, ((i + 1) * channelsPerSplit + lowerChan) * data.header.foff))

	if ((upperChannel - lowerChan) * data.header.foff / (channelsPerSplit * data.header.foff) % 1) != 0:
		channelsSplit.append(data[upperChannel - channelsPerSplit: upperChannel])
		freqs.append(((upperChannel - channelsPerSplit) * data.header.foff, upperChannel * data.header.foff))

	freqs = [(data.header.fch1 + freq[0], data.header.fch1 + freq[1]) for freq in freqs]

	# (channelsPerSplit, timesamples, splits (+ 1?))
	channelsSplit = np.dstack(channelsSplit)
	# Find the fraction of flagged channels in each block
	flaggedFraction = np.sum(np.isnan(np.sum(channelsSplit, axis = 1)), axis = 0) / channelsSplit.shape[0]
	# Sum across all the frequencies
	channelsSplit = np.nansum(channelsSplit, axis = 0)

	# Get the mean, std of each time series
	meanSplits = np.mean(np.vstack([channelsSplit[:int(data.shape[0] * offpulseFrac)], channelsSplit[-1 * int(data.shape[0] * offpulseFrac):]]), axis = 0)
	stdSplits = np.std(np.vstack([channelsSplit[:int(data.shape[0] * offpulseFrac)], channelsSplit[-1 * int(data.shape[0] * offpulseFrac):]]), axis = 0)

	normalisedSeries = (channelsSplit - meanSplits) / stdSplits
	return normalisedSeries, freqs, flaggedFraction

def main():
	files = {}
	for script in os.listdir('./scripts/'):
		if not script.endswith('.sh'):
			continue
		with open(f"./scripts/{script}", 'r') as ref:
			data = ref.readlines()
			if len(data) > 4:
				data = [line for line in data if 'paz ' in line or 'dspsr ' in line]
				files[script.replace('.sh', '')] = data

	fits = {}
	for fit in os.listdir('./pdmpDump/'):
		if not fit.endswith('.fit'):
			continue
		with open(f"./pdmpDump/{fit}", 'r') as ref:
			data = ref.readlines()
			
			snrData = [line for line in data if 'Best S/N' in line][0].strip('\n').split()
			snrVal = float(snrData[3])

			widthData = [line for line in data if 'Pulse width' in line][0].strip('\n').split()
			widthVal = float(widthData[4])

			dmData = [line for line in data if 'Best DM' in line][0].strip('\n').split()
			dmVals = list(map(float, (dmData[3], dmData[9])))

			fits[fit.replace('.fit', '')] = (snrVal, widthVal, dmVals)


	zaps = {}
	zapBase = np.zeros(3904, dtype = np.uint8)
	for __, lines in files.items():
		for line in lines:
			splitLine = line.split()
			if 'dspsr ' in line:
				dm = float(splitLine[splitLine.index('-D') + 1])
				prefix = splitLine[-1].replace("./", '')
	
				for line2 in lines:
					if 'paz' in line2 and prefix in line2:
						toZap = zapBase.copy()
						splitLine2 = line2.split(' -Z ')
						lims = []
						for element in splitLine2[1:]:
							limits = list(map(int, filter(None, element.replace("\"", '').split(' '))))
							lims.append(limits)
							toZap[limits[0]: limits[1]] = 1.
							
						zaps[prefix] = (dm, toZap.copy(), lims)

	cands = {}
	for cand in os.listdir("./cands/"):
		if not cand.endswith('.cand'):
			continue
		with open(f"./cands/{cand}", 'r') as ref:
			#print(ref.readlines())
			data = list([list(map(float, line.strip('\n').split())) for line in ref.readlines() if '#' not in line and len(line)])
			candWork = {}
			try:
				for c in data:
					dp = c[1]
					ds, de = c[-2] - dp, c[-1] - dp
					candWork[dp] = (ds, de, c[-1] - c[-2])
			except IndexError as e:
				print(f"Failed to process candidate for {cand} ({e})\n\n{data}\n\n ...continuing.")
			cands[cand.strip('.cand')] = candWork


	#zappy = list(zaps.keys())
	#zappy.sort()
	#for z in zappy:
	#	print(z)
	fils = os.listdir("./fils/")
	fils.sort()
	for fil in fils:
		if not fil.endswith(".fil"):
			continue
		if fil.strip('.fil') not in zaps.keys():
			for key in zaps.keys():
				if key.strip('_8bit') == fil[:len(key.strip('_8bit'))]:
					continue
			print(f"WARNING: Missing zap data for {fil}")
			zaps[key] = []

	if os.path.exists('./results.pkl'):
		with open('./results.pkl', 'rb') as ref:
			results = pickle.load(ref)
		with open('./results_full.pkl', 'rb') as ref:
			results_full = pickle.load(ref)
	else:
		results = {}
		results_full = {}
	with open("./dm.txt", 'r') as ref:
		dm = float(ref.readlines()[0].split()[0])
	for fil in fils:
		if not fil.endswith(".fil"):
			continue
		if fil in results.keys():
			continue

		sppReader = spp.FilReader(f"./fils/{fil}")
		obs = fil.split('_cDM')[0]
		src = fil.split('_')[0]
		prefix = fil.replace('.fil', '')

		dsData = sppReader.readBlock(0, sppReader.header.nsamples).dedisperse(dm, True).normalise()
		if prefix not in zaps.keys():
			for key in zaps.keys():
				if key.strip('_8bit') == fil[:len(key.strip('_8bit'))]:
					prefix = key
					break
			else:
				print(f"ERROR: Unable to find zap data for {key}, continuing.")
				continue
		# Edge case: no zap data (failed to parse? corrupted script?)
		if len(zaps[prefix]) > 2:
			for limit in zaps[prefix][2]:
				dsData[limit[0]:limit[1], :] = np.nan
		else:
			continue

		normalisedSeries, freqs, flaggedFraction = processFilterbank(dsData, targetBandwidth = fracBand)
		tinst = np.array(lofar_tinst_range(freqs))
		aeff = get_lofar_aeff_max(freqs)
		subbands = np.array([int((np.mean(f) - 100) / (100 / 512)) for f in freqs])
		jonesCorrection = generateJonesCorrection(subbands, Time(dsData.header.tstart + (dsData.header.tsamp * dsData.header.nsamples / 2) / 86400, format = 'mjd').datetime, (dsData.header.ra_rad, dsData.header.dec_rad, "J2000"))

		tsky = tempSky(src, freqs)

		intStart = int(float(fil.split('_pulseTOA')[1].split('_')[0]) / dsData.header.tsamp)
		try:
			startSample, endSample, tobs = list(map(int, cands[obs][intStart]))
		except KeyError:
			#print(f"Mismatch: {fil}")
			if not len(cands[obs]):
				print(f"ERROR: No candidtaes for {fil}, but there was a pulse. Continuing.")
				continue
			if intStart - 1 in cands[obs].keys():
				offst = -1
			elif intStart + 1 in cands[obs].keys():
				offst = 1
			else:
				keys = np.array(list(cands[obs].keys()))
				if ((keys - intStart) < 50).any():
					offst = np.min(keys - intStart)

				else:
					print(f"Unable to find key for {obs} time {int(float(fil.split('_pulseTOA')[1].split('_')[0]) / dsData.header.tsamp)} / {float(fil.split('_pulseTOA')[1].split('_')[0])} in keys (min val {(np.array(list(cands[obs].keys())) - intStart).min()})") #{cands[obs].keys()}")
					continue
			startSample, endSample, tobs = list(map(int, cands[obs][int(float(fil.split('_pulseTOA')[1].split('_')[0]) / dsData.header.tsamp) + offst]))

		windowLengths = np.arange(max(int((endSample - startSample) / 2), 2), max(int(1.5 * (endSample - startSample + 1)), 6))
		windows = [np.ones(size) / np.sqrt(size) for size in windowLengths]
		removeBuffer = int(dsData.shape[1] * 0.33)

		convolved = np.array([np.convolve(normalisedSeries[removeBuffer: -removeBuffer, :].sum(axis = 1), window, 'same') for window in windows])
		peakLoc = [np.max(winSer) for winSer in convolved]
		peakWinIdx = np.argmax(peakLoc)
		peakWindow = windowLengths[peakWinIdx]
		peakLoc = np.argmax(convolved[peakWinIdx]) + removeBuffer

		plt.plot(np.arange(removeBuffer, dsData.shape[1] - removeBuffer), normalisedSeries[removeBuffer:dsData.shape[1] - removeBuffer], alpha = 0.2)
		#plt.vlines([int(np.mean(startSample + peak)), int(np.mean(endSample + peak))], -3, +3, color = 'r')
		plt.vlines([int(np.mean(peakLoc - peakWindow // 2)), int(np.mean(peakLoc + peakWindow // 2))], -3, +3, color = 'r')
		plt.twinx().plot(np.arange(removeBuffer, dsData.shape[1] - removeBuffer), np.nansum(dsData.normalise(), axis = 0)[removeBuffer:dsData.shape[1] - removeBuffer], color = 'g', alpha = 0.5)
		plt.twinx().plot(np.arange(removeBuffer, dsData.shape[1] - removeBuffer), convolved[peakWinIdx], color = 'm', alpha = 0.5)
		plt.savefig('./snrPlots/' + fil.replace('.fil', '.png'))
		plt.close('all')

		#snrs = np.stack([np.sum(normalisedSeries[int(p - w // 2):int(p + w // 2), i]) for i, (p, w) in enumerate(zip(peak, peakWindow))]) / np.sqrt(peakWindow)
		snrs = np.sum(normalisedSeries[int(peakLoc - peakWindow // 2):int(peakLoc + peakWindow // 2)], axis = 0) / np.sqrt(peakWindow)
		tobs = peakWindow * dsData.header.tsamp
		#print(fracBand, flaggedFraction, jonesCorrection)
		brightness = calculateBrightness(snrs, aeff, jonesCorrection, tinst, tsky, tobs, bandwidth = fracBand, rfiflagged = flaggedFraction)

		results[fil] = brightness
		results_full[fil] = (brightness, (peakWindow, peakLoc, normalisedSeries), (snrs, aeff, jonesCorrection, tinst, tsky, tobs, fracBand, flaggedFraction))

	with open("./results.pkl", 'wb') as ref:
		pickle.dump(results, ref)

	with open("./results_full.pkl", 'wb') as ref:
		pickle.dump(results_full, ref)

if __name__ == '__main__':
	main()
