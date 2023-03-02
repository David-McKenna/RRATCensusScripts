import matplotlib.pyplot as plt
import numpy as np
import os
import pandas
import pickle
import sigpyproc as spp

from collections import defaultdict
from datetime import timedelta
from scipy import optimize as opt

from genericFuncs import powerl, tempSky, pointing, lofar_tinst_range, get_lofar_aeff_max, cachedAxisLookup, generateJonesCorrection, calculateBrightness_periodic


# Inclusive paz zaps, set ranges are the only non-zapped chans
nameChannelMapping = {
        '186MHz_176MHz': (473, 879),
        '176MHz_166MHz': (881, 1287),
        '166MHz_156MHz': (1289, 1695),
        '156MHz_146MHz': (1697, 2103),
        '146MHz_136MHz': (2105, 2511),
        '136MHz_126MHz': (2511, 2919),
        '126MHz_116MHz': (2921, 3327),
}

fch1 = lambda x: (100 + (100 / 512) * (x + 0.5))
obsFch1 = 499
chanFactor = 8
bandwidth = -1 * (100 / 512 / chanFactor)
chanToFreq = lambda chanL, chanU, bw: (fch1(obsFch1) + (chanL - 0.5) * bw, fch1(obsFch1) + (chanU + 0.5) * bw)
nameFrequencyMapping = {
        '186MHz_176MHz': chanToFreq(473, 879, bandwidth),
        '176MHz_166MHz': chanToFreq(881, 1287, bandwidth),
        '166MHz_156MHz': chanToFreq(1289, 1695, bandwidth),
        '156MHz_146MHz': chanToFreq(1697, 2103, bandwidth),
        '146MHz_136MHz': chanToFreq(2105, 2511, bandwidth),
        '136MHz_126MHz': chanToFreq(2511, 2919, bandwidth),
        '126MHz_116MHz': chanToFreq(2921, 3327, bandwidth),
}
bins = 256

def main():
	currentSource = os.path.basename(os.getcwd())
	files = defaultdict(dict)

	basedir = "./folds/"
	combined = defaultdict(dict)
	processedTime = 0
	for fil in os.listdir(basedir):
		prefix = fil.replace('.ar', '').replace('.clfd', '')
		fil = basedir + fil
		if fil.endswith('.fit') and not "10sL" in fil:
			if 'merged' in fil and not '.it' in fil:
				continue
			with open(fil, 'r') as ref:
				data = ref.readlines()
				
				snrData = [line for line in data if 'Best S/N' in line][0].strip('\n').split()
				snrVal = float(snrData[3])
				if snrVal < 0:
					snrVal = 0.

				periodData = [line for line in data if 'Best BC Period (ms) =' in line][0].split()
				periodVal = float(periodData[5]) / 1000.0

				widthData = [line for line in data if 'Pulse width' in line][0].strip('\n').split()
				widthVal = float(widthData[4])

				dmData = [line for line in data if 'Best DM' in line][0].strip('\n').split()
				dmVals = list(map(float, (dmData[3], dmData[9])))
				if 'merged' in fil:
					if 'block' in prefix and '.it' in fil:
						combined[prefix.replace('.fit', '').replace('.it', '').replace('.block_', '').split('_merged')[-1]]['snrs'] = (snrVal, periodVal, widthVal, dmVals)
					else:
						combined[prefix.replace('.fit', '').split('_merged')[0]]['snrs'] = (snrVal, periodVal, widthVal, dmVals)
				else:
					files[prefix.replace('.fit', '')]['snrs'] = (snrVal, periodVal, widthVal, dmVals)
		
		elif fil.endswith('_clfd_report.h5'):
			with pandas.HDFStore(fil, mode='r') as store:
				profmask = pandas.read_hdf(store, 'profmask').values
			flagged = profmask.mean(axis = 0)
			files[prefix.replace('_clfd_report.h5', '')]['zapped'] = flagged
		
		elif fil.endswith('.meta'):
			with open(fil, 'r') as ref:
				data = tuple(map(float, ref.readlines()[1].split()[1:]))
				#print(data)
			if 'merged' in fil:
				if 'block' in prefix and 'it' in fil:
					combined[prefix.replace('.meta', '').replace('.it', '').replace('.block_', '').split('_merged')[-1]]['time'] = data
				else:
					combined[prefix.replace('.meta', '').split('_merged')[0]]['time'] = data

			else:
				files[prefix.replace('.meta', '')]['time'] = data
				processedTime += data[0]
		else:
			#print(f"No actions found for {fil}")
			continue
	#print(combined.items())
	subbands = {key: tuple(int((np.mean(f) - 100) / (100 / 512)) for f in freqs) for key, freqs in nameFrequencyMapping.items()}
	#print(subbands.items())
	srcCoord = pointing(currentSource)
	pnting = [srcCoord.ra.rad, srcCoord.dec.rad, "J2000"]
	totalTime = 0
	#print(list(files.keys()))
	for obs, values in files.items():
		if 'time' not in values:
			raise RuntimeError(f"ERROR: Observation length missing for {obs}")
		if 'zapped' not in values:
			raise RuntimeError(f"ERROR: Observation flagging missing for {obs}")
		length, mjd = values['time']
		totalTime += length
		for key, sbbs in subbands.items():

			chans = nameChannelMapping[key]
			jonesCorrection = generateJonesCorrection(list(sbbs), Time(mjd, format = 'mjd'), tuple(pnting), dur = timedelta(seconds = length), integ = timedelta(seconds = min(length, 300.0)), antennaSet = 'HBA', stn = 'IE613', mdl = 'Hamaker-default', meanArray = True) * length

			if 'jones' not in combined[key]:
				combined[key]['jones'] = np.zeros_like(jonesCorrection)
				combined[key]['zapped'] = np.zeros_like(values['zapped'][chans[0]:chans[1] + 1])
				combined[key]['time'] = 0.

			combined[key]['jones'] += jonesCorrection
			combined[key]['zapped'] += values['zapped'][chans[0]:chans[1] + 1] * length
			combined[key]['time'] += length

	#print(combined.items())
	for key in combined:
		if 'MHz' not in key or 'block' in key:
			continue
		combined[key]['jones'] /= combined[key]['time']
		combined[key]['zapped'] /= combined[key]['time']

	mergedFull = [key for key in combined if 'MHz' not in key]
	if len(mergedFull) > 1:
		raise RuntimeError(f"ERROR: More than 1 candidate for combined fold detected: {mergedFull}")
	mergedFull = mergedFull[0]
	#print(mergedFull, combined[mergedFull])
	
	totalLength = combined[mergedFull]['time'][0]
	if abs(totalLength - processedTime) > 1e-3 * totalLength:
		if not os.path.exists(os.path.join(basedir, "NULLING")):
			raise RuntimeError(f"ERROR: Processed time and summed time differ significantly: {totalLength} vs {processedTime}")
		else:
			for key in combined:
				combined[key]['time'] = totalLength

	results = {}
	results_full = {}
	for key, values in combined.items():
		if 'MHz' not in key:
			continue

		if 'zapped' not in values.keys():
				print(f"ERROR: Unable to find zap data for {key}, continuing.")
				continue
		else:
			flaggedFraction = values['zapped']
		
		freqs = [nameFrequencyMapping[key]]
		flaggedFraction = values['zapped'].mean()
		fracBand = np.abs(np.diff(freqs))

		tinst = np.array(lofar_tinst_range(freqs))
		aeff = get_lofar_aeff_max(freqs)
		

		jonesCorrection = values['jones']
		tsky = tempSky(currentSource, freqs)


		snrs, period, widthBins, dm = values['snrs']
		width = (widthBins / bins) * period
		tobs = values['time']
		#print(tobs)

		#print(f"snrs = {snrs}, aeff = {aeff}, jonesCorrection = {jonesCorrection}, tinst = {tinst}, tsky = {tsky}, tobs = {tobs}, width = {width}, period = {period}, bandwidth = fracBand = {fracBand}, rfiflagged = flaggedFraction = {flaggedFraction}")
		brightness = calculateBrightness_periodic(snrs, aeff, jonesCorrection, tinst, tsky, tobs, width, period, bandwidth = fracBand, rfiflagged = flaggedFraction)

		results[key] = (brightness, snrs)
		results_full[key] = (brightness, (period, width, widthBins, bins, dm), (snrs, aeff, jonesCorrection, tinst, tsky, tobs, fracBand, flaggedFraction))

	#for key in sorted(results.keys()):
	#	print(key, results[key])

	with open("./results_periodic.pkl", 'wb') as ref:
		pickle.dump(results, ref)

	with open("./results_periodic_full.pkl", 'wb') as ref:
		pickle.dump(results_full, ref)


if __name__ == '__main__':
	main()

