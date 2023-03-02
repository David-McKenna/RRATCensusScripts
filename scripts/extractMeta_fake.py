import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

from astropy.table import Table
from copy import deepcopy
import astropy.units as u
import cdspyreadme


from genericFuncs import powerl, tempSky, pointing, lofar_tinst_range, fracBand, get_lofar_aeff_max, calculateBrightness, calculateBrightness_periodic

def main(sampleTime = 655.36e-6, periodicTime = 3600., additionalSources = []):
	## min, median, max for detected sources, need to expand to non-detected sources too
	highlightSources = [
		# Minimum K
		"J0939+45",
		# Median K
		"J2202+2147",
		# Maximum K
		"J2005+38",
	] + additionalSources

	singlePulseWidths = np.array([1, 2, 4, 16, 64, 256]) * sampleTime
	dutyCycles = np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3])

	result = []
	resultsPulse = np.zeros((len(highlightSources), singlePulseWidths.size)).T
	resultsPer = np.zeros((len(highlightSources), dutyCycles.size)).T

	for srcNum, src in enumerate(highlightSources):
		freqs = [(195.983891, 186.02297900000002), (186.02297900000002, 176.062067), (176.062067, 166.101155), (166.101155, 156.140243), (156.140243, 146.179331), (146.179331, 136.218419), (136.218419, 126.257507), (126.257507, 116.29659500000001), (117.859091, 107.898179)]
		freqs = freqs[1:-1]
		tinst = np.array(lofar_tinst_range(freqs))
		#plt.plot(freqs, tinst)
		#plt.show()
		aeff = get_lofar_aeff_max(freqs)
		subbands = np.array([int((np.mean(f) - 100) / (100 / 512)) for f in freqs])

		fracBand = abs(freqs[1][1] - freqs[1][0])
		tsky = tempSky(src, freqs)
		print(src, tsky[4])


		jonesCorrection = 1.
		flaggedFraction = 0.1

		snrs = 7.5
		res = []
		print(np.mean(tsky))
		for idx, tobs in enumerate(singlePulseWidths):
			brightness = calculateBrightness(8, np.mean(aeff), np.mean(jonesCorrection), np.mean(tinst), np.mean(tsky), tobs, bandwidth = freqs[0][0] - freqs[-1][1], rfiflagged = flaggedFraction)
			#brightness = calculateBrightness(snrs, aeff, jonesCorrection, tinst, tsky, tobs, bandwidth = fracBand, rfiflagged = flaggedFraction)
			print(tobs, snrs)
			print(brightness)
			print(np.mean(brightness))
			print()
			#res.append((tobs, brightness[-1], np.mean(brightness)))
			res.append((tobs, brightness))
			resultsPulse[idx, srcNum] = brightness

		snrs = 6
		timefrac = (1 - dutyCycles) / dutyCycles
		time = timefrac * periodicTime
		for idx, tobs in enumerate(time):
			brightness = calculateBrightness(snrs, np.mean(aeff), np.mean(jonesCorrection), np.mean(tinst), np.mean(tsky), tobs, bandwidth = freqs[0][0] - freqs[-1][1], rfiflagged = flaggedFraction)
			#print(tobs, snrs)
			#print(brightness)
			#print(np.mean(brightness))
			#print()
			res.append((tobs, brightness * 1000)) # Jy -> mJy
			resultsPer[idx, srcNum] = brightness * 1000 # Jy -> mJy

		result.append((src, np.vstack(res)))

	temps = [tempSky(src, [(150.0, 150.0), (150.0, 150.0)])[0] for src in highlightSources]

	columnUnitsPulse = [
		("Width", u.second, float, "Pulse width in seconds"),
		("S_Tsky_min", u.Jy, float, f"Sensitivity limit at the minimum observed sky temperature (v = 150MHz -> {int(temps[0])} K)"),
		("S_Tsky_med", u.Jy, float, f"Sensitivity limit at the median observed sky temperature (v = 150MHz -> {int(temps[1])} K)"),
		("S_Tsky_max", u.Jy, float, f"Sensitivity limit at the maximum observed sky temperature (v = 150MHz -> {int(temps[2])} K)"),
	]

	columnsUnitsPer = [
		("Duty Cycle", None, float, "Source duty cycle (fractional)"),
		("S_Tsky_min", u.Jy / 1000, float, f"Sensitivity limit at the minimum observed sky temperature (v = 150MHz -> {int(temps[0])} K)"),
		("S_Tsky_med", u.Jy / 1000, float, f"Sensitivity limit at the median observed sky temperature (v = 150MHz -> {int(temps[1])} K)"),
		("S_Tsky_max", u.Jy / 1000, float, f"Sensitivity limit at the maximum observed sky temperature (v = 150MHz -> {int(temps[2])} K)"),
	]

	print(resultsPulse.shape, len(columnUnitsPulse), [len([row[i] for row in columnUnitsPulse]) for i in range(4)])
	print(singlePulseWidths[:, np.newaxis].shape, resultsPulse.shape)

	tablePulse = Table(np.hstack([singlePulseWidths[:, np.newaxis], resultsPulse]), names = [row[0] for row in columnUnitsPulse], units = [row[1] for row in columnUnitsPulse], dtype = [row[2] for row in columnUnitsPulse], descriptions = [row[3] for row in columnUnitsPulse])
	tablePer = Table(np.hstack([dutyCycles[:, np.newaxis], resultsPer]), names = [row[0] for row in columnsUnitsPer], units = [row[1] for row in columnsUnitsPer], dtype = [row[2] for row in columnsUnitsPer], descriptions = [row[3] for row in columnsUnitsPer])

	maker = cdspyreadme.CDSTablesMaker()
	tabPul = maker.addTable(tablePulse, name = "SinglePulseSensitivityLimits")
	tabPer = maker.addTable(tablePer, name = "PeriodicSensitivityLimits")
	for idx, entry in enumerate(columnUnitsPulse):
		if entry[1] == u.Jy:
			tabPul.get_column(entry[0]).set_format("F4.2")
			tabPer.get_column(entry[0]).set_format("F4.2")
	maker.writeCDSTables()
	maker.makeReadMe()
	with open('sensitivityLimits.readme', 'w') as ref:
	    maker.makeReadMe(out = ref)
	with open("sensitivityLimits.pkl", 'wb') as ref:
	    pickle.dump([tablePulse, tablePer], ref)
	#for fmt in ['csv', 'votable', 'fits', 'ascii', 'ascii.cds', 'ascii.daophot', 'ascii.mrt']:
	fmt = 'votable'
	tablePulse.write('sensitivityLimitsPulse.' + fmt, overwrite=True, format = fmt) 
	tablePer.write('sensitivityLimitsPer.' + fmt, overwrite=True, format = fmt) 


	resultsList = [[f"{int(time //sampleTime) if time %sampleTime < 1e-6 else time}", f"{time * 1000:.3g}"] + [f"{ee:.2g}" for ee in e] for time, e in zip(singlePulseWidths, resultsPulse)]
	for row in resultsList:
		print(f"{' & '.join(row)} \\\\")

	print()

	resultsList = [[f"{time}"] + [f"{ee:.2g}" for ee in e] for time, e in zip(dutyCycles, resultsPer)]
	for row in resultsList:
		print(f"{' & '.join(row)} \\\\")

	#for src, res in result:
	#	print(src)
	#	for r in res:
	#		print(r)
	#
	#	print()
	#	print()

if __name__ == '__main__':
	main()

