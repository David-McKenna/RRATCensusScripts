import os, pickle, shutil
from lmfit.models import PowerLawModel, LognormalModel, GaussianModel, Model
from lmfit import Parameters
import numpy as np
import sigpyproc as spp

import matplotlib
#matplotlib.use("TkAgg")
matplotlib.use("Agg")
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.weight'] = 'bold'
fontsize = 48
matplotlib.rcParams['font.size'] = fontsize


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter, NullFormatter, FormatStrFormatter
plt.viridis()


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

fch1 = (100 + (100 / 512) * 499.5)
chanFactor = 8
bandwidth = -1 * (100 / 512 / chanFactor)
chanToFreq = lambda chanL, chanU, bw: (fch1 + (chanL - 0.5) * bw, fch1 + (chanU + 0.5) * bw)
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
	###
	# Prep work
	###

	# Setup objects to store our results
	overallResults = {}
	rawResults = {}
	srcs = os.listdir()
	srcs.sort(reverse = True)

	spectralIndexOutput = "./spectralIndexPeriodic/"
	if not os.path.exists(spectralIndexOutput):
		os.makedirs(spectralIndexOutput)

	pulseWidthOutput = "./pulseWidthPeriodic/"
	if not os.path.exists(pulseWidthOutput):
		os.makedirs(pulseWidthOutput)


	paperOutputs = "./paper/"
	for sub in ['./spectralPeriodic/']:
		if not os.path.exists(os.path.join(paperOutputs, sub)):
			os.makedirs(os.path.join(paperOutputs, sub))

	# For each item in the current directory
	for src in srcs:
		#print(src)
		# We only want to get soruce folders, identify them by the presence of a result_full pickle file and the fact they are folders starting eith J
		if (src[0] != 'J') or (os.path.isfile(f"./{src}")) or not os.path.exists(f"./{src}/results_periodic_full.pkl"):
			print(f"No work for {src}")
			continue
		
		# Load the data for the source
		print(src)
		with open(f"./{src}/results_periodic_full.pkl", 'rb') as ref:
			results = pickle.load(ref)

		fit = []
		widths = []
		snr = []
		freqs = []
		# For each pulse,
		for key in reversed(sorted(results.keys())):
			fit.append(np.ravel(results[key][0])[0])
			widths.append(results[key][1][1])
			snr.append(np.ravel(results[key][2][0])[0])
			freqs.append(np.mean(nameFrequencyMapping[key]))
		widths = np.array(widths)
		fit = np.array(fit)
		snr = np.array(snr)
		freqs = np.array(freqs)

		print("fit", fit)
		print("snr", snr)
		print("width", widths)

		# Build a power law model based on the pulse and store the results in the fit dict
		model = PowerLawModel()
		params = model.guess(fit, x = freqs)
		stderr = snr
		result = model.fit(fit, params, x = freqs, weights = (stderr))
		overallResults[src] = {'spectral': ((result.params['exponent'].value, result.params['exponent'].stderr), (result.params['amplitude'].value, result.params['amplitude'].stderr))}
		print(src, result.params['exponent'])
		result.plot_fit()
		plt.savefig(f"spectralIndexPeriodic/{src}.pdf")
		plt.close()

		# Save the results to overall/raw dicts

		meanW, stdW = (np.mean(widths), np.std(widths))
		#print(resultW)
		with open(f"./{src}/smeanPeriodic.txt", 'w+') as ref:
			ref.write(f"{fit.mean()}")
		with open(f"./{src}/spectralPeriodic.txt", 'w+') as ref:
			ref.write(f"{result.params['exponent'].value} {result.params['exponent'].stderr}")
		with open(f"./{src}/widthPeriodic.txt", 'w+') as ref:
			ref.write(f"{meanW} {stdW}")# \n{resultW.params['center'].value} {resultW.params['center'].stderr}")

		print(f"Width: {meanW}, {stdW}")


	# Save all of the results to disk
	with open("powerLawFits_periodic.pkl", 'wb') as ref:
		pickle.dump(overallResults, ref)


if __name__ == '__main__':
	main()