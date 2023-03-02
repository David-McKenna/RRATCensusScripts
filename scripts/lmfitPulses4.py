import os, pickle, shutil
from lmfit.models import PowerLawModel, LognormalModel, GaussianModel, Model
from lmfit import Parameters
import numpy as np
import sigpyproc as spp

from genericFuncs import stringWithErr

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

def stringWithErr(val, err, padding = 0):
        if isinstance(val, str) or isinstance(err, str):
                return val
        if val == 0 and err == 0:
                return '--'
        if err == 0 or type(err) == type(None):
                return f'{val:.2g}'

        length = -1 * int(np.floor(np.log10(err)))

        if (f"{err * 10 ** length:.0f}"[0] == '1'):
                length += 1

        errStr = f"{err * 10 ** length:.0f}"

        valStr = f"{val:0{max(padding + (int(str(float(val)).split('.')[0]) < 10) + length, 1)}.{max(length, 1)}f}"
        if val == err:
                return f"({valStr})"

        if len(errStr) == 2 and valStr[-2] == '.':
                errStr = f"{err * 10 ** (length - 1):2.1f}"

        return f"{valStr}({errStr})"
# Define as set of lmfit models
def brokenPowerLaw(x, sep, alp1, alp2, amp, returnBoth = False):
	# Broken power law -> 2 power laws of different dicies
	pwl1 = amp * x[x <= sep] ** alp1
	pwl2 = amp * (sep ** (alp1 - alp2)) * x[x > sep] ** alp2

	if returnBoth:
		return [amp * x ** alp1, amp * (sep ** (alp1 - alp2)) * x ** alp2]

	return np.hstack([pwl1, pwl2])
def brokenPowerLawPrep(data, x, **kws):
	params = Parameters()
	params.add_many(
		('sep', np.median(x), True, x.min(), x.max(), None, None),
		('alp1', 1, True, -np.inf, np.inf, None, None),
		('alp2', -1, True,  -np.inf, np.inf, None, None),
		('amp', 1., True, 0, np.inf, None, None),
	)
	return params

bPLModel = Model(brokenPowerLaw)
bPLModel.guess = brokenPowerLawPrep


def lognormalPrep(data, x, **kws):
	params = Parameters()
	params.add_many(
		('center', np.median(x), True, x.min(), x.max(), None, None),
		('amplitude', data.max() / 16, True, 0, data.size, None, None),
		('sigma', np.std(x) / 2, True, 0, x.max() - x.min(), None, None),
	)
	return params

lognormal = LognormalModel()
lognormal.guess = lognormalPrep

def LognormalPowerLaw(x, sep, A, mu, sigma, alp, amp, returnBoth = False):
	# Lognormal power law -> a log normal distribution followed by a power law tail
	lognorm = (A / (np.sqrt(2* np.pi) * sigma)) * np.exp(-np.square(np.log(x[x <= sep]) - mu)/(2 * sigma ** 2) ) / x[x <= sep]
	correction = (A / (np.sqrt(2* np.pi) * sigma)) * np.exp(-np.square(np.log(sep) - mu)/(2 * sigma ** 2) ) / sep
	pwl = amp* ((correction / amp) * sep ** -alp) * x[x > sep] ** alp

	if returnBoth:
		lognorm = (A / (np.sqrt(2* np.pi) * sigma)) * np.exp(-np.square(np.log(x) - mu)/(2 * sigma ** 2) ) / x
		correction = (A / (np.sqrt(2* np.pi) * sigma)) * np.exp(-np.square(np.log(sep) - mu)/(2 * sigma ** 2) ) / sep
		pwl = amp* ((correction / amp) * sep ** -alp) * x ** alp

		return [lognorm, pwl]

	return np.hstack([lognorm, pwl])

def PowerLawLognormal(x, sep, A, mu, sigma, alp, amp, returnBoth = False):
	# Power law LogNnormal -> power law followed by a lognormal distribution
	correction = (A / (np.sqrt(2* np.pi) * sigma)) * np.exp(-np.square(np.log(sep) - mu)/(2 * sigma ** 2) ) / sep
	pwl = amp* ((correction / amp) * sep ** -alp) * x[x <= sep] ** alp
	lognorm = (A / (np.sqrt(2* np.pi) * sigma)) * np.exp(-np.square(np.log(x[x > sep]) - mu)/(2 * sigma ** 2) ) / x[x > sep]

	#if np.isnan(np.hstack([pwl, lognorm])).any():
	#	print(correction, correction / amp, sep, alp, (sep ** -alp), x[x <= sep] ** alp)
	if returnBoth:
		correction = (A / (np.sqrt(2* np.pi) * sigma)) * np.exp(-np.square(np.log(sep) - mu)/(2 * sigma ** 2) ) / sep
		pwl = amp* ((correction / amp) * sep ** -alp) * x ** alp
		lognorm = (A / (np.sqrt(2* np.pi) * sigma)) * np.exp(-np.square(np.log(x) - mu)/(2 * sigma ** 2) ) / x

		return [pwl, lognorm]


	return np.hstack([pwl, lognorm])

def LognormalPowerLawPrep(data, x, **kws):
	params = Parameters()
	params.add_many(
		('sep', np.median(x), True, x.min(), x.max(), None, None),
		('alp', 1, True, -20, 20, None, None), # Sane limits, otherwise lmfit just tries to tend twards infinity for bad fits
		('amp', 1., True, 0, np.inf, None, None),
		('A', 1., True, 0, np.inf, None),
		('mu', 1., True, 0, np.inf, None),
		('sigma', 1., True, 0, np.inf, None),
	)
	return params
LNPLModel = Model(LognormalPowerLaw)
LNPLModel.guess = LognormalPowerLawPrep
PLLNModel = Model(PowerLawLognormal)
PLLNModel.guess = LognormalPowerLawPrep



def LogNormalSampling(x, A, mu, sigma):
	return (A / (np.sqrt(2* np.pi) * sigma)) * np.exp(-np.square(np.log(x) - mu)/(2 * sigma ** 2) ) / x


def LognormalLognormal(x, sep, A, A2, mu, mu2, sigma, sigma2):
	lognorm = (A / (np.sqrt(2* np.pi) * sigma)) * np.exp(-np.square(np.log(x) - mu)/(2 * sigma ** 2) ) / x
	lognorm2 = (A2 / (np.sqrt(2* np.pi) * sigma2)) * np.exp(-np.square(np.log(x) - mu2)/(2 * sigma2 ** 2) ) / x

	return lognorm + lognorm2

def LognormalLognormalPrep(data, x, **kws):
	params = Parameters()
	params.add_many(
		('sep', np.median(x), True, x.min(), x.max(), None, None),
		('A', 1., True, 0, np.inf, None),
		('mu', x.min(), True, 0, np.inf, None),
		('sigma', 1., True, 0, np.inf, None),
		('A2', 1., True, 0, np.inf, None),
		('mu2', x.max(), True, 0, np.inf, None),
		('sigma2', 1., True, 0, np.inf, None),
	)
	return params

LNLNModel = Model(LognormalLognormal)
LNLNModel.guess = LognormalLognormalPrep


def gaussian(x, params):
	amp = params['amplitude'].value
	std = params['sigma'].value
	mu = params['center'].value

	return (amp / (std* np.sqrt(2 * np.pi))) * np.exp(-1 * ((x - mu)** 2 / (2 * std ** 2)))

# Specify the indices of each pulse to sample
# 1:-1 -> drop the two samples on the edge of the band
startIdx = 1
endIdx = -1
def main():
	###
	# Prep work
	###

	# Load in the reference frequencies for all observation samples
	with open("./freqs.pkl", 'rb') as ref:
		freqs = pickle.load(ref)

	# Extract the mean frequencies in each block, ignoring specified values (should be the two edge samples)
	fmean = freqs['mean'][startIdx:endIdx]
	#print(fmean, freqs)
	# Setup objects to store our results
	overallResults = {}
	rawResults = {}
	srcs = os.listdir()
	srcs.sort(reverse = True)

	modIndexPath = "./modindex/"
	modIndexBestFitOutput = os.path.join(modIndexPath, "bestfit/")
	if not os.path.exists(modIndexBestFitOutput):
		os.makedirs(modIndexBestFitOutput)
	else:
		for fil in os.listdir(modIndexBestFitOutput):
			os.remove(os.path.join(modIndexBestFitOutput, fil))


	spectralIndexOutput = "./spectralIndex/"
	if not os.path.exists(spectralIndexOutput):
		os.makedirs(spectralIndexOutput)

	pulseWidthOutput = "./pulseWidth/"
	if not os.path.exists(pulseWidthOutput):
		os.makedirs(pulseWidthOutput)


	paperOutputs = "./paper/"
	for sub in ['./spectral/','./modulation/']:
		if not os.path.exists(os.path.join(paperOutputs, sub)):
			os.makedirs(os.path.join(paperOutputs, sub))

	# For each item in the current directory
	for src in srcs:
		#print(src)
		# We only want to get soruce folders, identify them by the presence of a result_full pickle file and the fact they are folders starting eith J
		if (src[0] != 'J') or (os.path.isfile(f"./{src}")) or not os.path.exists(f"./{src}/results_full.pkl"):
			continue
		
		# Load the data for the source
		print(src)
		with open(f"./{src}/results_full.pkl", 'rb') as ref:
			results = pickle.load(ref)


		fit = {}
		meanVal = []
		widths = []
		# For each pulse,
		for key, value in results.items():
			# Select all components of the pulse, in the specified sampling range, where the SNR > 0
			data = value[0][startIdx:endIdx]
			fwork = fmean[data > 0]
			if not fwork.size:
				continue
			sampledData = data[data > 0]
			meanVal.append(sampledData.mean())
			widths.append(value[2][5])
			# Build a power law model based on the pulse and store the results in the fit dict
			model = PowerLawModel()
			params = model.guess(sampledData, x = fwork)
			#stderr = np.std(value[1][2][:, startIdx:endIdx]) / (value[2][0][startIdx:endIdx] / np.mean(value[1][2][:, startIdx:endIdx]))
			stderr = value[2][0][startIdx: endIdx][data > 0]
			# Weights -> standard error of a single pulse flux density is 50\%
			#result = model.fit(data, params, x = fwork, weights = 1 / stderr)
			result = model.fit(sampledData, params, x = fwork, weights = 1 / stderr)
			if None in (result.params['exponent'].value, result.params['exponent'].stderr, result.params['amplitude'].value, result.params['amplitude'].stderr):
				continue
			fit[key] = [(result.params['exponent'].value, result.params['exponent'].stderr), (result.params['amplitude'].value, result.params['amplitude'].stderr)]
			#if abs(result.params['exponent'].value) > 6:
			#	print(key, result.params['exponent'], list(zip(sampledData, fwork)))
			#	print(value)
			#	result.plot_fit()
			#	plt.show()

		print(len(results.items()))
		if len(results.items()) == 0:
			print(f"ERROR: No results found for {src}, passing.")
			continue

		with open(f"./{src}/pulseCount.txt", 'w+') as ref:
			ref.writelines([f"{len(list(results.items()))}"])

		with open(f"./{src}/pulseBright.txt", 'w+') as ref:
			ref.writelines([f"{max(meanVal)} {max(meanVal) / min(meanVal)} {widths[meanVal.index(max(meanVal))]}"])

		# Perform some sampling on the resulting data
		rawData = np.array(list(fit.values())).reshape(-1, 4)
		meanPower= np.mean(rawData[:, 0])
		medianPower = np.median(rawData[:, 0])
		stdPower = np.sqrt(np.sum(np.square(rawData[:, 1]))) / rawData.shape[0]
		iqrstdPower = np.abs(np.diff(np.percentile(rawData[:, 1], (25, 75))))[0] / 1.35
		
		# Save the results to overall/raw dicts
		overallResults[src] = {'spectral': {'mean': meanPower, 'median': medianPower, 'std': stdPower, 'iqrstd': iqrstdPower}}
		rawResults[src] = {'spectral': list(fit.values())}

		if len(list(results.items())) > 4:
			plt.figure(figsize = (12, 12))
			valWs, binWs, patches = plt.hist(widths, bins = 'auto')
			widthModel = GaussianModel()
			binWs = (binWs[:-1] + 0.5 * (binWs[1] - binWs[0]))
			paramWs = widthModel.guess(valWs, x = binWs)
			resultWs = widthModel.fit(valWs, x = binWs, weights = 1 / np.sqrt(np.maximum(valWs, 1)))
			print(f"Width: {resultWs.params['center'].value}, {resultWs.params['center'].stderr}")
			resultWs.plot(numpoints = 128)
			plt.savefig(os.path.join(pulseWidthOutput, f"./{src}.png"))
			plt.close('all')
		else:
			print(f"Too few samples to calculate Gaussian width for {src}")

		meanW, stdW = (np.mean(widths), np.std(widths))
		with open(f"./{src}/width.txt", 'w+') as ref:
			ref.write(f"{meanW} {stdW}\n{resultWs.params['center'].value} {resultWs.params['center'].stderr}")

		print(f"Width: {meanW}, {stdW}")
		if len(list(results.items())) > 16:
			# If we have a sufficiently large sample of pulses....
			# 
			# Fit a Gaussian to their distribution of spectral indices to fit the underlying spectral index
			# Generate the histogram bins/values via matplotlib
			binG = np.histogram_bin_edges(rawData[:, 0], bins = 80, range = (-10, 10))
			valGs, binGTmp, patches = plt.hist(rawData[:, 0], bins = binG)
			# Centre the bins (previosly eachvalue represented the left edge of the bin)
			binGs = binGTmp[:-1] + 0.5 * (binGTmp[1] - binGTmp[0])

			# Build and fit a Gaussian distribution to the data
			gaussModel = GaussianModel()
			paramGs = gaussModel.guess(valGs, x = binGs)

			# Weights -> use sqrt(N_pulse) as the error of each bin, or 1 to prevent 1/0 infs
			resultGauss = gaussModel.fit(valGs, paramGs, x = binGs, weights = 1 / np.sqrt(np.maximum(valGs, 1)))
			resultGauss.plot()
			ylim = plt.ylim()
			print("alpha ", resultGauss.params['center'].value, resultGauss.params['center'].stderr)
			plt.vlines([resultGauss.params['center'].value - resultGauss.params['center'].stderr, resultGauss.params['center'].value + resultGauss.params['center'].stderr], ylim[0], ylim[1])
			plt.vlines([resultGauss.params['center'].value], ylim[0], ylim[1], linestyle = 'dashed')
			plt.title(f"$\\alpha$ =  {resultGauss.params['center'].value} $\\pm$ {resultGauss.params['center'].stderr}") 
			plt.ylim(ylim)
			plt.savefig(os.path.join(spectralIndexOutput, f"{src}_results_hist.png"))
			plt.close('all')

			plt.figure(figsize = (12,9), dpi = 100)
			#plt.hist(valGs, bins = binGTmp, fill = False, linewidth = 1.0)
			#plotVals, plotBins, patch = plt.hist(rawData[:, 0], bins = binG, fill = False, linewidth = 1.0, alpha = 0.05)
			#plt.errorbar(binGs, valGs, yerr = np.sqrt(valGs), fmt = 'none', elinewidth = 2., capsize = 4 * np.diff(binGs).min(), capthick = 2., color ='black', alpha = 0.1)

			firstBin = max(0, np.argwhere(valGs > 0)[0][0] - 2)
			lastBin=min(np.argwhere(valGs > 0)[-1][0] + 2, valGs.size)

			#plotVals, plotBins, patch = plt.hist(rawData[:, 0], bins = binG[firstBin:lastBin][::2], weights = 0.5 * np.ones_like(rawData[:, 0]),  fill = False, linewidth = 2.0, alpha = 1.0)
			plotVals, plotBins, patch = plt.hist(rawData[:, 0], bins = binG[firstBin:lastBin][::2],  fill = False, linewidth = 2.0, alpha = 1.0)
			ax2 = plt.gca()
			plt.tick_params(axis = 'y', direction = 'inout', width = 3, length = 16)
			plt.tick_params(axis = 'x', direction = 'inout', width = 3, length = 16)
			plt.gca().spines['right'].set_visible(False); plt.gca().spines['left'].set_visible(False)
			plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True, nbins = 5))

			plotVals, plotBins, patch = plt.twinx().hist(rawData[:, 0], bins = binG[firstBin:lastBin], fill = False, linewidth = 1.0, alpha = 0.15)
			lims = plt.xlim()
			plt.errorbar(binGs, valGs, yerr = np.sqrt(valGs), fmt = 'none', elinewidth = 2., capsize = 4 * np.diff(binGs).min(), capthick = 2., color ='black', alpha = 0.1)

			xSamples = np.linspace(lims[0], lims[1], 128)
			sf = max(1, int(np.ceil(np.log10(abs(resultGauss.params['center'].value))- np.log10(resultGauss.params['center'].stderr))))
			label=stringWithErr(resultGauss.params['center'].value, resultGauss.params['center'].stderr)
			plt.plot(xSamples, gaussian(xSamples, resultGauss.params), '--', c = 'r', linewidth = 4., label = f"$\\left<\\alpha\\right> =$ {label}")
			plt.xlim(lims)
			plt.legend(loc = 'upper right', fontsize = fontsize - 12)
			#plt.xlabel("Power Law $\\alpha$", fontsize = fontsize + 20)
			#plt.ylabel("Occurrences", fontsize = fontsize + 20)
			plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True, nbins = 5))
			for t in plt.gca().get_yticklabels():
				t.set_alpha(0.2)
			plt.tick_params(axis = 'y', direction = 'inout', width = 3, length = 16, grid_alpha = 0.2)
			plt.tick_params(axis = 'x', width = 3, length = 16)
			plt.gca().spines['right'].set_visible(False); plt.gca().spines['left'].set_visible(False); plt.grid(which ='major', axis = 'y', linestyle = '--', linewidth = 4, alpha = 0.4)
			ylim = plt.ylim()

			ax2.set_yticks(2 * plt.gca().get_yticks())
			ax2.set_xlim(lims)
			ax2.set_ylim(2 * np.array(ylim))
			#plt.gca().set_yticks([])

			plt.tight_layout(pad = 0.2)
			plt.savefig(os.path.join(paperOutputs, f"./spectral/{src}_spectral_fit.pdf"))
			plt.close('all')

			with open(f"./{src}/spectral.txt", 'w+') as ref:
				ref.writelines([f"{resultGauss.params['center'].value} {resultGauss.params['center'].stderr}"])

			# Fit a range of models to the brightness distributoon of the pulses to analyse the modulation index
			#vals, bins, patches = plt.hist(meanVal, bins = 'auto')
			__, binNs, __ = plt.hist(np.log10(np.array(meanVal) + 1), bins = 'auto', log = True)
			valNs, binNs, patches = plt.hist(meanVal, bins = 10 ** binNs, log = True)
			#valNs, binNs, patches = plt.hist(meanVal, bins = 'auto', log = True)
			plt.savefig(os.path.join(modIndexPath, f"{src}_prefit.png"))
			plt.close('all')
			bins = (binNs[:-1] + binNs[1:]) / 2
			bins = bins[valNs != 0]
			vals = valNs[valNs != 0]
			bestfit, bestAICc = None, np.inf

			errVals = np.sqrt(vals)
			modModels = {}
			for modelname, model in [
					['PowerLaw', PowerLawModel()],
					['LogNormal', lognormal],
					#['LogNormalLogNormal', LNLNModel],
					['BrokenPowerLaw', bPLModel],
					['LogNormalPowerTail', LNPLModel],
					['PowerLawLogNormal', PLLNModel]
						]:
				params = model.guess(vals, x = bins)
				# J1538+2345 fails without nan_policy='omit'
				try:
					result = model.fit(data = vals, params = params, x = bins, weights = 1 / errVals, nan_policy = 'omit')
					modModels[modelname] = result.params

					#params = model.guess(vals[2:], x = bins[2:])
					#result = model.fit(vals[2:], params, x = bins[2:])
					#print(src, result.fit_report())
					result.plot(numpoints = 128)
					# Correct the AIC for low N
					k = len(result.params)
					n = len(bins)
					#print(k, n)
					aicc = result.aic + (2 * k ** 2 +2 * k) / max(n - k - 1, 1)
					if aicc < bestAICc:
						bestfit = modelname
						bestAICc = aicc
					#print(result.fit_report())
					plt.gca().set_yscale('log')
					plt.gca().set_xscale('log')
					plt.title(f"Model(PowerLaw) {src} (N={len(meanVal)}, aic_c={aicc:.2f})"); plt.savefig(os.path.join(modIndexPath, f"{src}_{modelname}.png"))
					plt.close('all')
				except (ValueError, TypeError) as e:
					print(f"ERROR:{type(e)} was raised while trying to fit {src} to a {modelname}: {e}")
					continue
			# Copy the lowest AIC to the bestfit folder
			shutil.copy(os.path.join(modIndexPath, f"{src}_{bestfit}.png"), os.path.join(modIndexBestFitOutput, f"{src}_{bestfit}.png"))
			modModels['bestmodel'] = (bestfit, bestAICc)
			rawResults[src]['modindex'] = modModels
			overallResults[src]['modindex'] = {'bestmodel': modModels['bestmodel'], 'fit': modModels[bestfit]}
			print(modModels[bestfit])
			plt.close('all')
			# Plot the best model
			if bestfit not in ['PowerLaw', 'BrokenPowerLaw', 'LogNormalPowerTail', 'LogNormal', 'PowerLawLogNormal']:
				raise RuntimeError(f"The statistics must have improved! The best model was {bestfit}, which isn't implemnented for plotting in the paper. Fix that.")

			plt.figure(figsize = (12,9), dpi = 100)
			#print(valNs, binNs)
			plt.gca().set_yscale('log')
			plt.gca().set_xscale('log')
			vals, bins, patches = plt.hist(meanVal, binNs, fill = False, linewidth = 1.0)
			#print(vals, bins)
			bins = (bins[:-1] + bins[1:]) / 2
			ylim = plt.ylim()
			newylim = [ylim[0] * 0.33, ylim[1] * 3]
			lims = plt.xlim()
			lims = [lims[0] * 0.9, lims[1] * 1.15]
			plt.errorbar(bins, vals, yerr = np.sqrt(vals), fmt = 'none', alpha = 1.0, elinewidth = 2., capsize = 8 * np.diff(binNs).min(), capthick = 2., color ='black')

			xSamples = np.linspace(lims[0], lims[1], 4096)

			if bestfit == 'PowerLaw':
				#print(modModels[bestfit])
				sf1 = max(1, int(np.ceil(np.log10(abs(modModels[bestfit]['exponent'].value))- np.log10(modModels[bestfit]['exponent'].stderr))))
				label = stringWithErr(modModels[bestfit]['exponent'].value, modModels[bestfit]['exponent'].stderr)
				plt.plot(xSamples, brokenPowerLaw(xSamples, np.inf, modModels[bestfit]['exponent'].value, 0., modModels[bestfit]['amplitude'].value), '--', c = 'r', linewidth = 4., label = f"$n_{{F}} = {label}$")
			elif bestfit == 'BrokenPowerLaw':
				sf1 = max(1, int(np.ceil(np.log10(abs(modModels[bestfit]['alp1'].value))- np.log10(modModels[bestfit]['alp1'].stderr))) - 1)
				sf2 = max(1, int(np.ceil(np.log10(abs(modModels[bestfit]['alp2'].value))- np.log10(modModels[bestfit]['alp2'].stderr))) - 1)
				sepPnt = (xSamples[xSamples < modModels[bestfit]['sep'].value]).size
				label = stringWithErr(modModels[bestfit]['alp1'].value, modModels[bestfit]['alp1'].stderr)
				label2 = stringWithErr(modModels[bestfit]['alp2'].value, modModels[bestfit]['alp2'].stderr)
				#print(lims, sepPnt, modModels[bestfit]['sep'].value, xSamples[sepPnt])
				powerLawResult = brokenPowerLaw(xSamples, modModels[bestfit]['sep'].value, modModels[bestfit]['alp1'].value, modModels[bestfit]['alp2'].value, modModels[bestfit]['amp'].value, True)
				plt.plot(xSamples[:sepPnt], powerLawResult[0][:sepPnt], '--', c = 'r', linewidth = 4., label = f"$n_{{F,1}} = {label}$")
				plt.plot(xSamples[sepPnt:], powerLawResult[0][sepPnt:], '--', c = 'r', alpha = 0.2, linewidth = 4., label = "_nolegend_")
				plt.plot(xSamples[sepPnt:], powerLawResult[1][sepPnt:], '-.', c = 'g', linewidth = 4., label = f"$n_{{F,2}} = {label2}$")
				plt.plot(xSamples[:sepPnt], powerLawResult[1][:sepPnt], '-.', c = 'g', alpha = 0.2, linewidth = 4., label = "_nolegend_")
				plt.vlines(modModels[bestfit]['sep'].value, newylim[0], newylim[1], colors = 'k', alpha = 0.66, linestyle = 'dotted')
			elif bestfit == 'LogNormalPowerTail':
				lognormalPowerLawResult = LognormalPowerLaw(xSamples, modModels[bestfit]['sep'].value, modModels[bestfit]['A'].value, modModels[bestfit]['mu'].value, modModels[bestfit]['sigma'].value, modModels[bestfit]['alp'].value, modModels[bestfit]['amp'].value, True)
				sepPnt = (xSamples[xSamples < modModels[bestfit]['sep'].value]).size

				#print(result.fit_report())
				label0 = stringWithErr(modModels[bestfit]['mu'].value, modModels[bestfit]['mu'].stderr)
				label1 = stringWithErr(modModels[bestfit]['sigma'].value, modModels[bestfit]['sigma'].stderr)
				label2 = stringWithErr(modModels[bestfit]['alp'].value, modModels[bestfit]['alp'].stderr)
				plt.plot(xSamples[:sepPnt], lognormalPowerLawResult[0][:sepPnt], '--', c = 'r', linewidth = 4., label = f"$\\mu, \\sigma = {label0}, {label1}$")
				plt.plot(xSamples[sepPnt:], lognormalPowerLawResult[0][sepPnt:], '--', c = 'r', alpha = 0.2, linewidth = 4., label = "_nolegend_")
				plt.plot(xSamples[sepPnt:], lognormalPowerLawResult[1][sepPnt:], '-.', c = 'g', linewidth = 4., label = f"$n_{{F}} = {label2}$")
				plt.plot(xSamples[:sepPnt], lognormalPowerLawResult[1][:sepPnt], '-.', c = 'g', alpha = 0.2, linewidth = 4., label = "_nolegend_")
				plt.vlines(modModels[bestfit]['sep'].value, newylim[0], newylim[1], colors = 'k', alpha = 0.66, linestyle = 'dotted')
			elif bestfit == 'PowerLawLogNormal':
				powerLawLogNormalResult = PowerLawLognormal(xSamples, modModels[bestfit]['sep'].value, modModels[bestfit]['A'].value, modModels[bestfit]['mu'].value, modModels[bestfit]['sigma'].value, modModels[bestfit]['alp'].value, modModels[bestfit]['amp'].value, True)
				sepPnt = (xSamples[xSamples < modModels[bestfit]['sep'].value]).size

				#print(result.fit_report())
				label0 = stringWithErr(modModels[bestfit]['alp'].value, modModels[bestfit]['alp'].stderr)
				label1 = stringWithErr(modModels[bestfit]['mu'].value, modModels[bestfit]['mu'].stderr)
				label2 = stringWithErr(modModels[bestfit]['sigma'].value, modModels[bestfit]['sigma'].stderr)
				print(modModels[bestfit]['mu'].value, modModels[bestfit]['mu'].stderr, label1)
				plt.plot(xSamples[:sepPnt], powerLawLogNormalResult[0][:sepPnt], '--', c = 'r', linewidth = 4., label = f"$n_{{F}} = {label0}$")
				plt.plot(xSamples[sepPnt:], powerLawLogNormalResult[0][sepPnt:], '--', c = 'r', alpha = 0.2, linewidth = 4., label = "_nolegend_")
				plt.plot(xSamples[sepPnt:], powerLawLogNormalResult[1][sepPnt:], '-.', c = 'g', linewidth = 4., label = f"$\\mu, \\sigma = {label1}, {label2}$")
				plt.plot(xSamples[:sepPnt], powerLawLogNormalResult[1][:sepPnt], '-.', c = 'g', alpha = 0.2, linewidth = 4., label = "_nolegend_")
				plt.vlines(modModels[bestfit]['sep'].value, newylim[0], newylim[1], colors = 'k', alpha = 0.66, linestyle = 'dotted')
			elif bestfit == 'LogNormal':
				lognormalResult = LogNormalSampling(xSamples, modModels[bestfit]['amplitude'].value, modModels[bestfit]['center'].value, modModels[bestfit]['sigma'].value)

				label0 = stringWithErr(modModels[bestfit]['center'].value, modModels[bestfit]['center'].stderr)
				label1 = stringWithErr(modModels[bestfit]['sigma'].value, modModels[bestfit]['sigma'].stderr)
				plt.plot(xSamples, lognormalResult, '--', c = 'r', linewidth = 4., label = f"$\\mu, \\sigma = {label0}, {label1}$")

			else:
				raise RuntimeError(f"We do not have plotting setup for the {bestfit} model.")
				

			plt.xlim(lims)
			plt.ylim(newylim)
			plt.legend(loc = 'upper right', fontsize = fontsize - 12)
#
			plt.tick_params(axis = 'y', direction = 'inout', width = 3, length = 16)
			plt.tick_params(axis = 'x', width = 3, length = 16)
			plt.tick_params(axis = 'y', which = 'minor', width = 1.5, length = 8)
			plt.tick_params(axis = 'x', which = 'minor', width = 1.5, length = 8)
			plt.gca().spines['right'].set_visible(False); plt.gca().spines['left'].set_visible(False); plt.grid(which ='major', axis = 'y', linestyle = '--', linewidth = 4, alpha = 0.4)


			plt.gca().xaxis.set_major_formatter(FormatStrFormatter("%d"))
			plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%d"))
			plt.gca().set_xticks([1,3, 10, 30])
			plt.xlim(lims)
			
			plt.gca().xaxis.set_minor_formatter(NullFormatter())
			plt.gca().yaxis.set_minor_formatter(NullFormatter())
			
			plt.tight_layout(pad = 0.2)
			plt.savefig(os.path.join(paperOutputs, f"./modulation/{src}_modindex_fit.pdf"))



	# Save all of the results to disk
	with open("powerLawFits_narrow.pkl", 'wb') as ref:
		pickle.dump(overallResults, ref)

	with open("rawPLFits_narrow.pkl", 'wb') as ref:
		pickle.dump(rawResults, ref)
if __name__ == '__main__':
	main()
































