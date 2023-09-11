import pickle
import numpy as np
import os
import re

from operator import itemgetter
from copy import deepcopy
from itertools import groupby
from astropy.time import Time
from datetime import timedelta
from dreambeam.rime.scenarios import on_pointing_axis_tracking
from functools import lru_cache
from uncertainties import ufloat, ufloat_fromstr
from numpy.lib import stride_tricks

fracBand = 10
betaCorrection = 0.92


filepath = os.path.dirname(__file__)
# Load T_sky and source pointing data
with open(os.path.join(filepath, "sources_256samp_4hwhm_clean.pkl"), 'rb') as ref:
	tskys = pickle.load(ref)

with open(os.path.join(filepath, "./srcPointing_full.pkl"), 'rb') as ref:
	pnt = pickle.load(ref)

with open(os.path.join(filepath, "./sourceMappings.pkl"), 'rb') as ref:
    mappings = pickle.load(ref)

# Generic power law a * (x ** b)
def powerl(x, a ,b):
	return a * np.power(x, b)

# T_sky getter
def tempSky(src, freqs):
	pars = tskys[src][0]
	return np.mean(powerl(np.array(freqs), *pars), axis = 1)

# Pointing getter
def pointing(src):
	#print(src, pnt[src])
	return pnt[src]

pulsarNameRE = re.compile(r"[B,J]\d{4}[+-]\d{2,4}")
def _fixstr(inp):
    outp = deepcopy(inp)
    for res in re.findall(pulsarNameRE, inp):
        if res in mappings:
            outp = outp.replace(res, mappings[res])
    return outp

def fixNames(inp):
    if isinstance(inp, str):
        return _fixstr(inp)
    elif isinstance(inp, list):
        return [fixNames(line) for line in inp]
    else:
        raise RuntimeError(f"Unsupported input type: {type(inp)}")

class FakeError:

	def __init__(self, val):
		self.n = val
		self.s = np.nan

def wrapped_ufloat_fromstr(inp):
	if isinstance(inp, str):
		if inp == '--':
			return FakeError(np.nan)
		if inp[0] == '(':
			inp = f"{inp.replace('(', '').replace(')', '')}{inp}"
		working = ufloat_fromstr(inp)
		if '(' not in inp:
			working.std_dev = np.nan
		return working
	else:
		return FakeError(inp)

# Modified Kondratiev et al. T_sys calculator
def lofar_tinst_range(freqs=None):
	# polynomial coefficients
	T_inst_poly = [6.64031379234e-08, -6.27815750717e-05, 0.0246844426766, -5.16281033712, 605.474082663, -37730.3913315, 975867.990312]
	dpoly = len(T_inst_poly)


	tinsts=[]
	for flower, fupper in freqs:
		tot = 0
		df = fupper - flower
		for ii in range(101):
			freq = flower + ii*(df)/100.
			tinst = 0.0
			for jj in range(dpoly): tinst += T_inst_poly[jj]*(freq)**(dpoly-jj-1)
			tot += tinst
		tot /= 100.
		tinsts.append(tot)
	return tinsts



# Modified Kondratiev et al. a_eff calculator
# 2 tiles out of action -> 94
def get_lofar_aeff_max(freqs, nelem=94):
	"""
	Calculate the Aeff using given frequency and EL
	"""
	wavelen = 300.0 / np.vstack(freqs)
	# HBA
	if np.max(freqs) >= 100.:
		aeff = nelem * 16. * np.minimum((wavelen * wavelen)/3., 1.5625)
	# LBA (LBA_OUTER)
	else:
		aeff = nelem * (wavelen * wavelen)/3.
	return np.mean(aeff, axis = 1)

@lru_cache(8)
def cachedAxisLookup(tele, stn, antennaSet, mdl, time, dur, integ, pnt, do_parallactic_rot = True):
	return on_pointing_axis_tracking(tele, stn, antennaSet, mdl, Time(time).datetime, dur, integ, pnt, do_parallactic_rot=do_parallactic_rot)

def generateJonesCorrection(subbands, time, pnt, dur = timedelta(seconds = 1.0), integ = timedelta(seconds = 30.0), antennaSet = 'HBA', stn = 'IE613', mdl = 'Hamaker-default', meanArray = False):
	#print(subbands, time, pnt, dur, integ)
	# Get the Jones Matrix data
	#__, __, antJones, __ = on_pointing_axis_tracking("LOFAR", stn, antennaSet, mdl, time, dur, integ, pnt, do_parallactic_rot=True)
	__, __, antJones, __ = cachedAxisLookup("LOFAR", stn, antennaSet, mdl, time, dur, integ, pnt, do_parallactic_rot=True)

	# Extract our subbands
	#print(antJones.shape, subbands, type(subbands))
	jonesMatrix = antJones[subbands, :]

	# Swap the time and frequency axis
	jonesMatrix = jonesMatrix.transpose((1, 0, 2, 3))
	results = [2. / np.sum(np.multiply(ele, np.conj(ele))) for ele in jonesMatrix[0]]
	if meanArray:
		return np.mean(results, axis = 0).real
	return np.array(results).real

def calculateBrightness(snr, aeff, beamcorrection, tsys, tsky, tobs, bandwidth = fracBand, rfiflagged = 0., correction = betaCorrection):
	#print(snr.shape, aeff.shape, beamcorrection.shape, tsys.shape, tsky.shape, tobs, rfiflagged.shape)
	return snr * (1 * 2 * 1380 *(tsys + tsky) / (aeff * correction / beamcorrection)) / np.sqrt(2 * bandwidth * 1e6 * (1 - rfiflagged) * tobs)

def calculateBrightness_periodic(snr, aeff, beamcorrection, tsys, tsky, tobs, pulseWidth, period, bandwidth, rfiflagged, correction = betaCorrection):
	#print(snr.shape, aeff.shape, beamcorrection.shape, tsys.shape, tsky.shape, tobs, rfiflagged.shape)
	#return (snr * (1 * 2 * 1380 *(tsys + tsky) / (aeff / beamcorrection)) / np.sqrt(2 * bandwidth * 1e6 * (1 - rfiflagged) * tobs)) * np.sqrt(pulseWidth / (period - pulseWidth))
	effTobs = tobs * (period - pulseWidth) / pulseWidth
	return calculateBrightness(snr, aeff, beamcorrection, tsys, tsky, effTobs, bandwidth, rfiflagged, correction)

def stringWithErr(val, err, padding = 0, emptycell = '--', droporders = False):
		# Return formatted string if inputs are strings
		if isinstance(val, str) or isinstance(err, str):
				if val == emptycell:
					return val
				else:
					return f"{val}({err})"
		# Return empty cell if both values are 0 or valud  invalid
		if isinstance(val, type(None)) or (val == 0 and err == 0) or not np.isfinite(val):
				return emptycell
		# Return the value if the error is 0/invalid
		if isinstance(err, type(None)) or err == 0 or not np.isfinite(err):
				return f'{val:.2g}'
		if val == err:
			if val > 1:
				return f"({int(val)})"
			else:
				return f"({val:.2g})"

		if droporders:
			if np.log10(err) > (3 + np.log10(val)):
				if val > 1:
					return f"({int(val)})"
				else:
					return f"({val:.2g})"

		errFloat = ufloat(val, err)

		baseErrStr = f"{err:0{max(1,padding)}.0e}"
		baseErrStrTwo = f"{err:0{max(1,padding)}.1e}"
		# If the (rounded) leading error digit is a 1, we need 2 digits in the error/min 2 in the value
		if baseErrStr[0] == '1' or baseErrStrTwo[0] == '1' and err != val:
			numU = 2
		else:
			numU = 1

		returnVal = f"{errFloat:0{max(1,padding)}.{numU}uS}"
		if 'e' in returnVal:
			pre = float(returnVal.split('(')[0])
			post = float(returnVal.split('(')[1].split(')')[0])
			power = int(returnVal.split('e')[-1])

			order = int(np.log10(val)) - int(np.log10(err)) + 2
			returnVal = f"{int(float(f'{np.round(val):0{max(1,padding)}.{max(0,order)}g}'))}({int(float(f'{np.round(err):.{numU}g}'))})"

		return returnVal

def grouping(sequentialList):
	# Don't feel like bothering to setup an iterator
	outputList = []
	for __, group in groupby(enumerate(sequentialList), lambda idx: idx[0] - idx[1]):
		tmpList = list(map(itemgetter(1), group))
		tmpList.sort()
		outputList.append([tmpList[0][0], tmpList[-1][0]])
	return outputList

def rfiZapping(pulseProfile, zappedChans = np.arange(3700, 3904), gui = False, chanGroup = 8, windowSize = 16, std = 3., bandpassPassed = False, heimdallReturn = True):
	if bandpassPassed:
		bandpass = pulseProfile.copy()
	else:
		bandpass = pulseProfile.get_bandpass()

	rawBandpass = bandpass.copy()
	zapChans = np.zeros(bandpass.size)
	zapChans[zappedChans] = 1
	rfiMax = np.percentile(bandpass, 83)
	rfiMax *= 2.
	zapChans += bandpass > rfiMax

	discardLength = int(windowSize * chanGroup * 0.5)
	wordsize = bandpass.dtype.itemsize
	data_strided = stride_tricks.as_strided(bandpass[discardLength: -discardLength], shape = (bandpass[discardLength: -discardLength].size, windowSize * chanGroup), strides = (wordsize, wordsize))
	
	testVal = np.percentile(data_strided, 66, axis = 1)
	stDev = np.hstack([np.std(data_strided[idx, testVal[idx] > data_strided[idx, :]]) for idx in range(data_strided.shape[0])])

	zapChans[:discardLength] = 1
	zapChans[-discardLength:] = 1
	for chan in range(int(discardLength / 2), bandpass.shape[0] - discardLength):
		flagged = bandpass[chan] > testVal[chan - discardLength] + std * stDev[chan - discardLength]
		if zapChans[chan] or flagged:
			nearestVal = chan - chan % chanGroup
			nextVal = chan + (chanGroup - chan % chanGroup)
			bandpass[nearestVal: nextVal] = bandpass[nearestVal - chanGroup: nextVal - chanGroup]
			zapChans[nearestVal: nextVal] = 1.

	zapChansLoc = np.argwhere(zapChans)
	#bandpass[zapChansLoc] = np.median(bandpass)
	if gui:
		plt.close('all')
		fig = plt.figure(dpi = 240)
		ax = fig.gca()
		currBand, = ax.plot(bandpass, label = 'Flagged Bandpass')
		ylims = ax.get_ylim()

		ax.plot(rawBandpass, label = "Raw Bandpass", alpha = 0.33)
		ax.set_ylim(ylims)
		ax.legend()

		for tmpList in grouping(zapChansLoc):
			if len(tmpList) > 1:
				ax.axvspan(tmpList[0] - 0.5, tmpList[-1] + 0.5, alpha = 0.4, color = 'r')
			else:
				ax.axvspan(tmpList[0] - 0.5, tmpList[0] + 0.5, alpha = 0.4, color = 'r')

		def clickEvent(event):
			nearestVal = int(event.xdata)
			nearestVal -= nearestVal % chanGroup
			print(f"Click detected at {event.xdata}, toggling data flags from {nearestVal} to {nearestVal + chanGroup}")
			if (zapChans[nearestVal: nearestVal + chanGroup] == 1.).any():
				zapChans[nearestVal: nearestVal + chanGroup] = 0.
				ax.axvspan(nearestVal, nearestVal + chanGroup, alpha = 0.3, color = 'g')
			else:
				zapChans[nearestVal: nearestVal : chanGroup] = 1.
				ax.axvspan(nearestVal, nearestVal + chanGroup, alpha = 0.6, color = 'r')

			currBand.set_data(np.arange(bandpass.size), bandpass)
			currBand.figure.canvas.draw()

			hGroups = grouping(np.argwhere(zapChans))
			print(f"Heimdall flags: {' -zap_chans '.join([''] + [' '.join([str(ele[0]), str(ele[-1])]) if len(ele) > 1 else str(ele) + ' ' + str(ele) for ele in hGroups])}")



		cid = fig.canvas.mpl_connect('button_press_event', clickEvent)
		plt.show(block = True)


		zapChansLoc = np.argwhere(zapChans)

	if heimdallReturn:
		return f"{' -zap_chans '.join([''] + [' '.join([str(ele[0]), str(ele[-1])]) if len(ele) > 1 else str(ele) + ' ' + str(ele) for ele in grouping(zapChansLoc)])}"
	else:
		return zapChansLoc, grouping(zapChansLoc)


def exploreFolderTree(path, ext = 'cand'):
	returnList = []
	for base, dirs, files in os.walk(path):
		for file in files:
			if ext == file.split('.')[-1]:
				returnList.append(os.path.join(base, file))

	return returnList

def parseCandidate(line):
	if isinstance(line, str):
		return [func(lineEle) for func, lineEle in zip([float, int, float, int, int, float, int, int, int], line.strip('\n').split())]
	else:
		return [parseCandidate(lineEle) for lineEle in line]

def extractHeimdallCand(file):
	with open(file, 'r') as fileRef:
		cands = fileRef.readlines()
	return parseCandidate(cands)


def resample(data, ranges):
        rsr = []
        for l, r in ranges:
                rsr.append(data[l:r])

        rsr = np.hstack(rsr)
        data -= rsr.mean()


        rsr = []
        for l, r in ranges:
                rsr.append(data[l:r])

        rsr = np.hstack(rsr)
        data /= rsr.std()

        return data

def rollingAverage(series, size):
        ds = np.cumsum(series)
        return ds[size:] - ds[:-size]
