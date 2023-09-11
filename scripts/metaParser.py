import os
import numpy as np
import pygedm
import pickle
from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.units.cds as u_cds
from astropy.io import ascii
from astropy.table import Table
from collections import defaultdict
import cdspyreadme
from uncertainties import ufloat

from genericFuncs import stringWithErr, exploreFolderTree, fixNames, wrapped_ufloat_fromstr

nameSynonyms = {

}

manualCorrections = {

}

catalogue = {
	'J0054+66': 'RRATalog',
	'J0103+54': 'RRATalog',
	'J0746+55': 'CHIME/FRB',
	'J1005+3015': 'PRAO',
	'J1336+3346': 'PRAO',
	'J1400+2127': 'PRAO',
	'J1931+4229': 'CHIME/FRB',
	'J2215+45': 'CHIME/FRB',
}


# Sanity checked against tempo2 output 2021-10-22
def periodTransform(f, fdot, df = 0, dfdot = 0):
	p = 1/f
	dp = np.sqrt(np.square(df / f)) * p

	pdot = -fdot / np.square(f)
	if fdot >= 0:
		dpdot = 0.
	else:
		dpdot = np.sqrt(np.square(dfdot / fdot) + 2 * np.square(df / f)) * pdot

	return (p, dp), (pdot, dpdot)

def characteristicAge(p, pdot):
	if pdot == 0:
		return 0
	return p / (2 * pdot) / (365 * 24 * 60 * 60)
def magneticField(p, pdot):
	if pdot == 0:
		return 0
	return 3.2e19 * np.sqrt(p * pdot)


def parToDict(lines):
	result = {}
	for line in lines:
		line = line.strip('\n').split()

		## More than 1 components -> extract the value and the error
		if len(line) > 2:
			result[line[0]] = [line[1], line[-1]]
		else:
			result[line[0]] = [line[1], 0]

	print(result)

	if 'DM' in result.keys():
		if isinstance(result['DM'][1], str):
			result['DM'][1] = float(result['DM'][1])

		result['DM'][0] = float(result['DM'][0])
	expectedKeys = ['F0', 'F1', 'DM', 'PEPOCH', 'POSEPOCH', 'DMEPOCH', 'DM', 'TZRMJD', 'TZRFREQ', 'TRES', 'NTOA', 'START', 'FINISH']
	for key in ['P0', 'P1'] + expectedKeys:
		if key in result.keys():
			if len(result[key]) == 2:
				result[key][1] = float(result[key][1])

			result[key][0] = float(result[key][0])
		elif key in expectedKeys:
			print(key)
			result[key] = (np.nan, np.nan)

	for key in ['RAJ', 'DECJ']:
		if key in result.keys():
			result[key][1] = float(result[key][1])
			if result[key][0].count(':') != 2:
				result[key][0] = f"{result[key][0]}:00"
	if 'P0' not in result:
		result['P0'], result['P1'] = periodTransform(result['F0'][0], result['F1'][0], result['F0'][1], result['F1'][1])

	if result['PSRJ'][0] in manualCorrections.keys():
		for key, val in manualCorrections[result['PSRJ'][0]].items():
			print(f"Correction applied {result['PSRJ'][0]}: {key}: {val}")
			result[key] = val

	result['CATSRC'] = [result['PSRJ'][0] if 'NAME' in result.keys() else '--', 0]
	result['OURNAME'] = [result['NAME'][0] if 'NAME' in result.keys() else result['PSRJ'][0], 0]
	result['CAT'] = [catalogue[result['PSRJ'][0]] if result['PSRJ'][0] in catalogue else '--', 0]

	acToDeg = (360 / 24) / 60 / 60
	sToDeg = (1) / 60 / 60
	apSkyObj = SkyCoord(result['RAJ'][0], result['DECJ'][0], unit = 'hourangle, degree').transform_to('galactic')
	result['GALL'] = [apSkyObj.l.deg, result['RAJ'][1] * acToDeg]
	result['GALB'] = [apSkyObj.b.deg, result['DECJ'][1] * sToDeg]

	result['DIST'] = [pygedm.dm_to_dist(apSkyObj.l.deg, apSkyObj.b.deg, result['DM'][0], method = 'ymw16')[0].value, 0]
	result['AGE'] = [characteristicAge(result['P0'][0], result['P1'][0]) / 1e6, 0]
	result['B'] = [magneticField(result['P0'][0], result['P1'][0]) / 1e12, 0]
	print(result['B'])
	result['P1'] = [result['P1'][0] * 1e15, result['P1'][1] * 1e15]


	return result

def parsePar(path):
	with open(path,'r') as ref:
		lines = ref.readlines()

	return parToDict(lines)


def getTargetPars(path, names):
	localNames = names.copy()
	pars = exploreFolderTree(path, 'par')

	targets = []
	for par in pars:
		if os.path.basename(par) in localNames:
			targets.append(par)
			del localNames[localNames.index(os.path.basename(par))]

	if len(localNames) > 0:
		print(f"ERROR: Unable to find par(s) that were requested: {', '.join(localNames)}")
		return

	sources = {}
	for par in targets:
		src = os.path.basename(par).split('_')[0].split('.')[0]
		sources[src] = parsePar(par)
	"""
def getTargetPars(path, names):
	#sources = {}
	#for par in names:
	#	src = os.path.basename(par).split('_')[-1].rstrip('.par')
	#	sources[src] = parsePar(par)
	"""

	for src, par in sources.items():
		print(f"{src} P0: {stringWithErr(*par['P0'])}")
		print(f"{src} P1: {stringWithErr(*par['P1'])}")

	toTable = []
	columnsUnits = [
		('Name', None, str, "Source Name"),
		('Cat', None, str, "Source Catalogue"),
		('CatSrc', None, str, "Original name in source catalogue"),
		("RA", u.hourangle, str, "Right Ascension (J2000)"),
		("u_RAs", u.arcsecond * 15, float, "Uncertainty of Right Ascension (second)"),
		("DEC", u.degree, str, "Declination (J2000)"),
		("u_DECs", u.arcsecond, float, "Uncertainty of Declination (arcseconds)"),
		("GLON", u.degree, float, "Galactic Longitude"),
		("u_GLON", u.degree, float, "Uncertainty of Galactic Longitude (degree)"),
		("GLAT", u.degree, float, "Galatic Latitude"),
		("u_GLAT", u.degree, float, "Uncertainty of Galactic Latitude (degree)"),
		("DM", u.parsec / (u.cm ** 3), float, "Dispersion measure"),
		("u_DM", u.parsec / (u.cm ** 3), float, "Dispersion measure"),
		('DIST', u.parsec * 1000, float, "Source distance (as per YWM16)"),
		('u_DIST', u.parsec * 1000, float, "Uncertainty of source distance (10%, as per YWM16)"),
		('P0', u.second, float, "Rotation Period"),
		('u_P0', u.second, float, "Uncertainty of rotation period"),
		('P1', u.second * 1e-15 / u.second, float, "Spin-down Rate (seconds per second)"),
		('u_P1', u.second * 1e-15 / u.second, float, "Uncertainty of spin down rate (seconds per second)"),
		('AGE', u.megayear, float, "Characteristic Age"),
		('B', u.gauss * 1e12, float, "Surface Magnetic Field"),
		('START', u_cds.MJD, int, "Start of timing range"),
		('FINISH', u_cds.MJD, int, "End of timing range"),
		('PEPOCH', u_cds.MJD, int, "P0 reference epoch"),
		('NTOA', None, int, "Number of TOA measurements used in timing fit"),
		('TRES', u.microsecond, float, "Residual time in timing fit"),

	]
	for key, meta in sorted(sources.items(), key = lambda x: x[0]):
		raj = sources[key]['RAJ'][0].split('(')[0]
		rajs = ufloat(sources[key]['RAJ'][0].split(':')[-1], sources[key]['RAJ'][1])

		decj = sources[key]['DECJ'][0].split('(')[0]
		decjs = ufloat(sources[key]['DECJ'][0].split(':')[-1], sources[key]['DECJ'][1])

		gall_uFloat = ufloat(*sources[key]['GALL'])
		galb_uFloat = ufloat(*sources[key]['GALB'])
		toTable.append([
			fixNames(sources[key]['OURNAME'][0]), 
			sources[key]['CAT'][0],
			sources[key]['CATSRC'][0],
			raj,
			rajs.s,
			decj,
			decjs.s,
			gall_uFloat.n,
			gall_uFloat.s,
			galb_uFloat.n,
			galb_uFloat.s,
			sources[key]['DM'][0],
			sources[key]['DM'][1],
			sources[key]['DIST'][0],
			sources[key]['DIST'][0] * 0.1,
			sources[key]['P0'][0],
			sources[key]['P0'][1],
			sources[key]['P1'][0],
			sources[key]['P1'][1],
			sources[key]['AGE'][0],
			sources[key]['B'][0],
			sources[key]['START'][0],
			sources[key]['FINISH'][0],
			sources[key]['PEPOCH'][0],
			sources[key]['NTOA'][0],
			sources[key]['TRES'][0],
			])

	for i, row in enumerate(toTable):
		for j, col in enumerate(row):
			if not isinstance(col, str):
				if np.isnan(col):
					toTable[i][j] = np.ma.masked

	print(len(columnsUnits), [len(tab) for tab in toTable], len([row[0] for row in columnsUnits]), len([row[2] for row in columnsUnits]))

	table = Table(list(map(list, zip(*toTable))), names = [row[0] for row in columnsUnits], units = [row[1] for row in columnsUnits], dtype = [row[2] for row in columnsUnits], descriptions = [row[3] for row in columnsUnits])
	print(table)
	maker = cdspyreadme.CDSTablesMaker()
	cdstab = maker.addTable(table, name = "RRATTimingEphemerides")
	cdstab.get_column("RA").setSexaRa()
	cdstab.get_column("DEC").setSexaDe()
	maker.writeCDSTables()
	maker.makeReadMe()
	with open('rratEphemerides.readme', 'w') as ref:
		maker.makeReadMe(out = ref)
	with open("rratEphemerides.pkl", 'wb') as ref:
		pickle.dump(table, ref)
	#for fmt in ['csv', 'votable', 'fits', 'ascii', 'ascii.cds', 'ascii.daophot', 'ascii.mrt']:
	fmt = 'votable'
	table.write('rratEphemerides.' + fmt, overwrite=True, format = fmt) 




	print("\\hline\\hline")
	for key, label in [('OURNAME', 'Source'), ('CAT', 'Catalogue'), ('CATSRC', 'Catalogue Source Name')]:
		print(f"{label} ", end = '')
		for src in sorted(sources.keys()):
			print(f"& {sources[src][key][0]} ", end = '')
		print("\\\\")
		if label == 'Source':
			print("\\\\")
	
	for key in sources:
		sources[key]['DIST'][0] = f"{sources[key]['DIST'][0]:.3g}"





	print("\\hline")

	for key, label in [('RAJ', 'Right Ascension (hms)'), ('DECJ', 'Declination (dms)')]:
		print(f"{label} ", end = '')
		for src in sorted(sources.keys()):
			valsplit = sources[src][key][0].split(':')
			print("& ", end = '')
			if key == 'RAJ':
				print("\\hmsangle{", end = '')
			elif key == 'DECJ':
				print("\\dmsangle{", end = '')
			else:
				raise RuntimeError(f"Unexpected key: {key}")
			errStr = stringWithErr(float(valsplit[-1]), sources[src][key][1] or np.nan, padding = 2)
			#print(f"\n{key} {valsplit} {sources[src][key][1]} {errStr}")
			if errStr == '--' or sources[src][key][1] == 0.0 or sources[src][key][1] == np.nan:
				print(';'.join(valsplit), end = '}')
				continue

			print(f"{';'.join(valsplit[0:-1])};{errStr.split('(')[0]}}}({errStr.split('(')[1]} ", end = '')
			#print(f"& {':'.join(valsplit[0:-1])}:{stringWithErr(float(valsplit[-1]), sources[src][key][1], padding = 2)} ", end = '')
		print('\\\\')

	for key, label in [('GALL', 'Galactic Longitude ($^\\circ$)'), ('GALB', 'Galactic Latitude ($^\\circ$)')]:
		print(f"{label} ", end = '')
		for src in sorted(sources.keys()):
			print("& ", end = '')
			errStr = stringWithErr(*sources[src][key], padding = 2)
			if sources[src][key][1] == 0.0 or sources[src][key][1] == np.nan:
				print(f"\\galangle{{{sources[src][key][0]}}}", end = '')
				continue
			print(f"\\galangle{{{errStr.split('(')[0]}}}({errStr.split('(')[1]} ", end = '')
		print('\\\\')

	for key, label in [('DM', 'Dispersion Measure  (\\SI{}{\\parsec\\per\\centi\\metre\\cubed})'),
				('DIST', 'Distance (\\SI{}{\\parsec})'), ('', 'NEWLINE'), ('P0', 'Period (\\SI{}{\\second})'),
				('P1', 'Period Derivative (\\SI{e-15}{\\second\\per\\second})'),
				('AGE', 'Characteristic Age (\\SI{}{\\mega\\year})'), ('B', 'Magnetic Field (\\SI{e12}{\\gauss})')
				]:
		if label == 'NEWLINE':
			print("\\\\")
			continue
		print(f"{label} ", end = '')
		if key not in ['DIST', 'AGE', 'B']:
			for src in sorted(sources.keys()):
				print(f"& {stringWithErr(*sources[src][key])} ", end = '')
		else:
			for src in sorted(sources.keys()):
				print(f"& {np.format_float_positional(float(sources[src][key][0]), 3, fractional = False, trim = '-')} ", end = '')			
		print('\\\\')


	print('\\\\')
	for key, label in [('START', 'Timing Start (MJD)'), ('FINISH', 'Timing End (MJD)'), ('PEPOCH', 'Reference Epoch (MJD)'),
				('NTOA', 'N\\textsubscript{TOAs}'), ('TRES', 'Model Residuals (\\SI{}{\\micro\\second})')]:
		print(f"{label} ", end = '')
		for src in sorted(sources.keys()):
			if np.isnan(sources[src][key][0]):
				print('& --', end = '')
			else:
				print(f"& {int(sources[src][key][0])} ", end = '')
		print('\\\\')


	print("\\hline\\hline")
	return sources


def getAllPars(path):
	pars = exploreFolderTree(path, 'par')

	baseNames = []
	for par in pars:
		baseNames.append(os.path.basename(par))

def openread(path, raiseError = False, defaultReturn = '--'):
	if os.path.exists(path):
		with open(path, 'r') as ref:
			return ref.readlines()
	elif raiseError:
		raise RuntimeError(f"ERROR: Unable to find file at {path}")
	return [defaultReturn]

def openreaderr(path, raiseError = False, defaultReturn = '--'):
	if os.path.exists(path):
		with open(path, 'r') as ref:
			lines = ref.readlines()[0]
		return list(map(float, lines.split()))
	elif raiseError:
		raise RuntimeError(f"ERROR: Unable to find file at {path}")
	return [defaultReturn, defaultReturn]


def getTargetMetadata(initPath):
	paths = []
	for path in os.listdir(initPath):
		if os.path.isdir(os.path.join(initPath, path)) and path[0] == 'J':
			paths.append(os.path.join(initPath, path))

	template = {
		'catalogue': "--",
		'detected': "--",
		'period': "--",
		'dm': '--',
		'tobs': '--',
		'npulses': '--',
		'swidth': '--',
		'sduty': '--',
		'sSpeak': '--',
		'sSratio': '--',
		'sspectral': '--',
		'srate': '--',
		'pwidth': '--',
		'pduty': '--',
		'pSmean': '--',
		'pspectral': '--',
	}	


	sources = {}
	for src in paths:
		print(src)
		key = os.path.basename(src)
		sources[key] = template.copy()

		sources[key]['catalogue'] = openread(f"{src}/cat.txt", raiseError = True)[0][0]
		sources[key]['detected'] = openread(f"{src}/previous.txt")[0]
		period = eval(openread(f"{src}/period.txt")[0])
		sources[key]['period'] = f"{float(period):.4f}" if str(period)[0] != '-' else '--'
		sources[key]['dm'] = stringWithErr(*(openreaderr(f"{src}/dm.txt", raiseError = True)))
		tobs = float(openread(os.path.join(src, 'tobs.txt'), raiseError = True)[0])
		sources[key]['tobs'] = f"{tobs:.1f}"
		sources[key]['tobsraw'] = tobs

		if not os.path.exists(f"{src}/NO_PULSES"):
			if len(os.listdir(f"{src}/pulses/")) < 2:
				open(f"{src}/NO_PULSES", 'w').close()

		if not os.path.exists(f"{src}/NO_PULSES"):
			sources[key]['npulses'] = openread(f"{src}/pulseCount.txt", raiseError = True)[0]
			pulsewidths = openreaderr(f"{src}/width.txt", raiseError = True)
			sources[key]['swidth'] = stringWithErr(*(1000 * np.array(pulsewidths)))
			if sources[key]['period'] != '--':
				sources[key]['sduty'] = stringWithErr(*(100 * np.array(pulsewidths) / float(sources[key]['period'])))

			pulseBright = openreaderr(f"{src}/pulseBright.txt", raiseError = True)
			peak, ratio, width = map(float, pulseBright)
			sources[key]['sSpeak'] = f"{int(peak)}"
			sources[key]['lumWidthVals'] = (width, peak)
			sources[key]['sSratio'] = f"{int(np.round(ratio)):3d}"
			sources[key]['sSpeakRaw'] = (peak, ratio, width)
			sources[key]['srate'] = stringWithErr(float(sources[key]['npulses']) / tobs, np.sqrt(float(sources[key]['npulses'])) / tobs) if sources[key]['npulses'][0] != '-' else '--'
			sources[key]['sspectral'] = stringWithErr(*openreaderr(f"{src}/spectral.txt", raiseError = False))
		else:
			sources[key]['sSpeakRaw'] = (np.ma.masked, np.ma.masked, np.ma.masked)

		if os.path.exists(f"{src}/PERIODIC"):
			pwidth = openreaderr(f"{src}/widthPeriodic.txt", raiseError = True)
			sources[key]['pwidth'] = f"{int(1000*pwidth[0]):2d}"
			sources[key]['pduty'] = f"{100 * np.array(pwidth[0]) / float(sources[key]['period']):3.1f}"
			pmean = openreaderr(f"{src}/smeanPeriodic.txt", raiseError = True)[0]
			sources[key]['pSmean'] = f"{1000 * float(pmean):3.1f}"
			pspectral = openreaderr(f"{src}/spectralPeriodic.txt", raiseError = False)
			sources[key]['pspectral'] = stringWithErr(*(np.array(pspectral)))


	toTable = []
	columnsUnits = [
		('Name', None, str, "Source Name"),
		('Catalogue', None, str, "Original Source Catalogue"),
		('Previous', None, str, "Previous LOFAR work related to the source"),
		('Period', u.s, float, "Rounded rotation period"),
		('DM', u.parsec / (u.cm ** 3), float, "Source dispersion measure; default from single-pulse data, otherwise from folded profile best-fit."),
		('u_DM', u.parsec / (u.cm ** 3), float, "Uncertainty of dispersion measure"),
		('Tobs', u.hour, float, "Observing time spent on the source"),
		('Npulses', None, int, "Number of detected single pulses"),
		('sWidth', u.millisecond, float, "Mean single-pulse width"),
		('u_sWidth', u.millisecond, float, "Standard deviation of pulse widths"),
		('sDuty', u.percent, float, "Mean duty cycle for single pulses"),
		('u_sDuty', u.percent, float, "Standard deviation of pulse duty cycles"),
		('sPeak', u.Jy, float, "Brightness of the brightest single-pulse"),
		('u_sPeak', u.Jy, float, "Uncertainty of peak brightness (50%)"),
		('wSpeak', u.millisecond, float, "Width of the brightest single-pulse"),
		('SpeakR', None, float, "Ratio of brightnesses between the brightest and dimmest pulse from the source"),
		('Sp+IndexPulse', None, float, "Spectral power law fit to single-pulse data"),
		('u_Sp+IndexPulse', None, float, "Uncertainty of power law fit"),
		('sRate', 1 / u.hour, float, "Per-hour single-pulse burst rate"),
		('u_sRate', 1 / u.hour, float, "Uncertainty of burst rate (Poisson)"),
		('pWidth', u.millisecond, float, "Width of period emission"),
		#('u_pWidth', u.millisecond, float),
		('pDuty', u.percent, float, "Duty cycle of periodic emission"),
		#('u_pDuty', u.percent, float),
		('pSmean', u.Jy / 1000, float, "Brightness of periodic emission"),
		('u_pSmean', u.Jy / 1000, float, "Uncertainty of brightness (50%)"),
		('Sp+IndexPeriodic', None, float, "Spectral power law fit to periodic emission data"),
		('u_Sp+IndexPeriodic', None, float, "Uncertainty of power law fit"),
	]
	for key, meta in sorted(sources.items(), key = lambda x: x[0]):
		dm_uFloat = wrapped_ufloat_fromstr(meta['dm'])
		swidth_uFloat = wrapped_ufloat_fromstr(meta['swidth'])
		sduty_uFloat = wrapped_ufloat_fromstr(meta['sduty'])
		sspectral_uFloat = wrapped_ufloat_fromstr(meta['sspectral'])
		srate_uFloat = wrapped_ufloat_fromstr(meta['srate'])
		pwidth_uFloat = wrapped_ufloat_fromstr(meta['pwidth'])
		pduty_uFloat = wrapped_ufloat_fromstr(meta['pduty'])
		pspectral_uFloat = wrapped_ufloat_fromstr(meta['pspectral'])

		toTable.append([
			fixNames(key), 
			meta['catalogue'],
			meta['detected'] if meta['detected'] != '--' else np.ma.masked,
			wrapped_ufloat_fromstr(meta['period']).n,
			dm_uFloat.n,
			dm_uFloat.s,
			meta['tobsraw'],
			wrapped_ufloat_fromstr(meta['npulses']).n,
			swidth_uFloat.n,
			swidth_uFloat.s,
			sduty_uFloat.n,
			sduty_uFloat.s,
			wrapped_ufloat_fromstr(meta['sSpeak']).n,
			wrapped_ufloat_fromstr(meta['sSpeak']).n * 0.5,
			float(meta['sSpeakRaw'][2]), # width of brightest
			float(meta['sSpeakRaw'][1]), # ratio
			sspectral_uFloat.n,
			sspectral_uFloat.s,
			srate_uFloat.n,
			srate_uFloat.s,
			pwidth_uFloat.n,
			#pwidth_uFloat.s,
			pduty_uFloat.n,
			#pduty_uFloat.s,
			wrapped_ufloat_fromstr(meta['pSmean']).n,
			wrapped_ufloat_fromstr(meta['pSmean']).n * 0.5,
			pspectral_uFloat.n,
			pspectral_uFloat.s])

	for i, row in enumerate(toTable):
		for j, col in enumerate(row):
			if not isinstance(col, str):
				if np.isnan(col):
					toTable[i][j] = np.ma.masked

	#print(len(columnsUnits), [len(tab) for tab in toTable], len([row[0] for row in columnsUnits]), len([row[2] for row in columnsUnits]))

	table = Table(list(map(list, zip(*toTable))), names = [row[0] for row in columnsUnits], units = [row[1] for row in columnsUnits], dtype = [row[2] for row in columnsUnits], descriptions = [row[3] for row in columnsUnits])
	print(table)
	print(table['Name'])
	print(table['Sp+IndexPulse'], table['Sp+IndexPeriodic'])
	print(table['u_Sp+IndexPulse'], table['u_Sp+IndexPeriodic'])
	maker = cdspyreadme.CDSTablesMaker()
	maker.addTable(table, name = "RRATObservedCharacteristics")
	maker.writeCDSTables()
	maker.makeReadMe()
	with open('rratCharacteristics.readme', 'w') as ref:
		maker.makeReadMe(out = ref)
	with open("rratCharacteristics.pkl", 'wb') as ref:
		pickle.dump(table, ref)
	#for fmt in ['csv', 'votable', 'fits', 'ascii', 'ascii.cds', 'ascii.daophot', 'ascii.mrt']:
	fmt = 'votable'
	table.write('rratCharacteristics.' + fmt, overwrite=True, format = fmt) 


	print("\\begin{tabular}{lccccc|ccccccc|cccc}\n\\hline\\hline\n& & & & & & \\multicolumn{7}{c|}{Single Pulse} & \\multicolumn{4}{c}{Periodic Fold} \\\\")
	print("Source & Cat. & Prev. & Period & DM & T\\textsubscript{obs} & N\\textsubscript{pulses} &  w$_{10}$ & Duty Cycle & S$_{150}^{\\text{peak}}$\\textsuperscript{a} & SR &$\\alpha$\\textsuperscript{b} & Burst Rate\\textsuperscript{c} &  w$_{10}$ & Duty Cycle &S$_{150}^{\\text{mean}}$\\textsuperscript{a} &  $\\alpha$\\textsuperscript{b}\\\\")
	print("   & & & (\\SI{}{\\second}) & (\\SI{}{\\parsec\\per\\cubic\\centi\\metre})  & (\\SI{}{\\hour}) & & (\\SI{}{\\milli\\second}) & \\% & (\\SI{}{\\jansky}) & & & (\\SI{}{\\per\\hour}) & (\\SI{}{\\milli\\second}) & \\% & (\\SI{}{\\milli\\jansky}) & \\\\\n\\hline")

	footnotes = defaultdict(lambda: '', {
		'J0209+58': '\\textsuperscript{d}',
		'J2108+45': '\\textsuperscript{e}',
	})
	detectedCitation = defaultdict(lambda: '--', {
		'K15': '\\citetalias{karako-argamanDiscoveryFollowupRotating2015}',
		'M18': '\\citetalias{michilliSinglepulseClassifierLOFAR2018}',
		'S19': '\\citetalias{sanidasLOFARTiedArrayAllSky2019b}',
	})

	# Remove from census for now, altris sourced period
	sources['J0348+79']['period'] = '--'

	printables = []
	for key, meta in sorted(sources.items(), key = lambda x: x[0]):
		printables.append(f"{key}{footnotes[key]} & {meta['catalogue']} & {detectedCitation[meta['detected']]} & {meta['period']} & {meta['dm']} & {meta['tobs']} & {meta['npulses']} & {meta['swidth']} & {meta['sduty']} & {meta['sSpeak']} & {meta['sSratio']} & {meta['sspectral']} & {meta['srate']} & {meta['pwidth']} & {meta['pduty']} & {meta['pSmean']} & {meta['pspectral']} \\\\")

	printables = fixNames(printables)
	for row in printables:
		print(row)

	print("\n\\hline\\hline\n\\end{tabular}")
	return sources

def main():
	targets = [
		'J0054+66.par',
		'J0103+54.par',
#		'J0209+58_real.par',
		'J0746+55.par',
		'J1931+4229.par',
		'J2215+45.par',
		'J1005+3015.par',
		'J1336+34.par',
		'J1400+21.par'
	]
	pars = getTargetPars("./pars/", targets)

	print("\n\n\n\n\n\n")
	results = getTargetMetadata("./")

	with open("./srcPointing_full.pkl", 'rb') as ref:
		coords = pickle.load(ref)
	print(coords)
	for source in results.keys():
		if 'lumWidthVals' not in results[source]:
			continue
		apSkyObj = coords[source].transform_to('galactic')
		dm = float(results[source]['dm'].split('(')[0])
		dist = pygedm.dm_to_dist(apSkyObj.l.deg, apSkyObj.b.deg, dm, method = 'ymw16')[0].value
		print(f"{source}\ta\tb\tc\t{results[source]['lumWidthVals'][0] * 0.151}\t{results[source]['lumWidthVals'][1] * (dist / 1000) ** 2:6g}")

if __name__ == '__main__':
	main()
