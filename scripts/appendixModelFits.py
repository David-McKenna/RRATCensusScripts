import pickle
import numpy as np
from astropy.table import Table
from genericFuncs import stringWithErr, fixNames
import cdspyreadme

def main(models = "rawPLFits_narrow.pkl"):
    with open(models, 'rb') as ref:
        models = pickle.load(ref)




    params = {'PowerLaw': ['exponent'], 'LogNormal': ['center', 'sigma'], 'BrokenPowerLaw': ['sep', 'alp1', 'alp2'], 'PowerLawLogNormal': ['sep', 'amp', 'mu', 'sigma'], 'LogNormalPowerTail': ['sep', 'mu', 'sigma', 'amp']}
    modelsOrder = ['PowerLaw',
     'BrokenPowerLaw',
     'PowerLawLogNormal']

    columns = [
        ('Name', str, "Source Name"),
        ('BestModel', str, "AICc determined best fit"),
        ('aPL', float, "Power Law alpha"),
        ('u_aPL', float, "Uncertainty of Power Law alpha"),
        ('bBPL', float, "Brokwn Power Law Break"),
        ('u_bBPL', float, "Uncertainty of Brokwn Power Law Break"),
        ('a1BPL', float, "Broken Power Law alpha_1"),
        ('u_a1BPL', float, "Uncertainty of Broken Power Law alpha_1"),
        ('a2BPL', float, "Broken Power Law alpha_2"),
        ('u_a2BPL', float, "Uncertainty of Broken Power Law alpha_2"),
        ('bPLLN', float, "Power Law Log Normal Break"),
        ('u_bPLLN', float, "Uncertainty of Power Law Log Normal Break"),
        ('aPLLN', float, "Power Law Log Normal alpha"),
        ('u_aPLLN', float, "Uncertainty of Power Law Log Normal alpha"),
        ('mPLLN', float, "Power Law Log Normal mu"),
        ('u_mPLLN', float, "Uncertainty of Power Law Log Normal mu"),
        ('sPLLN', float, "Power Law Log Normal sigma"),
        ('u_sPLLN', float, "Uncertainty of Power Law Log Normal sigma"),
        ('mLN', float, "Log Normal mu"),
        ('u_mLN', float, "Uncertainty of Log Normal mu"),
        ('sLN', float, "Log Normal sigma"),
        ('u_sLN', float, "Uncertainty of Log Normal sigma"),
        ('bLNPL', float, "Log Normal Power Law break"),
        ('u_bLNPL', float, "Uncertainty of Log Normal Power Law break"),
        ('mLNPL', float, "Log Normal Power Law mu"),
        ('u_mLNPL', float, "Uncertainty of Log Normal Power Law mu"),
        ('sLNPL', float, "Log Normal Power Law sigma"),
        ('u_sLNPL', float, "Uncertainty of Log Normal Power Law sigma"),
        ('aLNPL', float, "Log Normal Power Law alpha"),
        ('u_aLNPL', float, "Uncertainty of Log Normal Power Law alpha"),

    ]
    vizierModelsOrder = modelsOrder + ['LogNormal', 'LogNormalPowerTail']
    toTable = []
    for key, val in sorted(models.items(), key = lambda x: x[0]):
        if 'modindex' not in val:
            continue
        val = val['modindex']
        working = [fixNames(key), val['bestmodel'][0]]
        for model in vizierModelsOrder:
            for m in params[model]:
                working += [val[model][m].value, val[model][m].stderr]

        toTable.append(working)

    for i, row in enumerate(toTable):
        for j, col in enumerate(row):
            if not isinstance(col, str):
                if isinstance(col, type(None)) or np.isnan(col):
                    toTable[i][j] = np.ma.masked

    print(len(columns), [len(tab) for tab in toTable], len([row[0] for row in columns]), len([row[2] for row in columns]))

    table = Table(list(map(list, zip(*toTable))), names = [row[0] for row in columns], dtype = [row[1] for row in columns], descriptions = [row[2] for row in columns])
    print(table)
    maker = cdspyreadme.CDSTablesMaker()
    maker.addTable(table, name = "AppendixAmplitudeModelFits")
    maker.writeCDSTables()
    maker.makeReadMe()
    with open('appendixAmplitudeModelFits.readme', 'w') as ref:
        maker.makeReadMe(out = ref)
    with open("appendixAmplitudeModelFits.pkl", 'wb') as ref:
        pickle.dump(table, ref)
    #for fmt in ['csv', 'votable', 'fits', 'ascii', 'ascii.cds', 'ascii.daophot', 'ascii.mrt']:
    fmt = 'votable'
    table.write('appendixAmplitudeModelFits.' + fmt, overwrite=True, format = fmt) 

    tbf = '\\textbf{'
    for key, val in sorted(models.items(), key = lambda x: x[0]):
        if 'modindex' not in val:
            continue
        print(f"{fixNames(key)} ", end = '')
        val = val['modindex']
        for model in modelsOrder:
            print(" & ", end = '')
            #if model == val['bestmodel'][0]:
            #    print("\\textbf{", end = '')
            
            print(f"{' & '.join([(tbf if  model == val['bestmodel'][0] else '') + stringWithErr(val[model][m].value, val[model][m].stderr, droporders = True) + ('}' if model == val['bestmodel'][0] else '') for m in params[model]])}", end = '')
            #if model == val['bestmodel'][0]:
            #    print("}", end = '')
        print("\\\\")

if __name__ == '__main__':
    main()
