# coding: utf-8
import os
import sigpyproc as spp


def main(dirPath = = '.', toaerr = 1000):
    data = {}
    for fil in os.listdir(os.path.join(dirPath, "/fils/")):
        if fil.endswith('P000.fil'):
            data[fil.split('_cDM')[0]] = [spp.FilReader(os.path.join(dirPath, "./fils/{fil}"))]
        elif fil.endswith('rescale.fil'):
            data[fil.split('_0001')[0]] = [spp.FilReader(os.path.join(dirPath, "./fils/{fil}"))]
        else:
            continue

    data2 = data.copy()
    for key in data.keys():
        try:
            with open(os.path.join(dirPath, "/cands/{key}.cand"), 'r') as ref:
                data2[key].append(ref.readlines())
        except:
            del data2[key]

    data = data2
            
    toas = {}
    for key, (reader, cands) in data.items():
        mjdstart = reader.header.tstart
        centre = [mjdstart + float(line.split()[1]) * reader.header.tsamp / (3600 * 24) for line in cands]
        start = [mjdstart + float(line.split()[7]) * reader.header.tsamp / (3600 * 24) for line in cands]
        end = [mjdstart + float(line.split()[8]) * reader.header.tsamp / (3600 * 24) for line in cands]
        snr = [float(line.split()[0]) for line in cands]
        if reader.header.telescope_id == 1916:
            telescope = 'Ielfrhba'
        elif reader.header.telescope_id == 0:
            telescope = 'Frlfrhba'
        toas[key] = (centre, start, end, snr, reader.header.fcenter, telescope)

    for key, (toas, start, end, snrs, fcen, tele) in toas.items():
        if not os.path.exists(os.path.join(dirPath, "./cands/heimdallTOA/")):
            os.mkdir(os.path.join(dirPath, "./cands/heimdallTOA/"))
        with open(os.path.join(dirPath, "./cands/heimdallTOA/{key}.tim"), 'w') as ref:
            ref.write("FORMAT 1\n")
            for snr, toa in zip(snrs, toas):
                ref.write(f"{key}_{snr} {fcen} {toa} {toaerr:.2f} {tele}\n")

if __name__ == '__main__':
    main()

