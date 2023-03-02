import sigpyproc
import sys
import os

def main(target):
	print(target, target.replace('.tim', '.vap'))
	with open(target, 'r') as ref:
		tim = ref.readlines()

	with open(target.replace('.tim', '.vap'), 'r') as ref:
		vap = ref.readlines()

	offsets = {line.split()[0]: float(line.split()[1]) for line in vap}
	print(target.replace('.tim', '_corrected.tim'))
	with open(target.replace('.tim', '_corrected.tim'), 'w') as ref:
		ref.write("FORMAT 1\n")
		for line in tim[1:]:
			split = line.split()
			rdr = sigpyproc.FilReader(split[0].replace('.zap.ar', '.fil'))
			delay = (rdr.header.tstart % 1) - offsets[os.path.basename(split[0])]
			print(rdr.header.tstart % 1, offsets[os.path.basename(split[0])], delay, delay * 86400)
			ref.write(f"{split[0]}\t{split[1]}\t{float(split[2]) + delay}\t{split[3]}\t{split[4]}\n")


if __name__ == '__main__':
	main(sys.argv[1])