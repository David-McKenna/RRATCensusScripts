import numpy as np
import sys

def main(inputs):
	print(inputs[0])
	with open(inputs[0], 'r') as ref:
		dataraw = np.array([list(map(float, line.strip('\n').split())) for line in ref.readlines()])

	rawmean = dataraw[:, 0].mean()
	outliersIdx = np.abs(dataraw[:, 0] - rawmean) > 0.5
	outliers = dataraw[outliersIdx]
	largeerrorIdx = dataraw[:, 1] > 0.5
	largeerror = dataraw[largeerrorIdx]
	print(len(largeerror), largeerror)

	data = dataraw[np.logical_and(np.logical_not(outliersIdx), np.logical_not(largeerrorIdx))]

	dm = data[:, 0].mean()
	ddm = np.sqrt(np.sum(np.square(data[:, 1]))) / data.shape[0]

	print(inputs[0], dm, ddm)
	print("\n\n")
	with open(inputs[1], 'w') as ref:
		ref.write(f"{dm} {ddm}")


if __name__ == '__main__':
	main(sys.argv[1:])