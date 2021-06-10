import data
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import pandas as pd
import tensorly as tl
from sklearn.decomposition import PCA
from tensorly.decomposition import parafac2

def parse_line(line):
	line = line.replace('\n', '').split('\t')
	data = []
	for number in line:
		if number == '':
			data.append(0)
		elif not number.isdigit() and not "," in number:
			continue
		else:
			data.append(float(number.replace(',', '.')))

	return data

# reading sample out of the file
def read_data(filename, line=None):
	with open(filename) as file:
		lines = file.readlines()
		y = parse_line(lines[3])
		x = []
		# making matrix for the input data
		matrix = []
		for i in range(4, len(lines)):
			numbers = parse_line(lines[i])
			x.append(numbers[0])
			if len(numbers) < 52:
				numbers.extend([0 for _ in range(77 - len(numbers))])
			zs = [numbers[j] for j in range(1, len(numbers))]
			matrix.append(zs)
		z = np.array(matrix).T
		return x, y, z


# getting samples out of files list
def get_data(files_list, dir):
	z_samples = []
	for filename in files_list:
		X, Y, z_matrix, t = read_data(dir + filename + data.file_type)
		z_samples.append(z_matrix)
	return X, Y, z_samples


# use to draw 3 dimensional graph
def draw_3d_plot(x_mesh, y_mesh, z_mesh, name):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_surface(x_mesh, y_mesh, z_mesh, cmap='inferno')
	ax.set_xlabel('Emission, nm.')
	ax.set_ylabel('Excitation, nm.')
	ax.set_zlabel('Intencity')
	ax.set_xlabel(name)


# use to draw contour graph
def draw_contour(x_mesh, y_mesh, z_mesh, name):
	fig = plt.figure()
	ax = fig.add_subplot()
	cs = ax.contourf(x_mesh, y_mesh, z_mesh, cmap='inferno', levels=60)
	add_rects(ax)
	ax.set_xlabel('Emission, nm.')
	ax.set_ylabel('Excitation, nm.')
	ax.set_xlabel(name)
	plt.colorbar(cs)


if __name__ == '__main__':
	path = data.path_Kivu
	files = data.files_Kivu

	files_list = []
	for elem in files:
		files_list.append(elem)
	# getting data from them
	x, y, z_samples, t_samples = get_data(files_list, path)
	for i in range(1, 5):
		decomposition, trial_e1rrs = parafac2(np.transpose(z_samples[1:5]), i, return_errors=True, tol=1e-8,
			n_iter_max=500)
		est_weights, (est_A, est_B, est_C) = tl.parafac2_tensor.apply_parafac2_projections(decomposition)
		est_A, est_projected_Bs, est_C = tl.parafac2_tensor.apply_parafac2_projections(decomposition)[1]

		sign = np.sign(est_A)
		est_A = np.abs(est_A)
		est_projected_Bs = sign[:, np.newaxis] * est_projected_Bs

		est_A_normalised = est_A / la.norm(est_A, axis=0)
		est_Bs_normalised = [est_B / la.norm(est_B, axis=0)
		                     for est_B in est_projected_Bs]
		est_C_normalised = est_C / la.norm(est_C, axis=0)

		fig, axes = plt.subplots(1, 1, figsize=(15, 3 * 3 + 1))
		axes.plot(est_A, '--', label='Estimated')
		fig.suptitle(path + str(i))
		plt.show()
