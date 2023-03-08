import numpy as np
from numpy.linalg import cond
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import copy
import os

def approx_cond(A, s, indices):
	sample_indices = np.sort(np.random.choice(indices, s, replace=True))
	AS = len(indices) * A[sample_indices][:, sample_indices] / s
	cond_num = cond(AS)
	return cond_num

def multiple_trials(A, max_size, steps, trials=50):
	indices = list(range(len(A)))
	sizes = list(range(10, max_size, steps))
	vals = np.zeros((len(sizes), trials))

	for i in tqdm(range(len(sizes))):
		s = sizes[i]
		interim_vec = []
		for j in range(trials):
			interim_vec.append(approx_cond(A, s, indices))
		vals[i,:] = np.array(interim_vec)
	return vals

def stat_computer(A, do_std=False, p1=20, p2=80):
	means = np.mean(A, axis=1)
	if do_std:
		stds = np.std(A, axis=1)
		percentile1 = means - stds
		percentile2 = means + stds
	else:
		percentile1 = np.percentile(A, p1, axis=1)
		percentile2 = np.percentile(A, p2, axis=1)

	return np.log(means), np.log(percentile1), np.log(percentile2)

def plot(vals, labels, x_axis, dataset_name):
	#plot_labels = labels[0:len(vals)]
	axis_labels = [labels[-4], labels[-3]]
	header = labels[-2]
	filename = labels[-1]
	
	plt.gcf().clear()
	plt.plot(x_axis, vals[0], alpha=1.0)
	plt.fill_between(x_axis, vals[1], vals[2], alpha=0.2)
	#plt.ylim((-1,10))
	plt.ylabel(axis_labels[1])
	plt.xlabel(axis_labels[0])
	plt.title(header)

	dirname = "./figures/"+dataset_name+"/cond_num/"
	if not os.path.isdir(dirname):
		os.makedirs(dirname)
	filename = dirname+filename
	plt.savefig(filename)
	

A = np.random.random((500, 380))
PSD = A@A.T
PSD = PSD / PSD.max()

A = np.random.random((500,500))
symmetric = (A + A.T) / 2

PSD_cond_num = cond(PSD)
symmetric_cond_num = cond(symmetric)
print(PSD_cond_num, symmetric_cond_num)

max_size = 500
steps = 10
trials = 10
percentiles = [20,80]
labels = ["log sampling rate", "log of errors", "Approximate condition number"]

approx_PSD_conds = multiple_trials(PSD, max_size, steps, trials)
approx_sym_conds = multiple_trials(symmetric, max_size, steps, trials)

error_PSD = approx_PSD_conds - PSD_cond_num
error_sym = approx_sym_conds - symmetric_cond_num
stats_PSD = stat_computer(error_PSD, True, percentiles[0], percentiles[1])
stats_sym = stat_computer(error_sym, True, percentiles[0], percentiles[1])

x_axis = np.log(list(range(10, max_size, steps)))

label1 = copy(labels)
label2 = copy(labels)

label1.append("PSD_approx_cond_num.pdf")
plot(stats_PSD, label1, x_axis, "random")
label2.append("sym_approx_cond_num.pdf")
plot(stats_sym, label2, x_axis, "random")


