import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys

def get_data(path, max_length=10000):
	data = {
		"f_best": [],
		"f_mean": [],
		"f_stddev": [],
		"m_best": [],
		"m_mean": [],
		"m_stddev": []
	}

	for i in range(5):
		with open(path + "/trial" + str(i) + ".pkl", "rb") as f:
			trial = pickle.load(f)
			for key in trial:
				data[key].append(trial[key])

	for key in data:
		data[key] = np.stack(data[key])

	keys = ["f_best", "f_mean", "m_best", "m_mean"]
	final_data = {}

	for key in keys:
		mean = data[key].mean(axis=0)
		std = data[key].std(axis=0)
		final_data[key] = {
			"mean": mean,
			"std": std
		}

	return final_data

fixed_data = get_data("Experiments/layered-fixedgoal")
mvg_data_0 = get_data("Experiments/layered-fixedstruct-mvg")
mvg_data_1 = get_data("Experiments/layered-mvg") 
plt.plot(
	#fixed_data["f_mean"]["mean"], ".", 
	fixed_data["m_best"]["mean"], ".",
	#mvg_data_0["f_mean"]["mean"], ".", 
	mvg_data_0["m_best"]["mean"], ".",
	#mvg_data_1["f_mean"]["mean"], ".", 
	mvg_data_1["m_best"]["mean"], "."
)
plt.show()
