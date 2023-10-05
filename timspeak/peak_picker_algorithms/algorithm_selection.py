
def smoothing_function_1():
	import timspeak.peak_picker_algorithms.smooth.smoothing_algorithm_1
	return timspeak.peak_picker_algorithms.smooth.smoothing_algorithm_1.Smoother

def smooth_algorithm(algorithm: str):
	smooth_algorithms = {
		'smoothing_algorithm_1': smoothing_function_1(),
	}
	return smooth_algorithms[algorithm]

def cluster_function_1():
	import timspeak.peak_picker_algorithms.cluster.clustering_algorithm_1
	return timspeak.peak_picker_algorithms.cluster.clustering_algorithm_1.Clusterer

def cluster_algorithm(algorithm: str):
	cluster_algorithms = {
		'clustering_algorithm_1': cluster_function_1(),
	}
	return cluster_algorithms[algorithm]
