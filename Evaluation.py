import math
def purity(prediction, ground_truth):
	clusters = set((value) for value in prediction.itervalues())

	purity = 0
	for cluster in clusters:
		stat = dict()
		for element in prediction.iterkeys():
			if prediction[element] == cluster:
				if stat.has_key(ground_truth[element]):
					stat[ground_truth[element]] += 1
				else:
					stat[ground_truth[element]] = 1
		purity += stat[max(stat.iterkeys(),key=lambda k:stat[k])]
	purity /= (float)(len(prediction))

	return purity

def entropy(prediction, ground_truth):
	clusters = set((value) for value in prediction.itervalues())

	entropy = 0
	for cluster in clusters:
		stat = dict()
		real = 0
		for element in prediction.iterkeys():
			if prediction[element] == cluster:
				if stat.has_key(ground_truth[element]):
					stat[ground_truth[element]] += 1
				else:
					stat[ground_truth[element]] = 1 
				real += 1

		for element in stat.iterkeys():
			entropy +=  (float)(stat[element]) * (float)(stat[element]) / real 
	entropy /= (float)(len(prediction)) 
	entropy = 1 - entropy
	return entropy 

