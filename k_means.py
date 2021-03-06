#Based on problem set of Stanford CS221
#http://stanford.edu/~cpiech/cs221/handouts/kmeans.html

def K_means(dataSet, k):
	numFeatures = dataSet.shape[1]
	def getRandomCentroids(numFeatures, dataSet, k):
		centroids_list = []
		for x in range(0, k):
			center = []
			for i in range(0, numFeatures):
				center.append(random.randint(dataSet[[i]].min(), dataSet[[i]].max()))
			centroids_list.append(center)
		return centroids_list
	def shouldStop(oldCentroids, centroids, iterations):
		if iterations > 10:
			return True
		return oldCentroids == centroids
	def getLabels(dataSet, centroids):
		centroids = pd.DataFrame(centroids)
		label_dict = {}
		for i in range(0, dataSet.shape[0]):
			x = dataSet.iloc[i]
			x_label = None
			x_dis = 100
			for j in range(0, centroids.shape[0]):
				if distance(x, centroids.iloc[j]) < x_dis:
					x_label = j
					x_dis = distance(x, j)
			label_dict[i] = x_label
		return label_dict
	def getCentroids(dataSet, labels, k):
		label_dict = labels
		label_col = np.transpose(pd.DataFrame(label_dict, index = [0]))
		labeled_data = pd.concat([dataSet,label_col], axis = 1)
		centroid_list = []
		for i in range(0,k):
			new = labeled_data.loc[labeled_data.iloc[:,numFeatures]== i]
			centroid_list.append(new.iloc[:,0:numFeatures].mean().tolist())
		return centroid_list
	def distance(x, y):
		return np.sqrt(sum((x - y) ** 2))
	centroids = getRandomCentroids(numFeatures, dataSet, k)
	iterations = 0
	oldCentroids = None
	while not shouldStop(oldCentroids, centroids, iterations):
		for x in centroids:
			if np.isnan(x).any():
				centroids = getRandomCentroids(numFeatures, dataSet, k)
		oldCentroids = centroids
		iterations += 1
		labels = getLabels(dataSet, centroids)
		centroids = getCentroids(dataSet, labels, k)
	return centroids
