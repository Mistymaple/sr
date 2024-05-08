from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------------

def vqlbg(d, k):
	# Reshape data if necessary
	d = d.T if d.shape[0] != len(d) else d
	
	# Create KMeans model
	kmeans = KMeans(n_clusters=k, random_state=0)
	
	# Fit the model to the data
	kmeans.fit(d)
	
	# Get cluster centers
	r = kmeans.cluster_centers_.T
	
	return r
