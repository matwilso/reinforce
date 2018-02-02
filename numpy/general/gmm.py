import numpy as np
from scipy.stats import multivariate_normal

class GMM(object):
    def __init__(self):
        pass

    def _initialize_params(self, data, K, method='kmeans_init'):
        """
        Initialize GMM weights, mean, and covariance

        The 0th step in the algorithm
        """
        self.N = N = data.shape[0] # number of samples
        self.D = D = data.shape[1] # number of dimensions
        self.K = K  

        self.mu = np.zeros([K, D]) # all cluster means
        self.Sigma = np.zeros([K, D, D]) # all cluster covariances
        self.weights = 1.0/self.K * np.ones(self.K) # probabilities for multinomial draw of K gaussians

        if method == 'kmeans_init':
            # Randomly select K data points to be the arbitrary cluster means 
            idxs = np.random.choice(np.arange(N), size=self.K, replace=False)
            cluster_means = data[idxs]

            # Find closest cluster mean to each data point and that is its cluster
            # (gives index of cluster for every data point)
            closest_mean = lambda pt: np.argmin(np.linalg.norm(pt-cluster_means, axis=1))
            clusters = np.apply_along_axis(closest_mean, 1, data)

            for k in range(K):
                cluster_pts = data[clusters == k]
                cluster_mean = np.mean(cluster_pts, 0)
                self.mu[k, :] = cluster_mean
                self.Sigma[k, :, :] = np.cov(cluster_pts.T) + np.eye(self.D)*2e-6
        else:
            raise NotImplementedError()

        return self.mu, self.Sigma

    def fit(self, data, K, init='kmeans_init'):
        """
        data is (N, D) measurements
        K is number of clusters
        """
        self._initialize_params(data, K, method=init)




if __name__ == '__main__':
    # dummy data with 3 cluster means
    np.random.seed(42)

    means = np.array([
             [0.0, 5.0],
             [5.0, 0.0],
             [-4.0, -1.0],
            ])
              
    covs = np.array([
            [[5.0, 0.0],[0.0, 1.0]],
            [[1.0, -1.0],[-1.0, 3.0]],
            [[1.0, 2.0],[2.0, 6.0]]
           ])
    
    N = 100
    D = means.shape[1]
    K = means.shape[0]
    
    data = np.ndarray([N, D])
    for i in range(N):
        j = np.random.randint(0, K)
        data[i] = np.random.multivariate_normal(means[j], covs[j])

    gmm = GMM()
    gmm.fit(data, K=5)


