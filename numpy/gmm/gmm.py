import numpy as np
from scipy.stats import multivariate_normal

class GMM(object):
    def __init__(self):
        self.weights = None
        self.mu = None
        self.Sigma = None

    def _compute_ll(self, data):
        ll = 0
        for i in range(self.N):
            log_sum = 0
            for j in range(self.K):
                mvar = multivariate_normal(self.mu[j], self.Sigma[j])
                log_sum += self.weights[j] * mvar.pdf(data[i])
            ll += np.log(log_sum)

        return ll


    def _initialize_params(self, data, K, init='kmeans'):
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

        if init == 'kmeans':
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

    def _estep(self, data):
        # E-step
        N = data.shape[0]

        # gamma[i][j] is estimated probability of ith sample belonging to jth Gaussian
        gamma = np.zeros([self.K, N])
        for j in range(self.K):
            mvar = multivariate_normal(self.mu[j], self.Sigma[j])
            gamma[j, :] = self.weights[j] * mvar.pdf(data)
        # normalize 
        row_sums = gamma.sum(axis=0)
        gamma = gamma / row_sums[np.newaxis, :]

        return gamma


    def fit(self, data, K, tolerance=1e-5, max_iterations=100, init='kmeans'):
        """
        data is (N, D) measurements
        K is number of clusters
        """
        self.K = K
        if self.Sigma is None or self.Sigma.shape[0] != self.K:
            self._initialize_params(data, self.K, init=init)

        prev_ll = self._compute_ll(data)

        for _ in range(max_iterations):

            gamma = self._estep(data)
            n_list = np.sum(gamma, 1) 

            # M-step (compute the actual updates, based on probs computed in E-step)
            # cluster weights are updated according to ~how well all the data fits
            weight_update = n_list / np.sum(n_list)  # shape = (K,)

            n_inv = (1.0/n_list)[:, np.newaxis] # inv of unnormalized cluster probs
            # Derivation for this is given in references section of this repo
            mu_update =  n_inv * np.dot(gamma, data) 

            Sigma_update = np.zeros([self.K, self.D, self.D])

            # TODO: optimize this
            for j in range(self.K):
                for i in range(self.N):
                    diff = (data[i] - mu_update[j])[np.newaxis, :]
                    Sigma_update[j, :, :] += gamma[j][i]*(diff.T.dot(diff))

            for j in range(self.K):
                Sigma_update[j, :, :] *= 1.0/n_list[j]

            self.weights = weight_update
            self.mu = mu_update
            self.Sigma = Sigma_update + np.eye(self.D)*2e-6

            curr_ll = self._compute_ll(data)
            #print(curr_ll)
            if ( np.abs(curr_ll - prev_ll) < np.abs(prev_ll*tolerance)):
                break
            assert(curr_ll >= prev_ll or np.isclose(curr_ll, prev_ll))
            prev_ll = curr_ll


        return curr_ll, gamma

    def predict(self, pts):
        """
        Return parameters for Normal-Inverse-Wishart Prior, based on some data pts
        The Wishart is used as a prior for Multivariate Gaussian
        """
        # Get posterior of GMM given the data points
        # Get probabilities of samples falling in each cluster (kind of a posterior)
        gamma = self._estep(pts)
        cluster_wts = (np.sum(gamma, 0) / np.sum(gamma))[:, np.newaxis]
        # mu0 is mean of all other means (for Wishart)
        mu0 = np.sum(cluster_wts * self.mu)

        # Compute overall covariance.
        diff = self.mu - np.expand_dims(mu0, axis=0)
        diff_expand = np.expand_dims(self.mu, axis=1) * \
                np.expand_dims(diff, axis=2)
        wts_expand = np.expand_dims(cluster_wts, axis=2)
        Phi = np.sum((self.Sigma + diff_expand) * wts_expand, axis=0)

        # Set hyperparameters.
        m = 1
        n0 = 1
        return mu0, Phi, m, n0



    def eval(self, pts, axis=None):
        """
        Computes neg log probs of a given set of data points, evaluated at the
        current fit (see dev notes, contour plotting for usage)
        """
        prob = np.zeros(pts.shape[0])
        for j in range(self.K):
            if axis is not None:
                s = 2*axis
                e = 2*(axis+1)
                mvar = multivariate_normal(self.mu[j,s:e], self.Sigma[j,s:e,s:e])
            else:
                mvar = multivariate_normal(self.mu[j], self.Sigma[j])
            prob += self.weights[j]*mvar.pdf(pts)
        return -np.log(prob)


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
    print(gmm.fit(data, K=5))


