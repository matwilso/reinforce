
# Gaussian Mixture Model

## Resources

For understanding and using:
- [1] https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html

For implementing yourself and going deeper on the math: 
- [2] https://www2.ee.washington.edu/techsite/papers/documents/UWEETR-2010-0002.pdf
- [3] http://cs229.stanford.edu/notes/cs229-notes7b.pdf
- [4] https://people.eecs.berkeley.edu/~pabbeel/cs287-fa13/slides/Likelihood_EM_HMM_Kalman.pdf

## About

A Gaussian Mixture Model (GMM) is a method used to cluster a set of data based
on the assumption that it can be described by K different Gaussian distributions,
each parameterized by a mean and covariance.  Intuitively, this is similar to 
other clustering algorithms, like [k-means](https://en.wikipedia.org/wiki/K-means_clustering), 
that seek to bunch up data together and draw boundaries.

Here is an plot of a Gaussian Mixture Model clusterng of 2D data from [scikit-learn's implementation](http://scikit-learn.org/stable/modules/mixture.html).
There are 2 ground truth clusters of the data, and 2 components are chosen for the GMM.
The contours show equal probabilities.  Notice how the peaks are highest 
around the datapoint clusters, but there is also decent density between the 
two clusters.

![gmm](scipy_gmm.png)


While, the GMM can work well for clustering data, the more important aspect of 
the GMM--and why it is often used in practice for probabilistic methods in
fields like robotics--is that it is an algorithm for *density estimation*.

Each of the Gaussian clusters represent a pocket of probability mass, but combined
they represent a full probability density over the data.  These can be used to
generate new data or form conditional probabilities over the data to form predictions,
as in [Guided Policy Search](https://arxiv.org/pdf/1504.00702.pdf).  For better
understanding Gaussian Mixture Models, [[1]](https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html) 
is a highly recommended resource.

## Parameterization: what values is the GMM learning?

A Gaussian Mixture Model is parameterized by a set of K different Gaussians.  
A standard Gaussian is parameterized by mean <img alt="$\mu$" src="https://raw.githubusercontent.com/matwilso/reinforce/5c5a4064ccbf02ca56c44d86c51d5c3fbbc633df/numpy/gmm/svgs/07617f9d8fe48b4a7b3f523d6730eef0.svg?invert_in_darkmode" align=middle width="9.867990000000004pt" height="14.102549999999994pt"/> and covariance <img alt="$\Sigma$" src="https://raw.githubusercontent.com/matwilso/reinforce/5c5a4064ccbf02ca56c44d86c51d5c3fbbc633df/numpy/gmm/svgs/813cd865c037c89fcdc609b25c465a05.svg?invert_in_darkmode" align=middle width="11.827860000000003pt" height="22.381919999999983pt"/>. A 
mixture model of Gaussians adds to this parameterization a weight <img alt="$w$" src="https://raw.githubusercontent.com/matwilso/reinforce/5c5a4064ccbf02ca56c44d86c51d5c3fbbc633df/numpy/gmm/svgs/31fae8b8b78ebe01cbfbe2fe53832624.svg?invert_in_darkmode" align=middle width="12.165285pt" height="14.102549999999994pt"/> that represents
how likely that Gaussian is.  If there are a lot of datapoints in one cluster (
for example if many points are bunched together), the weight of that cluster 
will he higher because it has more data points.

So we get K clusters each with a <img alt="$(w, \mu, \Sigma)$" src="https://raw.githubusercontent.com/matwilso/reinforce/5c5a4064ccbf02ca56c44d86c51d5c3fbbc633df/numpy/gmm/svgs/3adfd2dcb214fba13f8e187a1ce6bda0.svg?invert_in_darkmode" align=middle width="61.17804pt" height="24.56552999999997pt"/>, and we start the algorithm 
with a bad guess of what these values are, and gradually learn values for these
that best match the data.

## Fitting the data

Gaussian Mixture Models use the Expectation-Maximization (EM) algorithm for
fitting the data -- for choosing the parameters <img alt="$(w, \mu, \Sigma)$" src="https://raw.githubusercontent.com/matwilso/reinforce/5c5a4064ccbf02ca56c44d86c51d5c3fbbc633df/numpy/gmm/svgs/3adfd2dcb214fba13f8e187a1ce6bda0.svg?invert_in_darkmode" align=middle width="61.17804pt" height="24.56552999999997pt"/> so that 
the clusters give the highest likelihood for the data. If you are familiar 
with Maximum Likelihood estimators, this algorithm is a way to approximate
this for the GMM.

Expectation-Maximization is an iterative method which starts with an arbitrary 
guess of the true values of, and gradually updates and changes these to better 
fit the data.

The EM algorithm is usually explained by its two eponymous steps, (E)xpection and (M)aximization.

### E-step

The E-step computes the probalities of the data fitting the current values of the
cluster parameters <img alt="$(w, \mu, \Sigma)$" src="https://raw.githubusercontent.com/matwilso/reinforce/5c5a4064ccbf02ca56c44d86c51d5c3fbbc633df/numpy/gmm/svgs/3adfd2dcb214fba13f8e187a1ce6bda0.svg?invert_in_darkmode" align=middle width="61.17804pt" height="24.56552999999997pt"/>.  To compute these probabilities, my 
code computes the probability of each u sample <img alt="$i$" src="https://raw.githubusercontent.com/matwilso/reinforce/5c5a4064ccbf02ca56c44d86c51d5c3fbbc633df/numpy/gmm/svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width="5.642109000000004pt" height="21.602129999999985pt"/> belonging to each cluster 
<img alt="$j$" src="https://raw.githubusercontent.com/matwilso/reinforce/5c5a4064ccbf02ca56c44d86c51d5c3fbbc633df/numpy/gmm/svgs/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode" align=middle width="7.681657500000003pt" height="21.602129999999985pt"/> of <img alt="$K$" src="https://raw.githubusercontent.com/matwilso/reinforce/5c5a4064ccbf02ca56c44d86c51d5c3fbbc633df/numpy/gmm/svgs/d6328eaebbcd5c358f426dbea4bdbf70.svg?invert_in_darkmode" align=middle width="15.080505pt" height="22.381919999999983pt"/>. 

This is done by forming the Multivariate Gaussian probability distribution 
based on our current values of <img alt="$\mu$" src="https://raw.githubusercontent.com/matwilso/reinforce/5c5a4064ccbf02ca56c44d86c51d5c3fbbc633df/numpy/gmm/svgs/07617f9d8fe48b4a7b3f523d6730eef0.svg?invert_in_darkmode" align=middle width="9.867990000000004pt" height="14.102549999999994pt"/> and <img alt="$\Sigma$" src="https://raw.githubusercontent.com/matwilso/reinforce/5c5a4064ccbf02ca56c44d86c51d5c3fbbc633df/numpy/gmm/svgs/813cd865c037c89fcdc609b25c465a05.svg?invert_in_darkmode" align=middle width="11.827860000000003pt" height="22.381919999999983pt"/>. In code, that just looks like:
```
# create a multivariate Gaussian with the parameters of the j Gaussian
# mu is shape (K, D), where D is the dimension of the data
# (i.e., for a 2D Gaussian, mu[j] will be a 2 element array)
# Sigma[j] is a DxD matrix

mvar = scipy.stats.multivariate_normal(mu[j], Sigma[j])

```

We then evaluate this probability distribution ([scipy.multivariate_normal.pdf()](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.multivariate_normal.html)) 
for each data sample, for each cluster, weighted by how likely it was to draw
from each cluster. This value is called gamma (<img alt="$\gamma$" src="https://raw.githubusercontent.com/matwilso/reinforce/5c5a4064ccbf02ca56c44d86c51d5c3fbbc633df/numpy/gmm/svgs/11c596de17c342edeed29f489aa4b274.svg?invert_in_darkmode" align=middle width="9.388665000000001pt" height="14.102549999999994pt"/>) in the code.

Here is the full (non-optimized) code for computing gamma:
```
        # gamma[i][j] is estimated probability of ith sample belonging to jth Gaussian
        gamma = np.zeros([self.N, self.K])
        for i in range(self.N):
            for j in range(self.K):
                mvar = multivariate_normal(self.mu[j], self.Sigma[j])
                gamma[i][j] = self.weights[j] * mvar.pdf(data[i])
            gamma[i, :] /= np.sum(gamma[i, :])

```

### M-step

The M-step does the actual optimization, or fitting of the clusters.

From the E-step, we got gamma: a matrix of how probable it is that the ith 
data sample fits the jth Gaussian, where the probabilty calculations were
based on our current beliefs of our parameters <img alt="$(w, \mu, \Sigma)$" src="https://raw.githubusercontent.com/matwilso/reinforce/5c5a4064ccbf02ca56c44d86c51d5c3fbbc633df/numpy/gmm/svgs/3adfd2dcb214fba13f8e187a1ce6bda0.svg?invert_in_darkmode" align=middle width="61.17804pt" height="24.56552999999997pt"/>. We now
want to find updates for these that fit the data better. This is an optimization
problem, and we are trying to optimize the probability of seeing the data.
So update our parameters so that they give better predictions of the data.

See [here](https://www2.ee.washington.edu/techsite/papers/documents/UWEETR-2010-0002.pdf) for a more detailed derivation, but below are the basics

#### Derivation

NOTE: this part is still very rough

We are trying to maximize the log probability.  It is a Maximum Likelhiood Estimation
method: finding the parameter values that maximize the likelihood of making 
the observations given the parameters

We are trying to optimize the equation below, where <img alt="$Z_i$" src="https://raw.githubusercontent.com/matwilso/reinforce/5c5a4064ccbf02ca56c44d86c51d5c3fbbc633df/numpy/gmm/svgs/ed35373880183d013fc1bba898b2e3ae.svg?invert_in_darkmode" align=middle width="15.813105000000002pt" height="22.381919999999983pt"/> is cluster, <img alt="$y_i$" src="https://raw.githubusercontent.com/matwilso/reinforce/5c5a4064ccbf02ca56c44d86c51d5c3fbbc633df/numpy/gmm/svgs/2b442e3e088d1b744730822d18e7aa21.svg?invert_in_darkmode" align=middle width="12.662925000000003pt" height="14.102549999999994pt"/> is our observation, and <img alt="$\theta$" src="https://raw.githubusercontent.com/matwilso/reinforce/5c5a4064ccbf02ca56c44d86c51d5c3fbbc633df/numpy/gmm/svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode" align=middle width="8.143030500000002pt" height="22.745910000000016pt"/> is our parameters <img alt="$(w, \mu, \Sigma)$" src="https://raw.githubusercontent.com/matwilso/reinforce/5c5a4064ccbf02ca56c44d86c51d5c3fbbc633df/numpy/gmm/svgs/3adfd2dcb214fba13f8e187a1ce6bda0.svg?invert_in_darkmode" align=middle width="61.17804pt" height="24.56552999999997pt"/>.

<p align="center"><img alt="$$Q_i(\theta | \theta^{(m)} = E_{Z_i | y_i, \theta^{(m)}} [log p_X(y_i, Z_i | \theta)]$$" src="https://raw.githubusercontent.com/matwilso/reinforce/5c5a4064ccbf02ca56c44d86c51d5c3fbbc633df/numpy/gmm/svgs/2662af3c467851625f8e1aa887e9e176.svg?invert_in_darkmode" align=middle width="277.7082pt" height="22.046805pt"/></p>

We want to maximize <img alt="$Q_i(\theta | \theta^{(m)}$" src="https://raw.githubusercontent.com/matwilso/reinforce/5c5a4064ccbf02ca56c44d86c51d5c3fbbc633df/numpy/gmm/svgs/2a3ba2df44c8a0dbd36e4af47e378f88.svg?invert_in_darkmode" align=middle width="67.48401pt" height="29.12679000000001pt"/>, with the constraint that
all weights must add up to 1 (because they are probabilities).

We write gradients for <img alt="$Q_i(\theta | \theta^{(m)}$" src="https://raw.githubusercontent.com/matwilso/reinforce/5c5a4064ccbf02ca56c44d86c51d5c3fbbc633df/numpy/gmm/svgs/2a3ba2df44c8a0dbd36e4af47e378f88.svg?invert_in_darkmode" align=middle width="67.48401pt" height="29.12679000000001pt"/>, with respect to all the
parameters, <img alt="$(w, \mu, \Sigma)$" src="https://raw.githubusercontent.com/matwilso/reinforce/5c5a4064ccbf02ca56c44d86c51d5c3fbbc633df/numpy/gmm/svgs/3adfd2dcb214fba13f8e187a1ce6bda0.svg?invert_in_darkmode" align=middle width="61.17804pt" height="24.56552999999997pt"/>. We then set these to 0 to maximize, and solve
them to get (final code): 

```
            n_list = np.sum(gamma, 1) 
			# this will add gamma probabilities for each data point and normalize
            weight_update = n_list / np.sum(n_list)  # shape = (K,)

            n_inv = (1.0/n_list)[:, np.newaxis] # inv of unnormalized cluster probs
            mu_update =  n_inv * np.dot(gamma.T, data) 

            Sigma_update = np.zeros([self.K, self.D, self.D])

```

<!--
Specifically, for each data sample that we get, we want to maximize the 
probability a

seeing the observation, cluso

We want to maximize the Expectation.
This amounts to wanting the clusters that fit the data points better to also be
the ones that.

We want cluster

we are trying to maximize the Expectation of the probability of 
-->




<!--
The for EM generally is:
<p align="center"><img alt="$$Q_i(\theta | \theta^{(m)}) = E_{X_i|y_i,\theta^{(m)}}[log p(X_i, \theta)]$$" src="https://raw.githubusercontent.com/matwilso/reinforce/5c5a4064ccbf02ca56c44d86c51d5c3fbbc633df/numpy/gmm/svgs/d99b8870fc2f073c79bec17091521bde.svg?invert_in_darkmode" align=middle width="257.74154999999996pt" height="22.046805pt"/></p>
-->
