
# Gaussian Mixture Model

## Resources

I found this one to be most helpful: <br>
https://www2.ee.washington.edu/techsite/papers/documents/UWEETR-2010-0002.pdf

http://cs229.stanford.edu/notes/cs229-notes7b.pdf

https://people.eecs.berkeley.edu/~pabbeel/cs287-fa13/slides/Likelihood_EM_HMM_Kalman.pdf

NOTE: this page is pretty rough, still in progress


## About

A Gaussian Mixture Model (GMM) is a method used to cluster a set of data based
on the assumption that it can be described by K different Gaussian distributions,
each parameterized by a mean and covariance. 

Intuitively, it is similar to other clustering algorithms, like [k-means](https://en.wikipedia.org/wiki/K-means_clustering), that seek to bunch up data together and
draw boundaries.
However, it tends to be used more in practice in fields like robotics because it
uses Gaussians, which represent a lot of natural distributions richly (Central 
Limit Theorem), and are easier to work with for computing probabilties and 
using Bayesian methods.

Here is an example of such a cluster:

TODO: picture of GMM


## Parameterization: what values the GMM is learning

A Gaussian Mixture Model is parameterized by a set of K different Gaussians.  
A standard Gaussian is parameterized by mean <img src="https://rawgit.com/matwilso/implementations/master/numpy/gmm/svgs/07617f9d8fe48b4a7b3f523d6730eef0.svg?invert_in_darkmode" align=middle width=9.86799pt height=14.10255pt/> and covariance <img src="https://rawgit.com/matwilso/implementations/master/numpy/gmm/svgs/813cd865c037c89fcdc609b25c465a05.svg?invert_in_darkmode" align=middle width=11.82786pt height=22.38192pt/>. A 
GMM adds to this a weight <img src="https://rawgit.com/matwilso/implementations/master/numpy/gmm/svgs/31fae8b8b78ebe01cbfbe2fe53832624.svg?invert_in_darkmode" align=middle width=12.165285pt height=14.10255pt/> for each Gaussian, where the weight represents 
how likely that Gaussian is.  If there are a lot of datapoints in one cluster 
(for example if many points are bunched together), the weight of that cluster 
will he higher because it has more data points.

So we get K clusters each with a <img src="https://rawgit.com/matwilso/implementations/master/numpy/gmm/svgs/3adfd2dcb214fba13f8e187a1ce6bda0.svg?invert_in_darkmode" align=middle width=61.17804pt height=24.56553pt/>, and we start the algorithm 
with a bad guess of what these values are, and gradually learn values for these
that best match the data, using an iterative process.

## Fitting the data

Gaussian Mixture Models use the Expectation-Maximization (EM) algorithm for
fitting the data, for choosing the parameters <img src="https://rawgit.com/matwilso/implementations/master/numpy/gmm/svgs/3adfd2dcb214fba13f8e187a1ce6bda0.svg?invert_in_darkmode" align=middle width=61.17804pt height=24.56553pt/> for the
clusters that give the highest probability of seeing the data.

Expectation-Maximization is an iterative method which starts with an arbitrary 
guess of the true values of, and gradually updates and changes these to better 
fit the data.

The EM algorithm is usually explained by its two eponymous steps, (E)xpection and (M)aximization.

### E-step

The E-step computes the probalities of the data fitting the current values of the
cluster parameters <img src="https://rawgit.com/matwilso/implementations/master/numpy/gmm/svgs/3adfd2dcb214fba13f8e187a1ce6bda0.svg?invert_in_darkmode" align=middle width=61.17804pt height=24.56553pt/>.  To compute these probabilities, this 
code computes the probability of each u sample <img src="https://rawgit.com/matwilso/implementations/master/numpy/gmm/svgs/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode" align=middle width=5.642109pt height=21.60213pt/> belonging to each cluster 
<img src="https://rawgit.com/matwilso/implementations/master/numpy/gmm/svgs/36b5afebdba34564d884d347484ac0c7.svg?invert_in_darkmode" align=middle width=7.6816575pt height=21.60213pt/> of <img src="https://rawgit.com/matwilso/implementations/master/numpy/gmm/svgs/d6328eaebbcd5c358f426dbea4bdbf70.svg?invert_in_darkmode" align=middle width=15.080505pt height=22.38192pt/>. 

This is done by forming the Multivariate Gaussian probability distribution 
based on our current values of <img src="https://rawgit.com/matwilso/implementations/master/numpy/gmm/svgs/07617f9d8fe48b4a7b3f523d6730eef0.svg?invert_in_darkmode" align=middle width=9.86799pt height=14.10255pt/> and <img src="https://rawgit.com/matwilso/implementations/master/numpy/gmm/svgs/813cd865c037c89fcdc609b25c465a05.svg?invert_in_darkmode" align=middle width=11.82786pt height=22.38192pt/>. In code, that just looks like:
```
# create a multivariate Gaussian with the parameters of the j Gaussian
# mu is shape (K, D), where D is the dimension of the data
# for a 2D Gaussian, mu[j] will be a 2 element array
# Sigma[j] is a DxD matrix

scipy.stats.multivariate_normal(mu[j], Sigma[j])

```

We then evaluate this probability distribution ([scipy.multivariate_normal.pdf()](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.multivariate_normal.html)) 
with each data sample and weight that relative to our current belief of how probable
it was to draw from that cluster's probability distribution.  This value is 
called gamma (<img src="https://rawgit.com/matwilso/implementations/master/numpy/gmm/svgs/11c596de17c342edeed29f489aa4b274.svg?invert_in_darkmode" align=middle width=9.388665pt height=14.10255pt/>) in the code.

Here is the full code for computing gamma:
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
based on our current beliefs of our parameters <img src="https://rawgit.com/matwilso/implementations/master/numpy/gmm/svgs/3adfd2dcb214fba13f8e187a1ce6bda0.svg?invert_in_darkmode" align=middle width=61.17804pt height=24.56553pt/>. We now
want to find updates for these that fit the data better. This is an optimization
problem, and we are trying to optimize the probability of seeing the data.
So how likely the data is based on our parameterization.


I don't quite understand it fully, so see
[here](https://www2.ee.washington.edu/techsite/papers/documents/UWEETR-2010-0002.pdf) for a more detailed derivation.

I just know we are trying to optimize this, where Z_i is cluster, y_i is our observation, and \theta is our parameters <img src="https://rawgit.com/matwilso/implementations/master/numpy/gmm/svgs/3adfd2dcb214fba13f8e187a1ce6bda0.svg?invert_in_darkmode" align=middle width=61.17804pt height=24.56553pt/>.

<p align="center"><img src="https://rawgit.com/matwilso/implementations/master/numpy/gmm/svgs/2662af3c467851625f8e1aa887e9e176.svg?invert_in_darkmode" align=middle width=277.7082pt height=22.046805pt/></p>


We want to maximize <img src="https://rawgit.com/matwilso/implementations/master/numpy/gmm/svgs/2a3ba2df44c8a0dbd36e4af47e378f88.svg?invert_in_darkmode" align=middle width=67.48401pt height=29.12679pt/>, with the constraint that
all weights must add up to 1 (because they are probabilities).

We write gradients for <img src="https://rawgit.com/matwilso/implementations/master/numpy/gmm/svgs/2a3ba2df44c8a0dbd36e4af47e378f88.svg?invert_in_darkmode" align=middle width=67.48401pt height=29.12679pt/>, with respect to all the
parameters, <img src="https://rawgit.com/matwilso/implementations/master/numpy/gmm/svgs/3adfd2dcb214fba13f8e187a1ce6bda0.svg?invert_in_darkmode" align=middle width=61.17804pt height=24.56553pt/>. We then set these to 0 to maximize, and solve
them to get: 

```
			# this will add gamma probabilities for each data point and normalize
            weight_update = n_list / np.sum(n_list)  # shape = (K,)

            n_inv = (1.0/n_list)[:, np.newaxis] # inv of unnormalized cluster probs
            # Derivation for this is given in references section of this repo
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
<p align="center"><img src="https://rawgit.com/matwilso/implementations/master/numpy/gmm/svgs/d99b8870fc2f073c79bec17091521bde.svg?invert_in_darkmode" align=middle width=257.74155pt height=22.046805pt/></p>
-->
