#!/usr/bin/env python3
import copy
import numpy as np
import argparse

# make it possible to import from ../../utils/
import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from utils.optim import adam
from utils.common import CacheDict, GradDict # just automate some element checking (overkill)
from utils.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array, rel_error

"""
TODO:
- maybe implement batch norm (hopefully the method does not require it to work

Notes:

probably want to use extra dicts instead of prefixed weights

where do I need to compute 2nd derivatives?
...
I can use tf.hessians() to figure out how to do Hessians



alright, so we need to compute the gradient w.r.t the loss ()


Maybe start with the standard network update.  Alright, I am going
to start it slow, and just do the inner step right now.  I am going
to try to pull things out as much as possible.

And then after that, think about what I need to compute to get the 2nd
deriv


may want to build a separate class for optimization

"""

def build_params(name, hidden_dim=[200]):
    # Initialize all weights (model params) with "Xavier Initialization" 
    # weight matrix init = uniform(-1, 1) / sqrt(layer_input)
    # bias init = zeros()
    d = {}
    d['name'] = name
    d['W1'] = (-1 + 2*np.random.rand(1, hidden_dim[0])) / np.sqrt(1)
    d['b1'] = np.zeros(hidden_dim[0])
    d['W2'] = (-1 + 2*np.random.rand(hidden_dim[0], 1)) / np.sqrt(hidden_dim[0])
    d['b2'] = np.zeros(1)

    # Cast all parameters to the correct datatype
    for k, v in d.items():
        d[k] = v.astype(np.float32)
    return d

def zero_grads(self, grads):
    """Reset gradients to 0. This should be called during optimization steps"""
    for g in grads:
        grads[g] = np.zeros_like(grads[g])


class Network(object):
    """BYOW: Bring Your Own Weights

    Hard-code a 3 layer NN operations
    
    """
    def __init__(self):
        pass
        ## Configuration for Adam optimization
        #self.optimization_config = {'learning_rate': 1e-3}
        #if adam:
        #    self.adam_configs = {}
        #    for p in self.params:
        #        d = {k: v for k, v in self.optimization_config.items()}
        #        self.adam_configs[p] = d
        #else:
        #    self.adam_configs = None

    def forward(self, x, params, cache=None):
        p = params
        W1, b1, W2, b2 = p['W1'], p['b1'], p['W2'], p['b2']

        # forward computations
        affine1 = x.dot(W1) + b1
        relu1 = np.maximum(0, affine1)
        pred = affine2 = relu1.dot(W2) + b2 
        
        if cache is not None:
            # cache values for backward (based on what is needed for analytic gradient calc)
            cache['fwd_x'].append(x)
            cache['fwd_affine1'].append(affine1)
            cache['fwd_relu1'].append(relu1)

        return pred
    
    def backward(self, dout, params, cache, grads):
        p = params
        W1, b1, W2, b2 = p['W1'], p['b1'], p['W2'], p['b2']

        # get values from network forward passes (for analytic gradient computations)
        fwd_relu1 = np.concatenate(cache['fwd_relu1'])
        fwd_affine1 = np.concatenate(cache['fwd_affine1'])
        fwd_x = np.concatenate(cache['fwd_x'])

        # d 2nd layer 
        drelu1 = dout.dot(W2.T)
        dW2 = fwd_relu1.T.dot(dout)
        db2 = np.sum(dout, axis=0)

        daffine1 = np.where(fwd_affine1 > 0, drelu1, 0)

        # d 1st layer
        dW1 = fwd_x.T.dot(daffine1)
        db1 = np.sum(daffine1, axis=0)

        # update gradients 
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2

def test():
    # Test the network gradient 
    nn = Network()
    grads = GradDict()
    cache = CacheDict()

    np.random.seed(231)
    x = np.random.randn(15, 1)
    W1 = np.random.randn(1, 40)
    b1 = np.random.randn(40)
    W2 = np.random.randn(40, 1)
    b2 = np.random.randn(1)

    dout = np.random.randn(15, 1)

    params = p = {}
    p['W1'] = W1
    p['b1'] = b1
    p['W2'] = W2
    p['b2'] = b2

    def rep_param(params, name, val):
        clean_params = copy.deepcopy(params)
        clean_params[name] = val
        return clean_params

    dW1_num = eval_numerical_gradient_array(lambda w: nn.forward(x, rep_param(params, 'W1', w)), W1, dout)
    db1_num = eval_numerical_gradient_array(lambda b: nn.forward(x, rep_param(params, 'b1', b)), b1, dout)
    dW2_num = eval_numerical_gradient_array(lambda w: nn.forward(x, rep_param(params, 'W2', w)), W2, dout)
    db2_num = eval_numerical_gradient_array(lambda b: nn.forward(x, rep_param(params, 'b2', b)), b2, dout)

    out = nn.forward(x, params, cache)
    nn.backward(dout, params, cache, grads)

    # The error should be around 1e-10
    print()
    print('Testing affine_backward function:')
    print('dW1 error: ', rel_error(dW1_num, grads['W1']))
    print('db1 error: ', rel_error(db1_num, grads['b1']))
    print('dW2 error: ', rel_error(dW2_num, grads['W2']))
    print('db2 error: ', rel_error(db2_num, grads['b2']))
    print()


def main():
    nn = Network()
    grads = GradDict()
    cache = CacheDict()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAML')
    parser.add_argument('--test', type=int, default=0, help='Run gradient check and other tests')
    FLAGS = parser.parse_args()
    
    if FLAGS.test:
        test()
    main()


