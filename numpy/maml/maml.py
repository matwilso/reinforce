#!/usr/bin/env python3
import copy
import random
import numpy as np
import argparse

# make it possible to import from ../../utils/
import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from utils.optim import adam
from utils.common import GradDict # just automate some element checking (overkill)
from utils.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array, rel_error

from data_generator import DataGenerator

# TODO: probably add some plotting or something that shows that it actually 
# works, rather than just the loss. Basically add a test. 

# TODO: how would I adapt this to be able to take more than one gradient step 



class AdamOptimizer():
    def __init__(self, params, learning_rate=1e-3):
        # Configuration for Adam optimization
        self.optimization_config = {'learning_rate': learning_rate}
        self.adam_configs = {}
        for p in params:
            d = {k: v for k, v in self.optimization_config.items()}
            self.adam_configs[p] = d

    def apply_gradients(self, params, grads):
        for p in params: 
            next_w, self.adam_configs[p] = adam(params[p], grads[p], config=self.adam_configs[p])
            params[p] = next_w

def build_params(hidden_dim=[200]):
    # Initialize all weights (model params) with "Xavier Initialization" 
    # weight matrix init = uniform(-1, 1) / sqrt(layer_input)
    # bias init = zeros()
    d = {}
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
    def __init__(self, alpha=0.01):
        self.ALPHA = alpha

    def meta_forward(self, x_a, x_b, label_a, params, cache=None):
        p = params
        W1, b1, W2, b2 = p['W1'], p['b1'], p['W2'], p['b2']

        # standard forward and backward computations
        # (a)
        affine1_a = x_a.dot(W1) + b1
        relu1_a = np.maximum(0, affine1_a)
        affine2_a = relu1_a.dot(W2) + b2 
        pred_a = affine2_a

        dout_a = 2*(pred_a - label_a)

        drelu1_a = dout_a.dot(W2.T)
        dW2 = relu1_a.T.dot(dout_a)
        db2 = np.sum(dout_a, axis=0)

        daffine1_a = np.where(affine1_a > 0, drelu1_a, 0)

        dW1 = x_a.T.dot(daffine1_a)
        db1 = np.sum(daffine1_a, axis=0)

        # Forward on fast weights
        # (b)

        # grad steps
        W1_prime = W1 - self.ALPHA*dW1
        b1_prime = b1 - self.ALPHA*db1
        W2_prime = W2 - self.ALPHA*dW2
        b2_prime = b2 - self.ALPHA*db2

        affine1_b = x_b.dot(W1_prime) + b1_prime
        relu1_b = np.maximum(0, affine1_b)
        affine2_b = relu1_b.dot(W2_prime) + b2_prime
        pred_b = affine2_b

        if cache:
            cache = affine1_a, relu1_a, affine2_a, pred_a, dout_a, dW2, db2, dW1, db1, W1_prime, b1_prime, W2_prime, b2_prime, affine1_b, relu1_b, affine2_b, pred_b, label_a, x_a, x_b
            return pred_b, cache
        else:
            return pred_b
    
    def meta_backward(self, dout_b, params, cache, grads=None):
        affine1_a, relu1_a, affine2_a, pred_a, dout_a, dW2, db2, dW1, db1, W1_prime, b1_prime, W2_prime, b2_prime, affine1_b, relu1_b, affine2_b, pred_b, label_a, x_a, x_b = cache
        p = params
        W1, b1, W2, b2 = p['W1'], p['b1'], p['W2'], p['b2']

        # deriv w.r.t b (lower half)
        # d 1st layer
        dW2_prime = relu1_b.T.dot(dout_b)
        db2_prime = np.sum(dout_b, axis=0)
        drelu1_b = dout_b.dot(W2_prime.T)

        daffine1_b = np.where(affine1_b > 0, drelu1_b, 0)
        # d 2nd layer
        dW1_prime = x_b.T.dot(daffine1_b)
        db1_prime = np.sum(daffine1_b, axis=0)

        # deriv w.r.t a (upper half)

        # going back through the gradient descent step
        dW1 = dW1_prime
        db1 = db1_prime
        dW2 = dW2_prime
        db2 = db2_prime

        ddW1 = dW1_prime * -self.ALPHA
        ddb1 = db1_prime * -self.ALPHA
        ddW2 = dW2_prime * -self.ALPHA
        ddb2 = db2_prime * -self.ALPHA

        # backpropping through the first backprop
        ddout_a = relu1_a.dot(ddW2)
        ddout_a += ddb2
        drelu1_a = dout_a.dot(ddW2.T) # shortcut back because of the grad dependency

        ddaffine1_a = x_a.dot(ddW1) 
        ddaffine1_a += ddb1
        ddrelu1_a = np.where(affine1_a > 0, ddaffine1_a, 0)

        dW2 += ddrelu1_a.T.dot(dout_a)

        ddout_a += ddrelu1_a.dot(W2)

        dpred_a = ddout_a * 2 # = dout_a

        dW2 += relu1_a.T.dot(dpred_a)
        db2 += np.sum(dpred_a, axis=0)

        drelu1_a += dpred_a.dot(W2.T)

        daffine1_a = np.where(affine1_a > 0, drelu1_a, 0)

        dW1 += x_a.T.dot(daffine1_a)
        db1 += np.sum(daffine1_a, axis=0)

        if grads is not None:
            # update gradients 
            grads['W1'] = dW1
            grads['b1'] = db1
            grads['W2'] = dW2
            grads['b2'] = db2

   
def gradcheck():
    # Test the network gradient 
    nn = Network()
    grads = GradDict()

    np.random.seed(231)
    x_a = np.random.randn(15, 1)
    x_b = np.random.randn(15, 1)
    label = np.random.randn(15, 1)
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

    dW1_num = eval_numerical_gradient_array(lambda w: nn.meta_forward(x_a, x_b, label, rep_param(params, 'W1', w)), W1, dout)
    db1_num = eval_numerical_gradient_array(lambda b: nn.meta_forward(x_a, x_b, label, rep_param(params, 'b1', b)), b1, dout)
    dW2_num = eval_numerical_gradient_array(lambda w: nn.meta_forward(x_a, x_b, label, rep_param(params, 'W2', w)), W2, dout)
    db2_num = eval_numerical_gradient_array(lambda b: nn.meta_forward(x_a, x_b, label, rep_param(params, 'b2', b)), b2, dout)

    out, cache = nn.meta_forward(x_a, x_b, label, params, cache=True)
    nn.meta_backward(dout, params, cache, grads)

    # The error should be around 1e-10
    print()
    print('Testing meta-learning NN backward function:')
    print('dW1 error: ', rel_error(dW1_num, grads['W1']))
    print('db1 error: ', rel_error(db1_num, grads['b1']))
    print('dW2 error: ', rel_error(dW2_num, grads['W2']))
    print('db2 error: ', rel_error(db2_num, grads['b2']))
    print()

def main():
    nn = Network()
    params = build_params()
    optimizer = AdamOptimizer(params, learning_rate=FLAGS.learning_rate)
    # update_batch * 2, meta batch size
    data_generator = DataGenerator(10, 25)

    for itr in range(100000):
        batch_x, batch_y, amp, phase = data_generator.generate()

        inputa = batch_x[:, :5, :]
        labela = batch_y[:, :5, :]
        inputb = batch_x[:, 5:, :] # b used for testing
        labelb = batch_y[:, 5:, :]
        
        # META BATCH
        grads = GradDict() # zero grads
        losses = []
        for b in range(len(inputa)):
            ia, la, ib, lb = inputa[b], labela[b], inputb[b], labelb[b]
            pred_b, cache = nn.meta_forward(ia, ib, la, params, cache=True)
            losses.append((pred_b - lb)**2)
            dout_b = 2*(pred_b - lb)
            nn.meta_backward(dout_b, params, cache, grads)
        optimizer.apply_gradients(params, grads)
        if itr % 100 == 0:
            print("[itr: {}] Loss = {}".format(itr, np.sum(losses)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAML')
    parser.add_argument('--gradcheck', type=int, default=0, help='Run gradient check and other tests')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    FLAGS = parser.parse_args()
    
    if FLAGS.gradcheck:
        test()
    main()


