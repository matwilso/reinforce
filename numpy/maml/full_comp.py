# TODO: may want to rewrite the primes as just another dictionary, so its easier
# TODO: think about why doing this makes sense. how did they come up with this?
# TODO: a gradient check on the meta pass
# TODO: go over this again.  check the diagram and do another pass

# NOTE: junctions add on the way back
# NOTE: first pass is a, second pass is b

import copy
import numpy as np

# make it possible to import from ../../utils/
import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from utils.optim import adam
from utils.common import CacheDict, GradDict # just automate some element checking (overkill)
from utils.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array, rel_error

ALPHA = 0.01

def meta_forward(x_a, x_b, label_a, params, cache=False):
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
    W1_prime = W1 - ALPHA*dW1
    b1_prime = b1 - ALPHA*db1
    W2_prime = W2 - ALPHA*dW2
    b2_prime = b2 - ALPHA*db2

    affine1_b = x_b.dot(W1_prime) + b1_prime
    relu1_b = np.maximum(0, affine1_b)
    affine2_b = relu1_b.dot(W2_prime) + b2_prime
    pred_b = affine2_b


    # just cache
    if cache:
        #cache = dict(affine1_a=affine1_a, relu1_a=relu1_a, affine2_a=affine2_a, pred_a=pred_a, dout_a=dout_a, dW2=dW2, db2=db2, daffine1_a=daffine1_a, dW1=dW1, db1=db1, W1_prime=W1_prime, b1_prime=b1_prime, W2_prime=W2_prime, b2_prime=b2_prime, affine1_b=affine1_b, relu1_b=relu1_b, affine2_b=affine2_b, pred_b=pred_b,)
        cache = affine1_a, relu1_a, affine2_a, pred_a, dout_a, dW2, db2, dW1, db1, W1_prime, b1_prime, W2_prime, b2_prime, affine1_b, relu1_b, affine2_b, pred_b, label_a, x_a
        return pred_b, cache
    else:
        return pred_b

def meta_backward(dout_b, params, cache, grads=None):
    affine1_a, relu1_a, affine2_a, pred_a, dout_a, dW2, db2, dW1, db1, W1_prime, b1_prime, W2_prime, b2_prime, affine1_b, relu1_b, affine2_b, pred_b, label_a, x_a = cache
    p = params
    W1, b1, W2, b2 = p['W1'], p['b1'], p['W2'], p['b2']

    # deriv w.r.t b (lower half)
    # d 1st layer
    drelu1_b = dout_b.dot(W2_prime.T)
    dW2_prime = relu1_b.T.dot(dout_b)
    db2_prime = np.sum(dout_b, axis=0)

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

    ddW1 = dW1_prime * -ALPHA
    ddb1 = db1_prime * -ALPHA
    ddW2 = dW2_prime * -ALPHA
    ddb2 = db2_prime * -ALPHA

    # backpropping through the first backprop
    ddout_a = relu1_a.dot(ddW2)
    ddout_a += ddb2
    drelu1_a = dout_a.dot(ddW2.T) # shortcut back because of the grad dependency

    # this gradient actually gets killed because the 2nd deriv of relu is 0 everywhere
    #ddafine1_a = x_a.dot(ddW1.T) # TODO: check why this was wrong shape
    #ddafine1_a += ddb1

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

if __name__ == '__main__':
    # Test the network gradient 
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

    #import ipdb; ipdb.set_trace()
    dW1_num = eval_numerical_gradient_array(lambda w: meta_forward(x_a, x_b, label, rep_param(params, 'W1', w)), W1, dout)
    db1_num = eval_numerical_gradient_array(lambda b: meta_forward(x_a, x_b, label, rep_param(params, 'b1', b)), b1, dout)
    dW2_num = eval_numerical_gradient_array(lambda w: meta_forward(x_a, x_b, label, rep_param(params, 'W2', w)), W2, dout)
    db2_num = eval_numerical_gradient_array(lambda b: meta_forward(x_a, x_b, label, rep_param(params, 'b2', b)), b2, dout)

    out, cache = meta_forward(x_a, x_b, label, params, cache=True)
    meta_backward(dout, params, cache, grads)

    # The error should be around 1e-10
    print()
    print('Testing affine_backward function:')
    print('dW1 error: ', rel_error(dW1_num, grads['W1']))
    print('db1 error: ', rel_error(db1_num, grads['b1']))
    print('dW2 error: ', rel_error(dW2_num, grads['W2']))
    print('db2 error: ', rel_error(db2_num, grads['b2']))
    print()
