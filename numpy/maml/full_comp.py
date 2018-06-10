# TODO: may want to rewrite the primes as just another dictionary, so its easier
# first pass is a, second pass is b
# TODO: think about why doing this makes sense. how did they come up with this?
# TODO: a gradient check on the meta pass
# NOTE: junctions add on the way back
# TODO: go over this again.  check the diagram and do another pass

ALPHA = 0.01

def forward(xa, xb, params, cache=None):
    p = params
    W1, b1, W2, b2 = p['W1'], p['b1'], p['W2'], p['b2']

    # standard forward and backward computations
    # (a)
    affine1_a = x_a.dot(W1) + b1
    relu1_a = np.maximum(0, affine1_a)
    affine2_a = relu1.dot(W2) + b2 
    pred_a = affine2_a

    dout_a = 1 * 2*(pred_a - label_a)

    daffine1_a = dout_a.dot(W2.T)
    dW2 = relu1_a.T.dot(dout_a)
    db2 = np.sum(dout_a, axis=0)
    drelu1_a = np.where(affine1_a > 0, daffine1_a, 0)
    dW1 = x_a.T.dot(drelu1_a)
    db1 = np.sum(drelu1_a, axis=0)

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

    return pred_b

def backward():
    dout_b = 1

    # deriv w.r.t b (lower half)
    # d 1st layer
    daffine1_b = dout_b.dot(W2_prime.T)
    dW2_prime = relu1_b.T.dot(dout_b)
    db2_prime = np.sum(dout_b, axis=0)

    drelu1_b = np.where(affine1_b > 0, daffine1_b, 0)
    # d 2nd layer
    dW1_prime = x_a.T.dot(drelu1_b)
    db1_prime = np.sum(drelu1_b, axis=0)

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
    ddout_a = relu1_a.dot(ddW2.T)
    ddout_a += ddb2

    drelu1_a = ddW2.dot(dout_a.T) 

    #ddafine1_a = x_a.dot(ddW1.T)
    #ddafine1_a += ddb1
    # this gradient actually gets killed because the 2nd deriv of relu is 0 everywhere

    dpred_a = ddout_a * 2*(pred_a-label_a) # = dout_a
    dW2 += relu1_a.T.dot(dpred_a)
    db2 += np.sum(dpred_a, axis=0)

    drelu1_a += 

