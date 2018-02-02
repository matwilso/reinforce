from cs231n.gradient_check import *
from cs231n.layers import softmax_forward, softmax_backward


np.random.seed(43)

x = np.random.randn(1,4)
dout = np.ones(4)
#dout = np.array([[1.1, 2.23, -2.5, -0.7]])
#dout = np.array([[1.1, 2.23, -2.5, -0.7]])
x_num = eval_numerical_gradient_array(lambda x: softmax_forward(x), x, dout)

scores = softmax_forward(x)
x_real = softmax_backward(dout, scores, 0)

print(x_num)
print(x_real)
print(rel_error(x_num, x_real))
