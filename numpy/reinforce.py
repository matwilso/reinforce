import argparse
import gym
import numpy as np
from itertools import count
from cs231n.layers import affine_forward, affine_backward, softmax_forward, softmax_backward
from cs231n.layer_utils import affine_relu_forward, affine_relu_backward 
from cs231n.optim import adam, sgd
from cs231n.gradient_check import eval_numerical_gradient_array, rel_error

parser = argparse.ArgumentParser(description='Numpy REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=42, metavar='N',
                    help='random seed (default: 42)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='interval between training status logs (default: 100)')
parser.add_argument('--render_interval', type=int, default=100, metavar='N',
                    help='interval between rendering (default: 100)')
args = parser.parse_args()

env = gym.make('LunarLander-v2')
#env = gym.make('CartPole-v0')
#env.seed(args.seed)
#np.random.seed(args.seed)

# TODO: move the numerical gradient check to a test suite 

class Policy(object):
    """
    Neural network policy  
    """

    def __init__(self, ob_n, ac_n, hidden_dims=[400], dtype=np.float32):
    #def __init__(self, ob_n, ac_n, hidden_dims=[2, 2], dtype=np.float32):
        self.params = {} # dict to hold weight matrices
        self.hidden_dims = hidden_dims # list of sizes of hidden dims
        self.num_layers = len(self.hidden_dims) 
        self.ob_n = ob_n
        self.ac_n = ac_n

        # initialize all weight matrices
        layer_input_dim = ob_n
        for i, H in enumerate(self.hidden_dims):
            self.params['W{}'.format(i)] = np.random.randn(layer_input_dim, H) / np.sqrt(layer_input_dim)
            layer_input_dim = H
            self.params['b{}'.format(i)] = np.zeros(layer_input_dim)
        self.params['W{}'.format(i+1)] = np.random.randn(layer_input_dim, ac_n) / np.sqrt(ac_n)
        self.params['b{}'.format(i+1)] = np.zeros(ac_n)
        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
          self.params[k] = v.astype(dtype)

        # RL specifics
        self.saved_observations = []
        self.saved_actions = []
        self.saved_logits = []
        self.saved_neg_log_probs = []
        self.rewards = []
        self.cache = {}
        self.grads = {}

        self.config = {'learning_rate': 1e-3}
        self.configs = {}
        for p in self.params:
            d = {k: v for k, v in self.config.items()}
            self.configs[p] = d


    def __call__(self, x):
        return self.forward(x)

    def add_to_cache(self, name, val):
        """dict try add"""
        if name in self.cache:
            self.cache[name].append(val)
        else:
            self.cache[name] = [val]

    def update_grad(self, name, val):
        if name in self.cache:
            self.grads[name] += val
        else:
            self.grads[name] = val

    def ctol(self, cache):
        cx, cw, cb = zip(*cache)
        cache = (np.concatenate(cx), cw[0], cb[0])
        return cache

    def forward(self, x):
        layer_input = x
        #print("INPUT ", x)
        
        # run input through all hidden layers
        for layer in range(self.num_layers):
            layer_input, layer_cache = affine_relu_forward(layer_input, self.params['W%d'%layer], self.params['b%d'%layer])
            #print("LAYER{} ".format(layer), layer_input)
            self.add_to_cache(layer, layer_cache) # todo: don't save w and b because they are constant

        # run it through last layer to get activations
        logits, last_cache = affine_forward(layer_input, self.params['W%d'%self.num_layers], self.params['b%d'%self.num_layers])
        #print("LOGITS ", logits)
        policy.saved_logits.append(logits)
        self.add_to_cache(self.num_layers, last_cache)

        # pass through a softmax to get probabilities 
        scores = softmax_forward(logits)
        self.add_to_cache('soft', scores)
        return scores

    def zero_grads(self):
        for g in self.grads:
            self.grads[g] = np.zeros_like(self.grads[g])

    def backward(self, dout, saved_actions):

        logits = np.concatenate(policy.saved_logits) 
        dout_num = eval_numerical_gradient_array(lambda x: softmax_forward(x), logits, dout)
        #print(dout)
        dout = softmax_backward(dout, np.concatenate(self.cache['soft']), saved_actions)
        #import ipdb; ipdb.set_trace()
        #dout = dout * np.ones([1,self.ac_n])
        #print(dout)
        dout, dw, db = affine_backward(dout, self.ctol(self.cache[self.num_layers]))
        #print(dw)

        self.update_grad('W%d'%self.num_layers, dw)
        self.update_grad('b%d'%self.num_layers, db)

        for i in reversed(range(self.num_layers)):
            fc_cache, relu_cache = zip(*self.cache[i])
            fc_cache = self.ctol(fc_cache)
            relu_cache = np.concatenate(relu_cache)
            dout, dw, db = affine_relu_backward(dout, (fc_cache, relu_cache))
            #print(i, dw)
            self.update_grad('W%d'%i, dw)
            self.update_grad('b%d'%i, db)

        self.cache = {}


policy = Policy(8, 4)

def select_action(obs):
    obs = np.reshape(obs, [1, -1])
    probs = policy.forward(obs)[0]

    action = np.random.choice(policy.ac_n, p=probs)
    neg_log = -np.log(probs[action])
    policy.saved_neg_log_probs.append(neg_log)
    policy.saved_observations.append(obs)
    policy.saved_actions.append(action)

    return action

ti = 0
def finish_episode():
    global ti
    next_return = 0
    policy_loss = []
    returns = []

    for R in policy.rewards[::-1]:
        next_return = R + args.gamma * next_return
        returns.insert(0, next_return)

    returns = np.array(returns)
    returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float32).eps)

    for neg_log_prob, reward in zip(policy.saved_neg_log_probs, returns):
        policy_loss.append(neg_log_prob * reward)

    observations = np.stack(policy.saved_observations)
    actions = np.stack(policy.saved_actions)
    policy.backward(np.stack(policy_loss).reshape(-1, 1), actions)

    ti += 1

    if ti % 1 == 0:
        for p in policy.params:
            next_w, policy.configs[p] = adam(policy.params[p], policy.grads[p], config=policy.configs[p])
            policy.params[p] = next_w
            #print(p, next_w)
        policy.zero_grads()

    del policy.rewards[:]
    del policy.saved_neg_log_probs[:]
    del policy.saved_observations[:]
    del policy.saved_actions[:]
    del policy.saved_logits[:]


import matplotlib.pyplot as plt
def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()

def main():
    avg_reward = []
    oneh_img_list = []
    onet_img_list = []
    tent_img_list = []
    img_list = []
    for i_episode in count(1):
        ep_reward = 0
        obs = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            action = select_action(obs)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward

            policy.rewards.append(reward)

            if done:
                break

        finish_episode()


        if i_episode % args.log_interval == 0:
            print("Ave reward: {}".format(sum(avg_reward)/len(avg_reward)))
            avg_reward = []

            #hinton(policy.params['W0'])
            #plt.show()
        else:
            avg_reward.append(ep_reward)


if __name__ == '__main__':
    main()
