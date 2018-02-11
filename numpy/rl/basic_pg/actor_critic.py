import argparse
import gym
import numpy as np
from itertools import count
from cs231n.layers import affine_forward, affine_backward, softmax_forward, softmax_backward
from cs231n.layer_utils import affine_relu_forward, affine_relu_backward 
from cs231n.optim import adam, sgd
from cs231n.gradient_check import eval_numerical_gradient_array, rel_error

parser = argparse.ArgumentParser(description='Numpy REINFORCE')
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
env.seed(args.seed)
np.random.seed(args.seed)

# TODO: add support for continuous actions


class Policy(object):
    """
    Neural network policy  

    ARCHITECTURE:
    {affine - relu } x (L - 1) - affine - softmax  
    """
    def __init__(self, ob_n, ac_n, hidden_dims=[500], dtype=np.float32):
        """
        Initialize a neural network to choose actions

        Inputs:
        - ob_n: Length of observation vector
        - ac_n: Number of possible actions
        - hidden_dims: List of size of hidden layer sizes
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        """
        self.ob_n = ob_n
        self.ac_n = ac_n
        self.hidden_dims = hidden_dims 
        self.dtype = dtype

        self.num_layers = len(self.hidden_dims)  

        # Initialize all weights (model params) with "Javier Initialization" 
        # weight matrix init = uniform(-1, 1) / sqrt(layer_input)
        # bias init = zeros()
        self.params = {}
        layer_input_dim = ob_n
        for i, H in enumerate(self.hidden_dims):
            self.params['W{}'.format(i)] = (-1 + 2*np.random.rand(layer_input_dim, H)) / np.sqrt(layer_input_dim)
            layer_input_dim = H
            self.params['b{}'.format(i)] = np.zeros(layer_input_dim)
        self.params['W{}'.format(i+1)] = (-1 + 2*np.random.rand(layer_input_dim, ac_n)) / np.sqrt(hidden_dims[-1])
        self.params['b{}'.format(i+1)] = np.zeros(ac_n)

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)

        # Neural net bookkeeping 
        self.cache = {}
        self.grads = {}
        # Configuration for Adam optimization
        self.optimization_config = {'learning_rate': 1e-3}
        self.adam_configs = {}
        for p in self.params:
            d = {k: v for k, v in self.optimization_config.items()}
            self.adam_configs[p] = d

        # RL specific bookkeeping
        self.saved_neg_log_probs = []
        self.rewards = []

    def zero_grads(self):
        """Reset gradients to 0. This should be called during optimization steps"""
        for g in self.grads:
            self.grads[g] = np.zeros_like(self.grads[g])

    def _add_to_cache(self, name, val):
        """Helper function to add a parameter to the cache without having to do checks"""
        if name in self.cache:
            self.cache[name].append(val)
        else:
            self.cache[name] = [val]

    def _set_grad(self, name, val):
        """Helper fucntion to set gradient without having to do checks"""
        if name in self.cache:
            self.grads[name] += val
        else:
            self.grads[name] = val

    def _cache_to_list(self, cache):
        """Helper function to convert cache to list"""
        cx, cw, cb = zip(*cache)
        cache = (np.concatenate(cx), cw[0], cb[0])
        return cache

    def forward(self, x):
        """
        Forward pass observations (x) through network to get probabilities (scores) 
        of taking each action 
        """
        layer_input = x
        # run input through all hidden layers
        for layer in range(self.num_layers):
            layer_input, layer_cache = affine_relu_forward(layer_input, self.params['W%d'%layer], self.params['b%d'%layer])
            self._add_to_cache(layer, layer_cache) # todo: don't save w and b because they are constant
        # run it through last layer to get activations
        logits, last_cache = affine_forward(layer_input, self.params['W%d'%self.num_layers], self.params['b%d'%self.num_layers])
        self._add_to_cache(self.num_layers, last_cache)

        # pass through a softmax to get probabilities 
        scores = softmax_forward(logits)
        self._add_to_cache('soft', scores)
        return scores 


    def backward(self, dout):
        """
        Chain rule the derivatives backward through all network computations, 
        and set self.grads for each of the weights (to be used in stochastic gradient
        descent optimization (adam))
        """
        dout = dout * np.ones([1,self.ac_n])
        dout, dw, db = affine_backward(dout, self._cache_to_list(self.cache[self.num_layers]))
        self._set_grad('W%d'%self.num_layers, dw)
        self._set_grad('b%d'%self.num_layers, db)

        for i in reversed(range(self.num_layers)):
            fc_cache, relu_cache = zip(*self.cache[i])
            fc_cache = self._cache_to_list(fc_cache)
            relu_cache = np.concatenate(relu_cache)
            dout, dw, db = affine_relu_backward(dout, (fc_cache, relu_cache))
            self._set_grad('W%d'%i, dw)
            self._set_grad('b%d'%i, db)

        # reset cache for next backward pass
        self.cache = {}

ob_n = env.observation_space.shape[0]
ac_n = env.action_space.n
policy = Policy(ob_n, ac_n)

def select_action(obs):
    """
    Pass observations through network and sample an action to take. Keep track
    of dsoftmax to use to update weights
    """
    obs = np.reshape(obs, [1, -1])
    probs = policy.forward(obs)[0]
    action = np.random.choice(policy.ac_n, p=probs)
    # I am not really sure if this signal is standard or if the math checks out,
    # but it works and it makes sense
    # 1. for actions not taken, decrease weights proportional to their probabilties
    # (if the reward is positive, this will make these less probable after updating)
    dsoftmax = -probs 
    # 2. for the action that was chose, if the probability was loss, this will be higher
    # (if the reward is positive, this will make these more probable after updating)
    dsoftmax[action] += 1

    policy.saved_neg_log_probs.append(dsoftmax)

    # this is what is used in other implementations that I have seen, but I couldn't
    # figure out how to make it work
    #neg_log = -np.log(probs[action])
    #policy.saved_neg_log_probs.append(neg_log)
    return action

def finish_episode():
    """
    At the end of the episode, calculate the discounted return for each time step
    """
    next_return = 0
    policy_loss = []
    returns = []

    # Calculate discounted reward and normalize it
    for R in policy.rewards[::-1]:
        next_return = R + args.gamma * next_return
        returns.insert(0, next_return)
    returns = np.array(returns)
    returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float32).eps)

    # Multiply the signal that makes actions taken more probable by the discounted
    # return of that action.  This will pull the weights in the direction that
    # makes *better* actions more probable.
    for neg_log_prob, reward in zip(policy.saved_neg_log_probs, returns):
        policy_loss.append(neg_log_prob * reward)

    policy.backward(-np.stack(policy_loss))

    # run an optimization step on all of the model parameters
    for p in policy.params:
        next_w, policy.adam_configs[p] = adam(policy.params[p], policy.grads[p], config=policy.adam_configs[p])
        policy.params[p] = next_w
    policy.zero_grads() # required every call to adam

    del policy.rewards[:]
    del policy.saved_neg_log_probs[:]

def main():
    """Run REINFORCE algorithm to train on the environment"""
    avg_reward = []
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

        else:
            avg_reward.append(ep_reward)

if __name__ == '__main__':
    main()
