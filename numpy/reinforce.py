import argparse
import gym
import numpy as np
from itertools import count
from cs231n.layers import affine_forward, affine_backward
from cs231n.layer_utils import affine_relu_forward, affine_relu_backward, 
from cs231n.optim import adam

parser = argparse.ArgumentParser(description='Numpy REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='interval between training status logs (default: 100)')
parser.add_argument('--render-interval', type=int, default=100, metavar='N',
                    help='interval between rendering (default: 100)')
args = parser.parse_args()

env = gym.make('LunarLander-v2')
env.seed(args.seed)
np.random.seed(args.seed)


class Policy(object):
    def __init__(self, ob_n, ac_n, hidden_dims=2, dtype=np.float32):
        self.params = {}
        self.hidden_dims = hidden_dims
        self.num_layers = self.hidden_dims

        layer_input_dim = ob_n
        for i, H in self.hidden_dims:
            self.params['W{}'.format(i)] = np.random.randn(H, layer_input_dim) / np.sqrt(layer_input_dim)
            self.params['b{}'.format(i)] = np.zeros(layer_input_dim)
            layer_input_dim = H

        self.params['W{}'.format(i)] = weight_scale * np.random.randn(layer_input_dim, ac_n)
        self.params['b{}'.format(i)] = np.zeros(ac_n)

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
          self.params[k] = v.astype(dtype)

        # RL specifics
        self.saved_log_probs = []
        self.rewards = []
        self.cache = None
        self.grads = None

    def forward(self, x):
        layer_input = x
        self.cache = {}
        
        for layer in range(self.num_layers):
            layer_input, layer_cache = affine_relu_forward(layer_input, self.params['W%d'%layer], self.params['b%d'%layer])
            self.cache[layer] = layer_cache 
        scores, self.cache[self.num_layers] = affine_forward(layer_input, self.params['W%d'%self.num_layers], self.params['b%d'%self.num_layers])
        self.cache['soft'] = softmax_forward(scores)
    
    def backward(self, dout):
        self.grads = {}

        dout = softmax_backward(dout, self.cache['soft'])
        dout, dw, db = affine_backward(dout, self.cache[self.num_layers])

        self.grads['W%d'%self.num_layers] = dw 
        self.grads['b%d'%self.num_layers] = db
        for i in reversed(range(self.num_layers)):
            dout, dw, db = affine_relu_backward(hout, ar_cache[i])
            self.grads['W%d'%i] = dw
            self.grads['b%d'%i] = db

policy = Policy()

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(Variable(state))
    m = Multinomial(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.data[0]

def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)

    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]

def main():
    avg_reward = []
    oneh_img_list = []
    onet_img_list = []
    tent_img_list = []
    img_list = []
    for i_episode in count(1):
        ep_reward = 0
        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            if args.obstrial != -1:
                # block out a state
                state[args.obstrial] = 0

            action = select_action(state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward

            if args.timetrial:
                if i_episode == 100:
                    oneh_img_list.append(env.render(mode='rgb_array'))
                elif i_episode == 1000:
                    onet_img_list.append(env.render(mode='rgb_array'))
                elif i_episode == 10000:
                    tent_img_list.append(env.render(mode='rgb_array'))

            elif args.obstrial != -1:
                if i_episode == 1000:
                    img_list.append(env.render(mode='rgb_array'))

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
