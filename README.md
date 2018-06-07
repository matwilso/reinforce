# Reinforcement Learning (RL) Implementations

These are generally not best practices or very efficient. They are not for 
practical use, but more geared towards understanding the methods.

All of these that are listed here do actually work and you can try running them and they 
will train in a few minutes

## TensorFlow

### Basic RL algorithms
- [REINFORCE](/tensorflow/reinforce.py)
	- Discrete actions, tested on OpenAI gym CartPole, LunarLander

## numpy

### Basic RL algorithms
- [REINFORCE](/numpy/rl/reinforce.py/)
	- Discrete actions, tested on OpenAI gym CartPole, LunarLander
- [Actor-Critic (kind of)](/numpy/rl/actor_critic.py/)
	- Weird batched version. still a WIP



### TODO:
- create more official website documentation w/ Github Pages, or just have nice readmes
- create some weight visualizer to somehow display the learned weights and how 
different inputs cause trained neurons to fire (Tensorflow Beholder might be great for this)
