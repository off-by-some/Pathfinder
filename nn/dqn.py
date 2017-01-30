import numpy as np
import random as random
from collections import deque

from cnn import CNN

# See https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf for model description

class DQN:
  def __init__(self, num_actions, observation_shape):
    self.num_actions = num_actions
    self.epsilon = 0.1
    self.gamma = 0.9
    self.mini_batch_size = 10

    # memory
    self.memory = deque(maxlen=1000)

    # initialize network
    self.model = CNN(num_actions, observation_shape)
    print "model initialized"

  def select_action(self, observation, state_in):
    """
    Selects the next action to take based on the current state and learned Q.
    Args:
      observation: the current state
    """
    state_out = state_in

    if random.random() < self.epsilon:
      # with epsilon probability select a random action
      action = np.random.randint(0, self.num_actions)
      self.epsilon *= 0.99
    else:
      # select the action a which maximizes the Q value
      obs = np.array([observation])
      q_values, state_out = self.model.predict(obs, state_in)
      action = np.argmax(q_values)

    return action, state_out

  def update_state(self, action, observation, new_observation, state_in, state_out, reward, done):
    """
    Stores the most recent action in the replay memory.
    Args:
      action: the action taken
      observation: the state before the action was taken
      new_observation: the state after the action is taken
      state_out: the state of the lstm after the network was run
      reward: the reward from the action
      done: a boolean for when the episode has terminated
    """
    transition = {'action': action,
                  'observation': observation,
                  'new_observation': new_observation,
                  'state_in': state_in,
                  'state_out': state_out,
                  'reward': reward,
                  'is_done': done}
    self.memory.append(transition)

  def get_random_mini_batch(self):
    """
    Gets a random sample of transitions from the replay memory.
    """
    rand_idxs = random.sample(xrange(len(self.memory)), self.mini_batch_size)
    mini_batch = []
    for idx in rand_idxs:
      mini_batch.append(self.memory[idx])

    return mini_batch

  def train_step(self):
    """
    Updates the model based on the mini batch
    """
    if len(self.memory) > self.mini_batch_size:
      mini_batch = self.get_random_mini_batch()

      Xs = []
      ys = []
      cs = []
      hs = []
      actions = []

      for sample in mini_batch:
        y_j = sample['reward']
        c, h = sample['state_in']

        # for nonterminals, add gamma*max_a(Q(phi_{j+1})) term to y_j
        if not sample['is_done']:
          new_observation = sample['new_observation']
          new_obs = np.array([new_observation])
          q_new_values, _ = self.model.predict(new_obs, sample['state_in'])
          action = np.max(q_new_values)
          y_j += self.gamma*action

        action = np.zeros(self.num_actions)
        action[sample['action']] = 1

        observation = sample['observation']

        Xs.append(observation.copy())
        ys.append(y_j)
        cs.append(c[0])
        hs.append(h[0])
        actions.append(action.copy())

      Xs = np.array(Xs)
      ys = np.array(ys)
      cs = np.array(cs)
      hs = np.array(hs)
      actions = np.array(actions)

      self.model.train_step(Xs, ys, (cs, hs), actions)
