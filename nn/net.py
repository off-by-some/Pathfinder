from dqn import DQN
import numpy as np
from collections import deque
import subprocess as sp

class AgentController:

  def __init__(self):
    self.successes = 0
    self.fails = 0
    self.possible_action_count = 4
    self.steps = 200
    self.buf = deque(maxlen=100)
    self.reward_sum = 0
    self.distance = 200
    # initialize dqn learning
    self.dqn = DQN(self.possible_action_count, (200*2,1))

  def code_from_int(self, intt):
      if intt == 0: return "up"
      elif intt == 1: return "right"
      elif intt == 2: return "down"
      elif intt == 3: return "left"
      elif intt == 4: return "jump"
      else: return "attack"


  def calculate_action(self, o, a):
    too_damn_high = self.distance >= 125
    done = too_damn_high or self.distance <= 0

    reward = 0
    # Reward for going up
    if a == 0:
      reward = reward + 1
      self.distance = self.distance - 1
    else:
      reward = reward - 1
      self.distance = self.distance - 1

    # Punish/reward for meeting goal/failing
    if self.distance <= 0:
      reward = reward + 20
      self.distance = 100
      done = True
      self.successes = self.successes + 1

    elif too_damn_high:
      reward = reward - 20
      self.distance = 100
      self.fails = self.fails + 1
      done = True

    return o, reward, done, 1

  # Get an action for an observation
  def get_action(self, o):
    # select action based on the model
    return self.dqn.select_action(o)

  def reward_action(self, o, n_o, action, reward, done=False, train=True):
    # update the state
    self.dqn.update_state(action, o, n_o, reward, done)
    self.reward_sum += reward

    if train:
      # train the model
      self.dqn.train_step()


    # if done:
    #     self.buf.append(self.reward_sum)

  def run_observation(self, o):
    # observation = np.zeros((720, ))
    for t in range(self.steps):

      # Have we reached our target?
      if np.mean(self.buf) > 102:
        break

      # select action based on the model
      action = self.dqn.select_action(o)

      # execute actin in emulator
      new_observation, reward, done, _ = self.calculate_action(o, action)

      # update the state
      self.dqn.update_state(action, o, new_observation, reward, done)

      o = new_observation

      # train the model
      self.dqn.train_step()

      self.reward_sum += reward
      sp.call('clear',shell=True)
      print "##" * 30
      print "# Individual Runs #"
      print "##" * 30
      print
      print "Passing sets: " + (self.successes * ".")
      print "Failing sets: " + (self.fails * "x")
      print
      print "##" * 30
      print "# Average Reward per set: " + str(np.mean(self.buf)) + " #"
      print "##" * 30
      print
      print "Current Set's distance to goal: " + str(self.distance if not done else 0)
      print "Current key pressed:", self.code_from_int(action)
      if done:
          print "Last action taken:" + str(action)
          print "Step ", t
          print "Finished after {} timesteps".format(t+1)
          print "Reward for this episode: ", self.reward_sum
          self.buf.append(self.reward_sum)
          print "Average reward for last 100 episodes: ", np.mean(self.buf)
          break

    self.reward_sum = 0
