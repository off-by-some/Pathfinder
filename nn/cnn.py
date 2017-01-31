import tensorflow as tf
import numpy as np
import logging

class CNN:
  """
  Convolutional Neural Network model.
  """

  def __init__(self, num_actions, observation_shape, verbose=False):
    """
    Initialize the CNN model with a set of parameters.
    Args:
      params: a dictionary containing values of the models' parameters.
    """
    # 2n + 1
    self.verbose = verbose
    self.num_actions = num_actions

    # observation shape will be a tuple
    self.observation_shape = observation_shape[0]

    self.lr = 0.0001
    self.reg = 0.001
    self.hidden_size = (self.observation_shape * 1) + 4

    self.session = self.create_model()

  def add_placeholders(self):
    input_placeholder = tf.placeholder(tf.float32, shape=(None, self.observation_shape))
    labels_placeholder = tf.placeholder(tf.float32, shape=(None,))
    actions_placeholder = tf.placeholder(tf.float32, shape=(None, self.num_actions))
    c = tf.placeholder(tf.float32, shape=(None, self.hidden_size))
    h = tf.placeholder(tf.float32, shape=(None, self.hidden_size))
    state_placeholder = (c, h)

    return input_placeholder, labels_placeholder, actions_placeholder, state_placeholder


  def nn(self, input_obs, input_state):
    with tf.name_scope("Layer1") as scope:
      W1shape = [self.observation_shape, self.hidden_size]
      W1 = tf.get_variable("W1", shape=W1shape,)
      bshape = [1, self.hidden_size]
      b1 = tf.get_variable("b1", shape=bshape, initializer = tf.constant_initializer(0.0))

    with tf.name_scope("Layer2") as scope:
      W2shape = [self.hidden_size, self.hidden_size]
      W2 = tf.get_variable("W2", shape=W2shape,)
      bshape = [1, self.hidden_size]
      b2 = tf.get_variable("b2", shape=bshape, initializer = tf.constant_initializer(0.0))

    with tf.name_scope("Layer3") as scope:
      W3shape = [self.hidden_size, self.hidden_size]
      W3 = tf.get_variable("W3", shape=W3shape,)
      bshape = [1, self.hidden_size]
      b3 = tf.get_variable("b3", shape=bshape, initializer = tf.constant_initializer(0.0))

    with tf.name_scope("OutputLayer") as scope:
      Ushape = [self.hidden_size, self.num_actions]
      U = tf.get_variable("U", shape=Ushape)
      b3shape = [1, self.num_actions]
      b4 = tf.get_variable("b4", shape=b3shape, initializer = tf.constant_initializer(0.0))

    xW = tf.matmul(input_obs, W1)
    h = tf.nn.elu(tf.add(xW, b1))

    # xW = tf.matmul(h, W2)
    # h = tf.nn.elu(tf.add(xW, b2))

    xW = tf.matmul(h, W2)
    h, output_state = self.lstm(tf.add(xW, b2), input_state)

    xW = tf.matmul(h, W3)
    h = tf.nn.elu(tf.add(xW, b3))

    hU = tf.matmul(h, U)
    out = tf.add(hU, b4)

    reg = self.reg * (tf.reduce_sum(tf.square(W1)) + tf.reduce_sum(tf.square(W3)) + tf.reduce_sum(tf.square(U)))
    return out, reg, output_state


  def create_model(self):
    """
    The model definition.
    """

    self.lstm = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size, activation=tf.nn.softsign)
    self.input_placeholder, self.labels_placeholder, self.actions_placeholder, self.state_in = self.add_placeholders()
    outputs, reg, state = self.nn(self.input_placeholder, self.state_in)
    self.predictions = outputs
    self.state_out = state

    self.q_vals = tf.reduce_sum(tf.mul(self.predictions, self.actions_placeholder), 1)

    self.loss = tf.reduce_sum(tf.square(self.labels_placeholder - self.q_vals)) + reg

    optimizer = tf.train.AdamOptimizer(learning_rate = self.lr, use_locking=True)

    self.train_op = optimizer.minimize(self.loss)
    init = tf.initialize_all_variables()
    session = tf.Session()
    session.run(init)

    return session

  def train_step(self, Xs, ys, state_in, actions):
    """
    Updates the CNN model with a mini batch of training examples.
    """
    loss, _, prediction_probs, q_values, state_out = self.session.run(
      [self.loss, self.train_op, self.predictions, self.q_vals, self.state_out],
      feed_dict = {self.input_placeholder: Xs,
                  self.state_in: state_in,
                  self.labels_placeholder: ys,
                  self.actions_placeholder: actions
                  })

  def predict(self, observation, state_in):
    """
    Predicts the rewards for an input observation state.
    Args:
      observation: a numpy array of a single observation state
    """

    loss, prediction_probs, state_out = self.session.run(
      [self.loss, self.predictions, self.state_out],
      feed_dict = {self.input_placeholder: observation,
                  self.state_in: state_in,
                  self.labels_placeholder: np.zeros(len(observation)),
                  self.actions_placeholder: np.zeros((len(observation), self.num_actions))
                  })

    return prediction_probs, state_out
