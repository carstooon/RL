import numpy as np
import tensorflow as tf
import gym
import random
import matplotlib
matplotlib.use('PDF') # save as pdf
import matplotlib.pyplot as plt
from collections import deque

env = gym.make('CartPole-v0')

### CONFIGURATION ###
TIMESTEPS = 10000
EPISODES = 1000

ACTIONS = env.action_space.n
INPUT_DIMENSION = env.observation_space.shape[0]
OUTPUT_DIMENSION = ACTIONS
HIDDEN_NEURONS1 = 20
HIDDEN_NEURONS2 = 30
LEARNING_RATE = 0.01
DECAY_RATE = 0.1
BATCH = 32

GAMMA = 0.99
OBSERVE = 500
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.1
EXPLORE = 50.
REPLAY_MEMORY = 10000
K = 1 # only select an action every K frames


### MODEL ###
with tf.name_scope('Model'):
    with tf.name_scope('Model_Parameter'):
        x = tf.placeholder("float", [None, INPUT_DIMENSION], "input")

    # # Construct model
    # with tf.name_scope('NeuralNetwork'):
    #     with tf.name_scope('Weights'):
    #         weights = {
    #         'h1': tf.get_variable(name="Hidden_Layer1", shape=[INPUT_DIMENSION, HIDDEN_NEURONS1], initializer=tf.contrib.layers.xavier_initializer()),
    #         'h2': tf.get_variable(name="Hidden_Layer2", shape=[HIDDEN_NEURONS1, HIDDEN_NEURONS2], initializer=tf.contrib.layers.xavier_initializer()),
    #         'out': tf.get_variable(name="Weight_out", shape=[HIDDEN_NEURONS2, OUTPUT_DIMENSION], initializer=tf.contrib.layers.xavier_initializer())
    #         }
    #     with tf.name_scope('Biases'):
    #         biases = {
    #         'b1':  tf.get_variable(name="bias_1", shape=[HIDDEN_NEURONS1], initializer=tf.random_normal_initializer()),
    #         'b2':  tf.get_variable(name="bias_2", shape=[HIDDEN_NEURONS2], initializer=tf.random_normal_initializer()),
    #         'out': tf.get_variable(name="bias_out", shape=[OUTPUT_DIMENSION], initializer=tf.random_normal_initializer())
    #         }
    #     # Hidden layer with RELU activation
    #     with tf.name_scope('Layers'):
    #         layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    #         layer_1 = tf.nn.relu(layer_1)
    #         layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    #         layer_2 = tf.nn.relu(layer_2)
    #         # Output layer with linear activation
    #         out_layer2 = tf.matmul(layer_2, weights['out']) + biases['out']
    # Construct model
    with tf.name_scope('NeuralNetwork'):
        with tf.name_scope('Weights'):
            weights = {
            'h1': tf.get_variable(name="Hidden_Layer1", shape=[INPUT_DIMENSION, HIDDEN_NEURONS1], initializer=tf.contrib.layers.xavier_initializer()),
            'out': tf.get_variable(name="Weight_out", shape=[HIDDEN_NEURONS1, OUTPUT_DIMENSION], initializer=tf.contrib.layers.xavier_initializer())
            }
        with tf.name_scope('Biases'):
            biases = {
            'b1':  tf.get_variable(name="bias_1", shape=[HIDDEN_NEURONS1], initializer=tf.random_normal_initializer()),
            'out': tf.get_variable(name="bias_out", shape=[OUTPUT_DIMENSION], initializer=tf.random_normal_initializer())
            }
        # Hidden layer with RELU activation
        with tf.name_scope('Layers'):
            layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
            layer_1 = tf.nn.relu(layer_1)
            # Output layer with linear activation
            out_layer1 = tf.matmul(layer_1, weights['out']) + biases['out']

    a = tf.placeholder("float", [None, ACTIONS], "actions")
    y_ = tf.placeholder("float", [None], "output")

    # pred = out_layer2
    pred = out_layer1

    readout_action = tf.reduce_sum(tf.mul(pred, a), reduction_indices = 1)
    cost = tf.contrib.losses.mean_squared_error(y_, readout_action)
    tf.scalar_summary('loss', cost)

    with tf.name_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)


### Preparation for TRAINING #####

# Store in replay memory D
D = deque()

timestepsPerEpisode = [[],[]] # used for plotting later

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    epsilon = INITIAL_EPSILON
    t_total = 0
    for i_episode in range(EPISODES):
        observation = env.reset()
        s_t = observation # Store first state in memory D

        for t in range(TIMESTEPS):
            # env.render()

            action = np.zeros([ACTIONS])
            if random.random() <= epsilon or t_total<=OBSERVE:
                action_index = random.randrange(ACTIONS)
                action[action_index] = 1
            else:
                readout_t = pred.eval(feed_dict = {x : [s_t]})[0]
                action_index = np.argmax(readout_t)
                action[action_index] = 1

            # Get new state and append to buffer
            observation, reward, done, info = env.step(action_index)
            D.append((s_t, action, reward, observation, done))
            if len(D) > REPLAY_MEMORY:
                D.popleft()

            if t_total > OBSERVE:
                minibatch = random.sample(D, BATCH)

                s_j_batch  = [d[0] for d in minibatch]
                a_batch    = [d[1] for d in minibatch]
                r_batch    = [d[2] for d in minibatch]
                s_j1_batch = [d[3] for d in minibatch]

                y_batch = []
                # get prediction for new time frame
                readout_j1_batch = pred.eval(feed_dict = {x : s_j1_batch})
                for i in range(BATCH):
                    # if terminal only equals reward
                    if minibatch[i][4]:
                        y_batch.append(r_batch[i])
                    else:
                        y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))
                # print(y_batch)
                # perform gradient step
                _c, c = sess.run([optimizer, cost], feed_dict = {
                    y_ : y_batch,
                    a  : a_batch,
                    x  : s_j_batch})
                # print("Cost = ", c)

            s_t = observation
            t_total += 1

            if done or t == TIMESTEPS-1:
                print("Episode {} finished after {} timesteps".format(i_episode, t+1))
                break
            ## END OF FRAME
        #Adjust epsilon for next batch
        if epsilon > FINAL_EPSILON and t_total > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        timestepsPerEpisode[0].append(i_episode)
        timestepsPerEpisode[1].append(t+1)
        print("eps", epsilon, "")
        # END OF EPISODE

## Plotting stuff
plt.clf()
fig = plt.figure(0)
ax1 = fig.add_subplot(111)
ax1.plot(timestepsPerEpisode[0], timestepsPerEpisode[1], color='blue')
# plt.axis([-10., 10., -10., 10.])
plt.title('cartPole-v0')
plt.xlabel('episode')
plt.ylabel('steps')
plt.savefig('steps')
# env.monitor.close()
