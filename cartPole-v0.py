import numpy as np
import tensorflow as tf
import gym
import random
import matplotlib
matplotlib.use('PDF') # save as pdf
import matplotlib.pyplot as plt
from collections import deque

# env = gym.make('CartPole-v0')
# env = gym.make('CartPole-v1')
env = gym.make('Acrobot-v1')
# env = gym.make('MountainCar-v0')

### CONFIGURATION ###
RENDER = False

TIMESTEPS = 10000
EPISODES = 1000

ACTIONS = env.action_space.n
INPUT_DIMENSION = env.observation_space.shape[0]
OUTPUT_DIMENSION = ACTIONS
HIDDEN_NEURONS1 = 16
HIDDEN_NEURONS2 = 16
LEARNING_RATE = 0.01
DECAY_RATE = 0.01
BATCH = 64

GAMMA = 0.99
OBSERVE = 1000
EXPLORE = 100.
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.1
REPLAY_MEMORY = 100000

### MODEL ###
x = tf.placeholder("float", [None, INPUT_DIMENSION], "input")
fc1 = tf.contrib.layers.fully_connected(x, num_outputs=HIDDEN_NEURONS1, activation_fn=tf.nn.relu,
                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                        biases_initializer=tf.constant_initializer(0.1))
# fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=HIDDEN_NEURONS2, activation_fn=tf.nn.relu,
#                                         weights_initializer=tf.contrib.layers.xavier_initializer(),
#                                         biases_initializer=tf.constant_initializer(0.1))
out_layer1 = tf.contrib.layers.fully_connected(fc1, num_outputs=OUTPUT_DIMENSION, activation_fn=None,
                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                        biases_initializer=tf.constant_initializer(0.1))

a = tf.placeholder("float", [None, ACTIONS], "actions")
y_ = tf.placeholder("float", [None], "output")

pred = out_layer1

readout_action = tf.reduce_sum(tf.mul(pred, a), reduction_indices = 1)
cost = tf.contrib.losses.mean_squared_error(y_, readout_action)

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)


### Preparation for TRAINING #####

# Store in replay memory D
D = deque()

rewardPerEpisode = [[],[]] # used for plotting later
lossPerTimeframe = [[],[]] # used for plotting later

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    epsilon = INITIAL_EPSILON
    t_total = 0
    for i_episode in range(EPISODES):
        observation = env.reset()
        s_t = observation # Store first state in memory D
        sum_reward = 0
        for t in range(TIMESTEPS):
            if RENDER:
                env.render()
            loss = 0

            action = np.zeros([ACTIONS])
            if random.random() <= epsilon or t_total<=OBSERVE:
                action_index = random.randrange(ACTIONS)
            else:
                readout_t = pred.eval(feed_dict = {x : [s_t]})[0]
                action_index = np.argmax(readout_t)
            action[action_index] = 1

            # Get new state and append to buffer
            observation, reward, done, info = env.step(action_index)
            D.append((s_t, action, reward, observation, done))
            sum_reward += reward

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
                # print("pred:           ", pred.eval(feed_dict={x: s_j_batch}))
                # print("a :             ", a.eval(feed_dict={a: a_batch}))
                # print("readout_action: ", readout_action.eval(feed_dict = {x: s_j_batch, a: a_batch}))
                # perform gradient step
                _c, loss = sess.run([optimizer, cost], feed_dict = {
                    y_ : y_batch,
                    a  : a_batch,
                    x  : s_j_batch})
                # print("Cost = ", c)
                lossPerTimeframe[0].append(t_total)
                lossPerTimeframe[1].append(loss)

            s_t = observation

            t_total += 1

            if done or t == TIMESTEPS-1:
                print("Episode {} finished after {} timesteps".format(i_episode, t+1))
                break
            ## END OF FRAME
        #Adjust epsilon for next batch
        if epsilon > FINAL_EPSILON and t_total > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        rewardPerEpisode[0].append(i_episode)
        rewardPerEpisode[1].append(sum_reward)
        # print("eps", epsilon, "reward", sum_reward)
        # END OF EPISODE

## Plotting stuff
plt.clf()
fig = plt.figure(0)
ax1 = fig.add_subplot(111)
ax1.plot(rewardPerEpisode[0], rewardPerEpisode[1], color='blue')
plt.axis([0., EPISODES, 0., 500.])
plt.title('cartPole-v0')
plt.xlabel('episode')
plt.ylabel('reward')
plt.savefig('reward')

plt.clf()
fig = plt.figure(1)
ax1 = fig.add_subplot(111)
ax1.semilogy(lossPerTimeframe[0], lossPerTimeframe[1], color='red')
# ax1.loglog(lossPerTimeframe[0], lossPerTimeframe[1], color='red')
plt.xlim(lossPerTimeframe[0][0], TIMESTEPS)
plt.title('cartPole-v0')
plt.xlabel('timeframe')
plt.ylabel('loss')
plt.savefig('loss')
# env.monitor.close()
