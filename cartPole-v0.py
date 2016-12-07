import random
import argparse
from collections import deque

import nn

import numpy as np
import tensorflow as tf

import gym
import matplotlib
matplotlib.use('PDF') # save as pdf
import matplotlib.pyplot as plt
from scipy.interpolate import spline

#####################################
## Argument parser
parser = argparse.ArgumentParser(description='Reinforcement Learning')
parser.add_argument('-N', '--episodes', type=int, default=500, required=False,
                   help='Number of episodes to run training')
parser.add_argument('-R', '--render', type=int, default=1, required=False,
                   help='Show the animation in every r-th episode')
args = parser.parse_args()

print("Number of Episodes: ", args.episodes)
print("Rendering: ", args.render)

#####################################
### CONFIGURATION ###

## GYM CONFIG
RENDER = args.render
EPISODES = args.episodes
TIMESTEPS = 1000
ENVIRONMENT = 'CartPole-v0'
# ENVIRONMENT = 'CartPole-v1'
# ENVIRONMENT = 'Acrobot-v1'
# ENVIRONMENT = 'MountainCar-v0'
# ENVIRONMENT = 'LunarLander-v2'
env = gym.make(ENVIRONMENT)
ACTIONS = env.action_space.n

## Neural Network Config
INPUT_DIMENSION = env.observation_space.shape[0]
OUTPUT_DIMENSION = ACTIONS
HIDDEN_NEURONS1 = 32
HIDDEN_NEURONS2 = 256
LEARNING_RATE = 0.01
DECAY_RATE = 0.01
BATCH_SIZE = 64

## Reinforcement learning
GAMMA = 0.99 # learning discount
OBSERVE = 100 # timesteps
EXPLORE = 50. # episodes to go from INITIAL_EPSILON to FINAL_EPSILON
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.01
REPLAY_MEMORY = 100000
UPDATE_TARGET_FREQUENCY = 1000 # after how many timesteps the target network gets updated

## Plotting
SMOOTH = 2

#####################################
### MODEL
x = tf.placeholder("float", [None, INPUT_DIMENSION], "input")
a = tf.placeholder("float", [None, ACTIONS], "actions")
y_ = tf.placeholder("float", [None], "output")

# pred   = nn.TwoLayerNetwork(x, OUTPUT_DIMENSION, HIDDEN_NEURONS1, HIDDEN_NEURONS2)
# target = nn.TwoLayerNetwork(x, OUTPUT_DIMENSION, HIDDEN_NEURONS1, HIDDEN_NEURONS2)
pred   = nn.OneLayerNetwork(x, OUTPUT_DIMENSION, HIDDEN_NEURONS1)
target = nn.OneLayerNetwork(x, OUTPUT_DIMENSION, HIDDEN_NEURONS1)

readout_action = tf.reduce_sum(tf.mul(pred, a), reduction_indices = 1)
cost = tf.reduce_mean(tf.square(y_ - readout_action))
# cost = tf.contrib.losses.mean_squared_error(y_, readout_action)

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)


### Preparation for TRAINING #####

# Store in replay memory D
D = deque()

rewardPerEpisode = [[],[]] # used for plotting later
lossPerTimeframe = [[],[]] # used for plotting later

init = tf.initialize_all_variables()

print("Number of input dimensions: {}".format(INPUT_DIMENSION))
print("Number of actions: {}".format(ACTIONS))

t_total = 0
with tf.Session() as sess:
    sess.run(init)
    epsilon = INITIAL_EPSILON
    for i_episode in range(EPISODES): #
        observation = env.reset()
        s_t = observation # Store first state in memory D
        sum_reward = 0
        episode_render = False
        if i_episode % RENDER == 0:
            episode_render = True
        for t in range(TIMESTEPS):
            if episode_render:
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
            if ENVIRONMENT == 'LunarLander-v2':
                reward -= 1 ## time malus
            D.append((s_t, action, reward, observation, done))
            sum_reward += reward

            if len(D) > REPLAY_MEMORY:
                D.popleft()

            if t_total > OBSERVE:
                minibatch = random.sample(D, BATCH_SIZE)

                s_j_batch  = [d[0] for d in minibatch]
                a_batch    = [d[1] for d in minibatch]
                r_batch    = [d[2] for d in minibatch]
                s_j1_batch = [d[3] for d in minibatch]

                y_batch = []
                # get prediction for new time frame
                readout_j1_batch = pred.eval(feed_dict = {x : s_j1_batch})
                readout_j1_batch_t = target.eval(feed_dict = {x : s_j1_batch})
                for i in range(BATCH_SIZE):
                    # if terminal only equals reward
                    if minibatch[i][4]:
                        y_batch.append(r_batch[i])
                    else:
                        y_batch.append(r_batch[i] + GAMMA * readout_j1_batch_t[i][np.argmax(readout_j1_batch[i])])
                        # y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch_t[i]))
                # perform gradient step
                _c, loss = sess.run([optimizer, cost], feed_dict = {
                    y_ : y_batch,
                    a  : a_batch,
                    x  : s_j_batch})
                # print("Cost = ", c)
                lossPerTimeframe[0].append(t_total)
                lossPerTimeframe[1].append(loss)

            s_t = observation # old state is changed to new state

            t_total += 1

            # update target network
            if t_total % UPDATE_TARGET_FREQUENCY == 0:
                target = pred

            if done or t == TIMESTEPS-1:
                print("Episode {} finished after {} timesteps with reward {}".format(i_episode, t+1, sum_reward))
                break
            ## END OF TIMEFRAME
        #Adjust epsilon for next batch
        if epsilon > FINAL_EPSILON and t_total > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        rewardPerEpisode[0].append(i_episode)
        rewardPerEpisode[1].append(sum_reward)

        # print("eps", epsilon, "reward", sum_reward)
        # END OF EPISODE

        if i_episode % 100 == 0 and i_episode != 0 and len(lossPerTimeframe[0]) > 0:
            ## Plotting stuff
            plt.clf()

            xnew = np.linspace(np.array(rewardPerEpisode[0]).min(),np.array(rewardPerEpisode[0]).max(),float(EPISODES)/SMOOTH)
            power_smooth = spline(rewardPerEpisode[0],rewardPerEpisode[1],xnew)
            fig = plt.figure(0)
            ax1 = fig.add_subplot(111)
            ax1.plot(xnew, power_smooth, color='blue')
            plt.xlim(0, EPISODES)
            # plt.axis([0., EPISODES, 0., 500.])
            plt.title(ENVIRONMENT)
            plt.xlabel('episode')
            plt.ylabel('reward')
            plt.savefig(ENVIRONMENT + '_reward')

            plt.clf()
            fig = plt.figure(1)
            ax1 = fig.add_subplot(111)
            ax1.semilogy(lossPerTimeframe[0], lossPerTimeframe[1], color='red')
            # ax1.loglog(lossPerTimeframe[0], lossPerTimeframe[1], color='red')
            # print(lossPerTimeframe[0])
            plt.xlim(lossPerTimeframe[0][0], lossPerTimeframe[0][-1])
            plt.title(ENVIRONMENT)
            plt.xlabel('timeframe')
            plt.ylabel('loss')
            plt.savefig(ENVIRONMENT + '_loss')
            # env.monitor.close()

## Plotting stuff
plt.clf()

xnew = np.linspace(np.array(rewardPerEpisode[0]).min(),np.array(rewardPerEpisode[0]).max(),float(EPISODES)/SMOOTH)
power_smooth = spline(rewardPerEpisode[0],rewardPerEpisode[1],xnew)
fig = plt.figure(0)
ax1 = fig.add_subplot(111)
ax1.plot(xnew, power_smooth, color='blue')
plt.xlim(0, EPISODES)
# plt.axis([0., EPISODES, 0., 500.])
plt.title(ENVIRONMENT)
plt.xlabel('episode')
plt.ylabel('reward')
plt.savefig(ENVIRONMENT + '_reward')

plt.clf()
fig = plt.figure(1)
ax1 = fig.add_subplot(111)
ax1.semilogy(lossPerTimeframe[0], lossPerTimeframe[1], color='red')
# ax1.loglog(lossPerTimeframe[0], lossPerTimeframe[1], color='red')
# print(lossPerTimeframe[0])
plt.xlim(lossPerTimeframe[0][0], lossPerTimeframe[0][-1])
plt.title(ENVIRONMENT)
plt.xlabel('timeframe')
plt.ylabel('loss')
plt.savefig(ENVIRONMENT + '_loss')
# env.monitor.close()
