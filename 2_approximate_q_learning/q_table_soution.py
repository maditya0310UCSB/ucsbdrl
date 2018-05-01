#==========================================================================  
# Q-Table Learning Working Code 
# 
# The code was initially written for UCSB Deep Reinforcement Learning Seminar 2018
#
# Authors: Jieliang (Rodger) Luo, Sam Green
#
# April 17th, 2018
#
# Adapted from Arthur Juliani's code: 
# https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
#==========================================================================

import gym
import numpy as np
import random
import time
import matplotlib.pyplot as plt # run "conda update matplotlib" if you have trouble importing this library

num_episodes = 2000
max_steps = 99

epsilon = 0.1
learning_rate = 0.8
alpha = 0.95 #discounted factor

def main():

    env = gym.make('FrozenLake-v0')

    print(env.action_space)
    print(env.observation_space) # discrete finite observations

    # initialize the table with all zeros
    q_table = np.zeros([env.observation_space.n,env.action_space.n])
    rewards_list = []

    for episode in range(num_episodes):
        
        # Reset environment and get first new observation
        observation = env.reset()
        rewards = 0
        done = False
        step = 0

        # Q-Table learning algorithm 
        while step < max_steps:
                        
            step += 1

            # env.render()
            # time.sleep(1)

            # add noise to the table
            if episode < 500:
                action = np.argmax(q_table[observation,:] + np.random.randn(1,env.action_space.n)*(1./(episode+1)))
            else:
                action = np.argmax(q_table[observation,:])
            
            # (1) random walking if all the values in a state is zero; (2) decay epsilon;  
            # action = np.argmax(q_table[observation,:])
            # if (np.random.rand(1) < epsilon or np.count_nonzero(q_table[observation,:]) == 0) and episode < 200:
            # # if (np.random.rand(1) < np.exp(-0.001*episode)) and episode <200:
            #     action = env.action_space.sample()

            
            #Get new state and reward from environment
            observation_new, reward, done, info = env.step(action)
            
            # Update Q-Table
            if done:
                q_table[observation, action] = reward
            else:
                q_table[observation, action] += learning_rate * (reward + alpha * np.max(q_table[observation_new, :]) - q_table[observation, action]) 

            rewards += reward
            observation = observation_new
            
            if done == True:
                break

        rewards_list.append(rewards)

    print("Final Q-Table Values:")
    print(q_table)

    plt.plot(rewards_list)
    plt.show()

if __name__ == "__main__":
    main()