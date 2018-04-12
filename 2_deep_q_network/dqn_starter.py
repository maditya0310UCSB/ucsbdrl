#==========================================================================  
# Deep Q-Learning Starter Code 
# 
# The code was initially written for UCSB Deep Reinforcement Learning Seminar 2018
#
# Authors: Jieliang (Rodger) Luo, Sam Green
#
# April 11th, 2018
#==========================================================================

import gym
import numpy as np
import random
import time

num_episodes = 2000
max_steps = 99

epsilon = 1 # change to 0.1 when you implement DQN

def main():

    env = gym.make('FrozenLake-v0')

    print(env.action_space)
    print(env.observation_space)

    for episode in range(num_episodes):
        
        # Reset environment and get first new observation
        observation = env.reset()
        rewards = 0
        done = False
        step = 0
        
        #The Q-Network
        while step < max_steps:
                        
            step +=1

            # the following two lines should be commented out during the trainning to fasten the process
            env.render()
            time.sleep(1)

            # Choose an action by greedily (with e chance of random action) from the Q-network

            if np.random.rand(1) < epsilon:
                action = env.action_space.sample()
            
            #Get new state and reward from environment
            observation_new, reward, done, info = env.step(action)
            
            #Obtain the Q' values by feeding the new state through our network

            
            #Obtain maxQ' and set our target value for chosen action.

            
            #Train our network using target and predicted Q values

            rewards += reward
            observation = observation_new
            
            if done == True:
                
                #Reduce chance of random action as we train the model.
                print("episode {} ends".format(episode))
                print("rewards are {}".format(rewards))

                break

if __name__ == "__main__":
    main()