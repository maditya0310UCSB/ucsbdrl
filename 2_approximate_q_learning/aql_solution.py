#==========================================================================  
# Approximate Q-Learning Working Code 
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
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

num_episodes = 2000
max_steps = 99

gamma = 0.99
epsilon = 0.1 # e-greedy policy 

# build a one-layer network, 16 inputs, 4 outputs
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(16, 4)

    def forward(self, x):
        x = self.fc1(x)
        return x

# convert state info to a one-hot vector
def to_one_hot_vector(index, total_indexes):
    return np.identity(int(total_indexes))[index:index+1]

def main():

    env = gym.make('FrozenLake-v0')

    print(env.action_space)
    print(env.observation_space) # type: Discrete

    q_function = Net()

    # print out weights 
    for param in q_function.parameters():
        print(param.data)

    optimizer = optim.SGD(q_function.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.MSELoss()

    rewards_list = []

    for episode in range(num_episodes):
        
        # Reset environment and get first new observation
        observation = env.reset() # type: numpy.int64
        rewards = 0
        done = False
        step = 0
        
        #The Q-Network
        while step < max_steps:
                        
            step += 1

            # the following two lines should be commented out during the training to speed the learning process
            # env.render()
            # time.sleep(1)

            # convert observation to a one-hot vector 
            observation_vector = to_one_hot_vector(observation, env.observation_space.n) # return type: numpy array
            # print(observation_vector)

            # torch.FloatTensor([x,y]) is different than torch.FloatTensor(x,y)
            Q_values = q_function(Variable(torch.FloatTensor(observation_vector))) # return type: PyTorch Variable

            # Choose an action by greedily (with e chance of random action) from the Q-network
            predicted_Q_value = np.max(Q_values.data.numpy())            
            action = np.argmax(Q_values.data.numpy())
            if np.random.rand(1) < epsilon and episode < 1000:
                action = env.action_space.sample()
            
            #Get new state and reward from environment
            observation_prime, reward, done, info = env.step(action)
            
            #Obtain the Q' values by feeding the new state through our network
            observation_vector_prime = to_one_hot_vector(observation_prime, env.observation_space.n)     
            Q_values_prime = q_function(Variable(torch.FloatTensor(observation_vector_prime)))
            
            #Obtain maxQ' and set our target value for chosen action.
            max_Q_prime = np.max(Q_values_prime.data.numpy())
            
            target_Q_value = np.copy(Q_values.data.numpy())
            # print(target_Q_value)

            if done:
                target_Q_value[0, action] = reward
            else:
                target_Q_value[0, action] = reward + gamma * max_Q_prime
            
            target_Q_value = Variable(torch.FloatTensor(target_Q_value))
            
            #Train our network using target and predicted Q values
            optimizer.zero_grad()
            loss = criterion(Q_values, target_Q_value)
            loss.backward()
            optimizer.step()

            rewards += reward
            observation = observation_prime
            
            if done == True:
                break

        rewards_list.append(rewards)

    plt.plot(rewards_list)
    plt.show()

if __name__ == "__main__":
    main()