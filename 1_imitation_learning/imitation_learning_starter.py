#==========================================================================  
# Imitation Learning Starter Code 
# 
# The code was initially written for UCSB Deep Reinforcement Learning Seminar 2018
#
# Authors: Jieliang (Rodger) Luo, Sam Green
#
# April 9th, 2018
#==========================================================================

import gym
import getch

num_episodes = 5

# build neural network here 

# keyboard controls
# a -> left, s -> don't move, d -> right
def humanInput():

	char = getch.getch()
				
	if char == 'a':
		a = 0
	elif char == 's':
		a = 1
	elif char == 'd':
		a = 2

	return a

def main():

    # create environment
    env = gym.make('MountainCar-v0')

    print(env.action_space) # 3 actions: push left, no push, push right
    print(env.observation_space) # 2 observations: position, velocity 
    print(env.observation_space.high) # max position & velocity: 0.6, 0.07
    print(env.observation_space.low) # min position & velocity: -1.2, -0.07

    # initialize network, loss function, etc. here

    # learning from expert (user inputs)
    for episode in range(num_episodes):
        
        observation = env.reset()
        done = False
        step = 0
        actions = []
        observations = [] 

        while not done:
            env.render()
            print(observation)
            
            action = humanInput()
            observation, reward, done, info = env.step(action)

            # store all the observations and actions from one episode
            observations.append(observation)
            actions.append(action)
            
            step += 1
            
        print("Episode finished after {} steps".format(step))

        # train the network here 

    # evaluate the network when the training is finished 
            
if __name__ == "__main__":
    main()