import gym
import getch

def humanInput():

	char = getch.getch()
				
	if char == 'a':
		a = 0
	elif char == 's':
		a = 1
	elif char == 'd':
		a = 2

	return a

env = gym.make('MountainCar-v0')

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

num_episodes = 5

for episode in range(num_episodes):
    
    observation = env.reset()
    done = False
    step = 0
    actions = []
    observations = [] 

    while not done:
        env.render()
        #print(observation)
        action = humanInput()
        observation, reward, done, info = env.step(action)

        observations.append(observation)
        actions.append(action)
        step += 1
        
    print("Episode finished after {} steps".format(step))
            
