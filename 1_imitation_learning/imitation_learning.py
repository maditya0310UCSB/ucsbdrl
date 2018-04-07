import gym
import getch
import torch

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

def humanInput():

	char = getch.getch()
				
	if char == 'a':
		a = 0
	elif char == 's':
		a = 1
	elif char == 'd':
		a = 2

	return a

class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(2, 10)
		self.fc2 = nn.Linear(10, 3)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return F.softmax(x, dim=1)

env = gym.make('MountainCar-v0')

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

num_episodes = 5

#setup the network
net = Net()

# build the loss function
criterion = nn.L1Loss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

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
		
		if(action == 0):
			actions.append([1, 0, 0])
		elif(action == 1):
			actions.append([0, 1, 0])
		elif(action == 2):
			actions.append([0, 0, 1])
		
		step += 1
		
	print("Episode finished after {} steps".format(step))

	# print(observations)
	# print(actions)

	# wrap them in Variable
	observations = torch.FloatTensor(observations)
	actions = torch.FloatTensor(actions)

	# print(observations)
	# print(actions)

	inputs, labels = Variable(observations), Variable(actions)

	# zero the parameter gradients
	optimizer.zero_grad()

	# forward + backward + optimize
	outputs = net(inputs)
	print(outputs)
	loss = criterion(outputs, labels)
	loss.backward()
	optimizer.step()

	print("Network updated!")

torch.save(net, "model.pkl")
			
