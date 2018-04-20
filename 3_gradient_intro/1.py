import numpy as np
import random
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import ipdb

# build a one-layer network
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(16, 4)

    def forward(self, x):
        x = self.fc1(x)
        return x


# Instantiate the Net class. It takes a 16-entry vector as an input and returns a 4-entry vector
net = Net()

# Set the first entry to 1. and the rest to 0.
# [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
x = np.identity(16)[0]
print(x)

# First convert x to PyTorch Variable
x = Variable(torch.FloatTensor(x))

# Let's see what our randomized network thinks x is
# y_hat will have four entries (because of the way we defined class Net at the top)
y_hat = net(x)
print(y_hat)

# Now lets train the network to associate x with some specific value (934) at the 3rd entry
y = Variable(torch.FloatTensor([934]))

# This instantiates a PyTorch loss function object
loss_function = nn.MSELoss()

learning_rate = .01

for epoch in range(1000):
	y_hat = net(x)
	print(y_hat)

	# Calculate the current loss (want y_hat[2] to be equal to 934)
	loss = loss_function(y_hat[2], y)
	print(loss)

	# Standard bookkeeping. Zero the gradients before running the backward pass
	net.zero_grad()

	# Calculate the gradient of the loss with respect to the NN parameters.
	# The partial derivatives are stored in each parameter (accessed below)
	loss.backward()

	# Update the weights using gradient descent
	for param in net.parameters():
		param.data -= learning_rate * param.grad.data