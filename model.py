import torch
import torch.autograd as autograd
import torch.nn as nn

Variable = autograd.Variable

class DQN(nn.Module):

	def __init__(self):

		super(DQN, self).__init__()

		self.main = nn.Sequential(
			nn.Conv2d(4, 32, 8, 4, 0),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 64, 4, 2, 0),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, 3, 1, 0),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 512, 7, 4, 0),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 3, 1, 1, 0)
		)

	def forward(self, x):
		out = self.main(x).squeeze()
		return out
