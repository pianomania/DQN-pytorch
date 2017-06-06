import numpy as np
import random
import torch
import matplotlib.pyplot as plt


def save_statistic(ylabel, nums, std=None, save_path=None):

	n = np.arange(len(nums))

	plt.figure()
	plt.plot(n, nums)
	if std is not None:
		nums = np.array(nums)
		std = np.array(std)
		plt.fill_between(n, nums+std, nums-std, facecolor='blue', alpha=0.1)
	plt.ylabel(ylabel)
	plt.xlabel('Episodes')
	plt.savefig(save_path + '/' + ylabel + '.png')
	plt.close()

def sample_action(env, agent, var_phi, epsilon):

	if random.uniform(0,1) > epsilon:
		phi = env.current_phi
		var_phi.data.copy_(torch.from_numpy(phi))

		q_values = agent(var_phi)
		max_q, act_index = q_values.max(dim=0)
		act_index = np.asscalar(act_index.data.cpu().numpy())
	else:
		act_index = random.randrange(3)

	return act_index