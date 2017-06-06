import gym
import numpy as np
import cv2

class Pong(object):

	def __init__(self):

		self.env = gym.make('PongDeterministic-v3')
		self.current_phi = None
		self.reset()

	def step(self, action):
		
		obs, r, done, info = self.env.step(action)
		obs = self._rbg2gray(obs)
		phi_next = self._phi(obs)

		phi_phi = np.vstack([self.current_phi, obs[np.newaxis]])
		self.current_phi = phi_next

		return phi_phi, r, done

	def reset(self):
		x = self.env.reset()
		x = self._rbg2gray(x)
		phi = np.stack([x, x, x, x])
		self.current_phi = phi

		return phi

	def _rbg2gray(self, x):
		x = x.astype('float32')
		x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
		x = cv2.resize(x, (84, 84))/127.5 - 1.

		return x

	def _phi(self, x):

		new_phi = np.zeros((4, 84, 84), dtype=np.float32)
		new_phi[:3] = self.current_phi[1:]
		new_phi[-1] = x
		
		return new_phi

	def display(self):
		self.env.render()
