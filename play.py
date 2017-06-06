from model import DQN
from pong import Pong
import torch
import torch.autograd as autograd
from utils import sample_action


the_model = torch.load('./tmp/model.pth')

dqn = DQN()
dqn.load_state_dict(the_model)

pong = Pong()
done = False

VALID_ACTION = [0, 2, 5]
var_phi = autograd.Variable(torch.Tensor(1, 4, 84, 84), volatile=True)

while(not done):

	act_index = sample_action(pong, dqn, var_phi)

	phi_next, r, done = pong.step(VALID_ACTION[act_index])
	pong.display()
