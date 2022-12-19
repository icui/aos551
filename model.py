import numpy as np
from scipy.interpolate import make_interp_spline
import torch
from torch import nn


class NN(nn.Module):
    def __init__(self, layers):
        super().__init__()

        seq = []

        for i, j in zip(layers, layers[1:]):
            seq.append(nn.Linear(i, j, bias=True))
            seq.append(nn.Tanh())

        self.nn = nn.Sequential(*seq[:-1])


class NN_model(NN):
    def forward(self, x):
        return self.nn(x)


m = NN_model([1, 16, 16, 1])
opt = torch.optim.LBFGS(m.parameters())
# opt = torch.optim.Adam(m.parameters(), lr=0.1)
mse = torch.nn.MSELoss()


x = np.load('x.npy')
c = np.load('c.npy')
x = x / x[-1]
dim = [len(x), 1]
x_t = torch.tensor(x.reshape(dim), dtype=torch.float)
c_t = torch.tensor(c.reshape(dim), dtype=torch.float)


def loss():
    loss = mse(m(x_t), c_t)
    opt.zero_grad()
    loss.backward()
    return loss


for epoch in range(100):
    l = loss()

    opt.step(loss)

    with torch.autograd.no_grad():
        print(epoch,"Traning Loss:",l.data)
    
# torch.save(m.state_dict(), 'hetero.pt')

import matplotlib.pyplot as plt
plt.plot(x, c, label='original')
plt.plot(x, m(x_t).detach().numpy().flatten(), label='nn')
plt.legend()
plt.savefig('model.png')
