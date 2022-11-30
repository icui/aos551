from solver import Solver, np
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

    def forward(self, x, t):
        return self.nn(torch.cat([x, t], dim=1))


class PINN(Solver):
    name = 'pinn'

    layers = [2, 32, 16, 16, 32, 1]
    activation='tanh'

    ntrain = 10000
    batch = 15000
    weights = [1, 1, 1, 1, 1, 1, 1, 1]

    factr = 1e5
    m = 50
    maxls = 50
    niters = 50

    def run(self):
        self.nn = NN(self.layers)
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=0.01)
        # optimizer = torch.optim.SGD(nn.parameters(), lr=0.01)
        # self.optimizer = torch.optim.LBFGS(self.nn.parameters(), history_size=10, line_search_fn='strong_wolfe')

        self.create_data()
        self.train()
        self.postprocess()
    
    def create_data(self):
        # training input
        dim = [self.ntrain, 1]

        self.x_eqn = torch.rand(dim, requires_grad=True)
        self.t_eqn = torch.rand(dim, requires_grad=True)

        self.x_ini = torch.rand(dim, requires_grad=False)
        self.t_ini = torch.zeros(dim, requires_grad=True)

        self.x_lb = torch.zeros(dim, requires_grad=True)
        self.t_lb = torch.rand(dim, requires_grad=False)

        self.x_rb = torch.ones(dim, requires_grad=True)
        self.t_rb = torch.rand(dim, requires_grad=False)

        self.x_dat = torch.zeros(dim, requires_grad=False)
        self.t_dat = torch.rand(dim, requires_grad=False)

        # training output
        def spl(s, f, x):
            return torch.tensor(make_interp_spline(s, f)(x.detach().flatten()).reshape([len(x), 1]), dtype=torch.float)

        x = np.linspace(0, 1, self.nx)
        t = np.linspace(0, 1, self.nt)

        self.u_ini_true = spl(x, self.u, self.x_ini)
        self.v_ini_true = spl(x, self.v, self.x_ini)
        self.u_dat_true = spl(t, np.load('data_lb.npy').flatten(), self.t_dat)

        self.zero = torch.zeros([self.ntrain, 1])
        self.ctx2 = (self.c[0] * self.t[-1] / self.x[-1]) ** 2
    
    def train(self):
        self.nn.train()

        self.hist = []

        for epoch in range(self.niters):
            loss = self.loss()
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step(self.loss)

            with torch.autograd.no_grad():
                print(epoch,"Traning Loss:",loss.data)
                self.hist.append(loss.data)
    
    def grad(self, f, x):
        return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, retain_graph=True)[0]

    def loss(self):
        mse = torch.nn.MSELoss()

        u_eqn = self.nn(self.x_eqn, self.t_eqn)
        u_xx = self.grad(self.grad(u_eqn, self.x_eqn), self.x_eqn)
        u_tt = self.grad(self.grad(u_eqn, self.t_eqn), self.t_eqn)
        mse_eqn = mse(u_tt, self.ctx2 * u_xx)

        u_ini = self.nn(self.x_ini, self.t_ini)
        mse_ini =  mse(u_ini, self.u_ini_true) + mse(self.grad(u_ini, self.t_ini) / self.t[-1], self.v_ini_true)

        u_lb = self.nn(self.x_lb, self.t_lb)
        if self.lb:
            mse_lb = mse(self.grad(u_lb, self.x_lb), self.zero)
        else:
            mse_lb = mse(u_lb, self.zero) + mse(self.grad(self.grad(u_lb, self.x_lb), self.x_lb), self.zero)

        u_rb = self.nn(self.x_rb, self.t_rb)
        if self.rb:
            mse_rb = mse(self.grad(u_rb, self.x_rb), self.zero)
        else:
            mse_rb = mse(u_rb, self.zero) + mse(self.grad(self.grad(u_rb, self.x_rb), self.x_rb), self.zero)

        loss = mse_eqn + mse_ini + mse_lb + mse_rb
        # loss = mse_eqn + mse_ini

        # if self.dat:
        #     u_dat = nn(x_dat, t_dat)
        #     mse_dat = mse(u_dat, u_dat_true)
        #     loss += mse_dat

        return loss
    
    def postprocess(self):
        import matplotlib.pyplot as plt

        plt.plot(self.hist)
        plt.savefig('hist.png')

        x = np.tile(self.x / self.x[-1], self.nt)
        t = np.tile(self.t / self.t[-1], (self.nx, 1)).transpose().flatten()
        
        dim = (self.nx * self.nt, 1)
        x = torch.tensor(x.reshape(dim), requires_grad=True, dtype=torch.float)
        t = torch.tensor(t.reshape(dim), requires_grad=True, dtype=torch.float)
        
        self.nn.eval()
        torch.save(self.nn, 'nn.pt')

        with torch.no_grad():
            u = self.nn(x, t)

        # v = self.grad(u, t)
        # u_xx = self.grad(self.grad(u, x), x)
        # u_tt = self.grad(v, t)
        u = u.detach().numpy().flatten()

        # with torch.no_grad():
        #     res = (u_tt - self.ctx2 * u_xx).detach().numpy().flatten()

        # self.v_pred = v[:self.nx].detach().numpy().flatten()
        self.residual = np.zeros([self.nt, self.nx])

        for it in range(self.nt):
            self.field[it,:] = u[it*self.nx: (it+1)*self.nx]
            # self.residual[it,:] = res[it*self.nx: (it+1)*self.nx]
