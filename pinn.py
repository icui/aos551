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


class NN_wave(NN):
    def forward(self, x, t):
        return self.nn(torch.cat([x, t], dim=1))


class NN_model(NN):
    def forward(self, x):
        return self.nn(x)


class PINN(Solver):
    name = 'pinn'
    prefix = 'homo'

    load_wave = False
    load_model = False
    update_wave = True
    update_model = False

    layers_wave = [2, 32, 16, 16, 32, 1]
    layers_model = [1, 16, 16, 1]
    activation='tanh'
    x_scale = 1.0
    t_scale = 1.0

    ntrain = 10000
    batch = 15000
    weights = [1, 1, 1, 1, 1, 1, 1, 1]
    adam_lr = 0.005

    model_retry = 10
    niter_adam = 1000
    niter_lbfgs = 500
    niter_model = 100
    model_threshold = 1e-6

    def run(self):
        self.wave = NN_wave(self.layers_wave)
        self.model = NN_model(self.layers_model)
        self._x = self.x[-1] / self.x_scale
        self._t = self.t[-1] / self.t_scale

        self.create_data()
        
        if self.load_model:
            self.model.load_state_dict(torch.load(self.prefix + '_model.pt'))
        
        else:
            self.create_model()

        if self.load_wave:
            self.wave.load_state_dict(torch.load(self.prefix + '_wave.pt'))
        
        if self.update_wave or self.update_model:
            self.train()

        self.postprocess()

    def create_model(self):
        mse = torch.nn.MSELoss()

        dim = [self.nx, 1]
        x = torch.tensor((self.x / self._x).reshape(dim), dtype=torch.float)
        c = torch.tensor(self.c.reshape(dim), dtype=torch.float)

        for i in range(self.model_retry):
            self.model.train()
            opt = torch.optim.LBFGS(self.model.parameters())

            def loss_model():
                loss = mse(self.model(x), c)
                opt.zero_grad()
                loss.backward()
                return loss

            m = 1.0

            for epoch in range(self.niter_model):
                loss = loss_model()
                opt.step(loss_model)

                with torch.autograd.no_grad():
                    m = loss.data
                    print(epoch,"Model Loss:", m)
            
            if m < self.model_threshold:
                break
            
            print('model training failed, retrying...')
        
        if m >= self.model_threshold:
            raise RuntimeError('model training failed')
        
        torch.save(self.model.state_dict(), self.prefix + '_model.pt')

        import matplotlib.pyplot as plt
        plt.plot(x, c, label='model')
        plt.plot(x, self.model(x).detach().numpy().flatten(), label='nn')
        plt.legend()
        plt.savefig('model.png')

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
        self.hist = []
    
    def train(self):
        params = []

        if self.update_wave:
            self.wave.train()
            params += list(self.wave.parameters())

        if self.update_model:
            self.model.train()
            params += list(self.model.parameters())

        self.optimizer = torch.optim.Adam(params, lr=self.adam_lr)
        
        for epoch in range(self.niter_adam):
            loss = self.loss()
            self.optimizer.step()

            with torch.autograd.no_grad():
                print(epoch,"Adam Loss:",loss.data)
                self.hist.append(loss.data)
        
        self.optimizer = torch.optim.LBFGS(params, line_search_fn='strong_wolfe')
        
        for epoch in range(self.niter_lbfgs):
            loss = self.loss()
            self.optimizer.step(self.loss)

            with torch.autograd.no_grad():
                print(epoch,"LBFGS Loss:",loss.data)
                self.hist.append(loss.data)

    def grad(self, f, x):
        return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, retain_graph=True)[0]

    def loss(self):
        mse = torch.nn.MSELoss()

        u_eqn = self.wave(self.x_eqn, self.t_eqn)
        u_xx = self.grad(self.grad(u_eqn, self.x_eqn), self.x_eqn)
        u_tt = self.grad(self.grad(u_eqn, self.t_eqn), self.t_eqn)
        mse_eqn = mse(u_tt, self.ctx2(self.x_eqn) * u_xx)

        u_ini = self.wave(self.x_ini, self.t_ini)
        mse_ini =  mse(u_ini, self.u_ini_true) + mse(self.grad(u_ini, self.t_ini) / self._t, self.v_ini_true)

        u_lb = self.wave(self.x_lb, self.t_lb)
        if self.lb:
            mse_lb = mse(self.grad(u_lb, self.x_lb), self.zero)
        else:
            mse_lb = mse(u_lb, self.zero) + mse(self.grad(self.grad(u_lb, self.x_lb), self.x_lb), self.zero)

        u_rb = self.wave(self.x_rb, self.t_rb)
        if self.rb:
            mse_rb = mse(self.grad(u_rb, self.x_rb), self.zero)
        else:
            mse_rb = mse(u_rb, self.zero) + mse(self.grad(self.grad(u_rb, self.x_rb), self.x_rb), self.zero)

        loss = mse_eqn + mse_ini + mse_lb + mse_rb

        if self.dat:
            u_dat = self.wave(self.x_dat, self.t_dat)
            mse_dat = mse(u_dat, self.u_dat_true)
            loss += mse_dat

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)

        return loss

    def ctx2(self, x):
        return (self.model(x) * self._t / self._x) ** 2

    def postprocess(self):
        import matplotlib.pyplot as plt

        plt.plot(self.hist)
        plt.savefig('hist.png')

        x = np.tile(self.x / self._x, self.nt)
        t = np.tile(self.t / self._t, (self.nx, 1)).transpose().flatten()
        
        dim = (self.nx * self.nt, 1)
        x = torch.tensor(x.reshape(dim), requires_grad=True, dtype=torch.float)
        t = torch.tensor(t.reshape(dim), requires_grad=True, dtype=torch.float)
        
        self.wave.eval()
        self.model.eval()

        if self.update_wave:
            torch.save(self.wave.state_dict(), self.prefix + '_' + ('wave_new.pt' if self.load_wave else 'wave.pt'))
        
        if self.update_model:
            torch.save(self.model.state_dict(), self.prefix + '_model_new.pt')

        u = self.wave(x, t)
        u_xx = self.grad(self.grad(u, x), x)
        u_tt = self.grad(self.grad(u, t), t)
        u = u.detach().numpy().flatten()

        with torch.no_grad():
            res = (u_tt - self.ctx2(x) * u_xx).detach().numpy().flatten()

        self.residual = np.zeros([self.nt, self.nx])

        for it in range(self.nt):
            self.field[it,:] = u[it*self.nx: (it+1)*self.nx]
            self.residual[it,:] = res[it*self.nx: (it+1)*self.nx]
