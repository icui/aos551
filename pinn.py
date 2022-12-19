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

    layers = [2, 32, 16, 16, 32, 1]
    layers_model = [1, 16, 16, 1]
    activation='tanh'

    ntrain = 10000
    batch = 15000
    weights = [1, 1, 1, 1, 1, 1, 1, 1]

    # factr = 1e5
    # m = 50
    # maxls = 50
    niter_adam = 1000
    niter_lbfgs = 500
    niter_model = 100
    model_threshold = 1e-6

    # def run(self):
    #     model = NN_model(self.layers)

    def run(self):
        self.nn = NN_wave(self.layers)
        # self.nn.load_state_dict(torch.load('nn_inv.pt'))
        # self.nn = torch.load('nn.pt')
        # self.nn_model.load_state_dict(torch.load('homo.pt'))
        # self.nn_model.load_state_dict(torch.load('inv.pt'))
        
        # x = np.linspace(0, 1, 500)
        # c = self.nn_model(torch.tensor(x.reshape([500,1]), dtype=torch.float))
        # import matplotlib.pyplot as plt
        # plt.plot(x, c.detach().numpy().flatten())
        # plt.savefig('inv.png')
        # exit()
        self.create_model()
        self.create_data()
        self.train()
        self.postprocess()

    def create_model(self):
        mse = torch.nn.MSELoss()

        dim = [self.nx, 1]
        x = torch.tensor(self.x.reshape(dim), dtype=torch.float)
        c = torch.tensor(self.c.reshape(dim), dtype=torch.float)

        for _ in range(10):
            self.nn_model = NN_model(self.layers_model)
            self.nn_model.train()
            opt = torch.optim.LBFGS(self.nn_model.parameters())

            def loss_model():
                loss = mse(self.nn_model(x), c)
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
            
        torch.save(self.nn_model.state_dict(), 'model_init.pt')

        import matplotlib.pyplot as plt
        plt.plot(x, c, label='model')
        plt.plot(x, self.nn_model(x).detach().numpy().flatten(), label='nn')
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
        self.nn.train()
        self.nn_model.train()

        # params = list(self.nn.parameters()) + list(self.nn_model.parameters())
        params = list(self.nn.parameters())
        # params = list(self.nn_model.parameters())

        self.optimizer = torch.optim.Adam(params, lr=0.005)
        
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

        u_eqn = self.nn(self.x_eqn, self.t_eqn)
        u_xx = self.grad(self.grad(u_eqn, self.x_eqn), self.x_eqn)
        u_tt = self.grad(self.grad(u_eqn, self.t_eqn), self.t_eqn)
        mse_eqn = mse(u_tt, self.ctx2(self.x_eqn) * u_xx)

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

        if self.dat:
            u_dat = self.nn(self.x_dat, self.t_dat)
            mse_dat = mse(u_dat, self.u_dat_true)
            loss += mse_dat

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)

        return loss

    def ctx2(self, x):
        return (self.nn_model(x) * self.t[-1] / self.x[-1]) ** 2

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
        self.nn_model.eval()

        torch.save(self.nn.state_dict(), 'nn.pt')
        torch.save(self.nn_model.state_dict(), 'model_inv.pt')

        # plt.figure()
        # xx = np.linspace(0, 1, 500)
        # cc = self.nn_model(torch.tensor(xx.reshape([500,1]), dtype=torch.float))
        # plt.plot(xx, cc.detach().numpy().flatten())

        # with torch.no_grad():
        #     u = self.nn(x, t)
        u = self.nn(x, t)
        u_xx = self.grad(self.grad(u, x), x)
        u_tt = self.grad(self.grad(u, t), t)
        u = u.detach().numpy().flatten()

        with torch.no_grad():
            res = (u_tt - self.ctx2(x) * u_xx).detach().numpy().flatten()
            # res = (u_tt / u_xx).detach().numpy().flatten()

        self.residual = np.zeros([self.nt, self.nx])

        for it in range(self.nt):
            self.field[it,:] = u[it*self.nx: (it+1)*self.nx]
            self.residual[it,:] = res[it*self.nx: (it+1)*self.nx]
        
        # self.residual[np.where(self.residual < 1)] = 1
        # self.residual = np.sqrt(self.residual) * self.x[-1] / self.t[-1]
        # self.residual[np.where(self.residual > 2)] = 2
        # self.residual[np.where(self.residual < 1)] = 1
        # res = np.mean(self.residual, axis=0)
        # plt.plot(self.x/self.x[-1], res)
        # plt.savefig('inv.png')
