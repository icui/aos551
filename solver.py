import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec


class Solver:
    name: str

    nt: int
    nx: int
    dt: float
    dx: float
    
    c: np.ndarray
    rho: np.ndarray

    t: np.ndarray
    x: np.ndarray
    field: np.ndarray
    residual: np.ndarray | None = None

    lb: int
    rb: int
    dat: int

    def __init__(self, c, rho, nx, dx, tmax):
        self.dx = dx
        self.nx = nx
        self.x = np.arange(self.nx) * dx

        self.c = self._parse(c)
        self.rho = self._parse(rho)
        self.kappa = self.c ** 2 * self.rho

        self.dt = min(dx / self.c) / 2
        self.nt = int(np.ceil(tmax / self.dt))
        self.t = np.arange(self.nt) * self.dt

    def _parse(self, v):
        if callable(v):
            vx = np.zeros(self.nx)
            for i, x in enumerate(self.x):
                vx[i] = v(x)
            return vx
        
        if isinstance(v, (int, float)):
            return np.ones(self.nx) * v
        
        return v
    
    def init(self, u, v):
        self.u = self._parse(u)
        self.v = self._parse(v)
        self.field = np.zeros([self.nt, self.nx])

    def plot(self, nsnap=4):
        x, t = np.meshgrid(self.x, self.t)

        nrows = 3 if self.residual is not None else 2
        # nrows = 2
        plt.figure(figsize=(nsnap*3, nrows*2))
        gs = GridSpec(nrows, nsnap)
        # vmax = abs(self.u).max() / 2
        vmax = abs(self.field).max() / 2
        vmin = -vmax

        plt.subplot(gs[0, :])
        u = self.field.reshape(t.shape)
        plt.pcolormesh(t, x, u, cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax), shading='auto')
        plt.xlabel('t')
        plt.ylabel('x')
        cbar = plt.colorbar(pad=0.05, aspect=10)
        cbar.set_label('u(t,x)')
        cbar.mappable.set_clim(vmin, vmax)

        # t_cross_sections = list(range(0, self.nt, self.nt // (nsnap+1)))[1:1+nsnap]
        # for it in t_cross_sections:
        #     plt.axvline(it * self.dt)

        if self.residual is not None:
            plt.subplot(gs[1, :])
            u = self.residual.reshape(t.shape)
            plt.pcolormesh(t, x, u, cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax), shading='auto')
            plt.xlabel('t')
            plt.ylabel('x')
            cbar = plt.colorbar(pad=0.05, aspect=10)
            cbar.set_label('Δ(t,x)')
            cbar.mappable.set_clim(vmin, vmax)
        
        def lim(d):
            bottom = None if d.min() < vmin else vmin
            top = None if d.max() > vmax else vmax
            plt.ylim(bottom, top)

        # for i, it in enumerate(t_cross_sections):
        #     plt.subplot(gs[nrows-2, i])
        #     plt.plot(self.x, self.field[it])
        #     plt.title(f'u(t={it*self.dt:.1f})')
        #     plt.ylim(vmin, vmax)
        #     plt.xlabel('x')
        #     plt.tight_layout()

        # plt.subplot(gs[2, 0])
        # plt.plot(self.x, self.c)
        # plt.xlabel('x')
        # plt.title('c')
        # plt.tight_layout()

        # plt.subplot(gs[3, 0])
        # plt.plot(self.x, self.rho)
        # plt.xlabel('x')
        # plt.title('rho')
        # plt.tight_layout()

        plt.subplot(gs[nrows-1, 0])
        if self.residual is not None:
            plt.plot(self.x, self.field[0,:], label='predict')
            plt.plot(self.x, self.u, label='true')
            plt.legend()
        else:
            plt.plot(self.x, self.u)
        lim(self.u)
        plt.xlabel('x')
        plt.title('u(t=0)')
        plt.tight_layout()

        plt.subplot(gs[nrows-1, 1])
        if self.residual is not None:
            plt.plot(self.x, (self.field[1,:]-self.field[0,:])/self.dt, label='predict')
            plt.plot(self.x, self.v, label='true')
            plt.legend()
        else:
            plt.plot(self.x, self.v)
        lim(self.v)
        plt.xlabel('x')
        plt.title('v(t=0)')
        plt.tight_layout()

        plt.subplot(gs[nrows-1, 2])
        if self.residual is not None:
            plt.plot(self.t, self.field[:, 0], label='predict')
            plt.plot(self.t, np.load('data_lb.npy'), label='true')
            plt.legend()
        else:
            plt.plot(self.t, self.field[:, 0])
        lim(self.field[:,0])
        plt.xlabel('t')
        plt.title('u(x=0)')
        plt.tight_layout()

        plt.subplot(gs[nrows-1, 3])
        if self.residual is not None:
            plt.plot(self.t, self.field[:, -1], label='predict')
            plt.plot(self.t, np.load('data_rb.npy'), label='true')
            plt.legend()
        else:
            plt.plot(self.t, self.field[:, -1])
        lim(self.field[:,-1])
        plt.xlabel('t')
        plt.title(f'u(x={self.x[-1]:.1f})')
        plt.tight_layout()

        plt.savefig('u_' + self.name + '.png')
        # plt.show()
