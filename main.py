import numpy as np
from fd import FD

nn = 1
src = 0
homo = 1
lb = 1  # 0: dirichlet 1: neumann
rb = 0  # 0: dirichlet 1: neumann
dat = 0

if src:
    u = lambda x: np.exp(-0.1*(x-50) ** 2)
    v = 0.0
    lx = 60
    t = 180

    nx = 1000
    dx = 0.1

else:
    u = np.load('u.npy').flatten()
    v = np.load('v.npy').flatten()
    lx = 1.5
    t = 4

    nx = len(u)
    dx = 0.002

rho = 1.0
c = 1.0 if homo else lambda x: 1 if x < lx else 1.2
ref = None


def run(S):
    s = S(c, rho, nx, dx, t)
    s.dat = dat
    s.lb = lb
    s.rb = rb
    s.init(u, v)
    s.run()
    return s

fd = run(FD)

if nn:
    from pinn import PINN

    np.save('data_lb.npy', fd.field[:, 0])
    np.save('data_rb.npy', fd.field[:, -1])

    pinn = run(PINN)
    pinn.plot()
    print(f'{np.sum((pinn.field - fd.field) ** 2):.2e}')

else:
    fd.plot()
