from solver import Solver, np


class FD(Solver):
    name = 'fd'

    def run(self):
        u = self.u.copy()
        v = self.v.copy()
        T = np.gradient(u, self.dx)
        dtx2 = 0.5 * self.dt / self.dx

        for it in range(self.nt):
            v_old = v
            T_old = T

            v[1:-1] = v_old[1:-1] + dtx2 / self.rho[1:-1] * (T_old[2:] - T_old[:-2])
            T[1:-1] = T_old[1:-1] + dtx2 / self.kappa[1:-1] * (v_old[2:] - v_old[:-2])
            
            if self.lb:
                v[0] = v[1]
                T[0] = 0
            else:
                v[0] = 0
                T[0] = T[1]
            
            if self.rb:
                v[-1] = v[-2]
                T[-1] = 0
            
            else:
                v[-1] = 0
                T[-1] = T[-2]

            self.field[it] = u
            u += v * self.dt
