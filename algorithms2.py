import numpy as np
from autograd import jacobian
from algorithms import GsGDA


class Hamiltonian(GsGDA):
    """Stochastic Hamiltonian Gradient Methods for Smooth Games
    This implementation is only for illustration purpose
    """

    def step(self, gamma, gamma_y=None):
        if gamma_y is None:
            gamma_y = gamma
        vx = self.vx(self.x, self.y)
        vy = self.vy(self.x, self.y)
        # Computing Jacobians
        jxx = jacobian(self.vx, 0)(self.x, self.y)
        jxy = jacobian(self.vx, 1)(self.x, self.y)
        jyx = jacobian(self.vy, 0)(self.x, self.y)
        jyy = jacobian(self.vy, 1)(self.x, self.y)
        # Update ((J^T)V)
        self.x = self.proj(self.x - gamma*(jxx@vx+jyx.T@vy))
        self.y = self.proj(self.y - gamma_y*(jxy.T@vx+jyy@vy))
        self.x_his.append(self.x.copy())
        self.y_his.append(self.y.copy())


class StochHamiltonian(object):

    def __init__(self, X1, Jv_stoch):
        self.X = X1.copy()
        self.X_his = [X1.copy()]
        self.Jv = Jv_stoch

    def step(self, gamma):
        J1, v1 = self.Jv(self.X)
        J2, v2 = self.Jv(self.X)
        self.X = self.X - gamma*(J1.T@v2+J2.T@v1)/2
        self.X_his.append(self.X.copy())

    def run(self, n_iters, gamma, dec_rate=0, dec=True, offset=1):
        for k in range(n_iters):
            if dec:
                self.step(gamma/(k+offset)**dec_rate)
            else:
                self.step(gamma)


class SVRE(object):

    def __init__(self, x1, y1, qu, repeat_sampling=False):
        self.x = x1.copy()
        self.y = y1.copy()
        self.x_his = [x1.copy()]
        self.y_his = [y1.copy()]
        self.qu = qu
        self.n = qu.n
        self.counter = 0
        self.repeat_sampling = repeat_sampling

    def step(self, gamma):
        if self.counter == 0:
            self.qu.get_perfect()
            self.x_epoch = self.x.copy()
            self.y_epoch = self.y.copy()
            self.vx_epoch = self.qu.grad_x(self.x_epoch, self.y_epoch)
            self.vy_epoch = self.qu.grad_y(self.x_epoch, self.y_epoch)
            self.counter = np.random.geometric(1/self.n)
            # print(self.counter)
        self.qu.draw_example()
        grad_x = self.qu.grad_x(self.x, self.y)
        grad_y = self.qu.grad_y(self.x, self.y)
        grad_dx, grad_dy = self.gradient_estimate(grad_x, grad_y)
        # print(grad_x == grad_dx)
        # print(grad_x)
        # print(grad_dx)
        x_inter = self.x - gamma * grad_dx
        y_inter = self.y + gamma * grad_dy
        if not self.repeat_sampling:
            self.qu.draw_example()
        self.qu.draw_example()
        grad_x = self.qu.grad_x(x_inter, y_inter)
        grad_y = self.qu.grad_y(x_inter, y_inter)
        grad_dx, grad_dy = self.gradient_estimate(grad_x, grad_y)
        self.x = self.x - gamma * grad_dx
        self.y = self.y + gamma * grad_dy
        self.counter -= 1
        self.x_his.append(self.x.copy())
        self.y_his.append(self.y.copy())

    def gradient_estimate(self, grad_x, grad_y):
        x_diff = self.qu.grad_x(self.x_epoch, self.y_epoch) - self.vx_epoch
        # print(x_diff)
        # print(x_diff == np.zeros_like(x_diff))
        grad_x = grad_x - x_diff
        # print('hi')
        # print(self.vx_epoch)
        # print(self.qu.grad_x(self.x_epoch, self.y_epoch))
        y_diff = self.qu.grad_y(self.x_epoch, self.y_epoch) - self.vy_epoch
        # print(y_diff)
        # print(y_diff == np.zeros_like(y_diff))
        grad_y = grad_y - y_diff
        return grad_x, grad_y

    def run(self, n_iters, gamma, dec_rate=0, dec=True, offset=1):
        for k in range(n_iters):
            if dec:
                self.step(gamma/(k+offset)**dec_rate)
            else:
                self.step(gamma)


class DssEG(object):

    def __init__(self, x1, y1, qu, repeat_sampling=False):
        self.x = x1.copy()
        self.y = y1.copy()
        self.x_his = [x1.copy()]
        self.y_his = [y1.copy()]
        self.qu = qu
        self.counter = 0
        self.repeat_sampling = repeat_sampling

    def step(self, gamma, eta):
        self.qu.draw_example()
        grad_x = self.qu.grad_x(self.x, self.y)
        grad_y = self.qu.grad_y(self.x, self.y)
        x_inter = self.x - gamma * grad_x
        y_inter = self.y + gamma * grad_y
        if not self.repeat_sampling:
            self.qu.draw_example()
        grad_x = self.qu.grad_x(x_inter, y_inter)
        grad_y = self.qu.grad_y(x_inter, y_inter)
        self.x = self.x - eta * grad_x
        self.y = self.y + eta * grad_y
        self.x_his.append(self.x.copy())
        self.y_his.append(self.y.copy())

    def run(self, n_iters, gamma, eta, dec_g=0, dec_e=1, dec=True, offset=1):
        for k in range(n_iters):
            if dec:
                gamma_div = (k+offset)**dec_g
                eta_div = (k+offset)**dec_e
                self.step(gamma/gamma_div, eta/eta_div)
            else:
                self.step(gamma, eta)
