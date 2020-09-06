import numpy as np


class GsGDA(object):

    def __init__(self, x1, y1, vx, vy, proj=None):
        self.x = x1.copy()
        self.y = y1.copy()
        self.x_his = [x1.copy()]
        self.y_his = [y1.copy()]
        self.vx = vx
        self.vy = vy
        if proj is not None:
            self.proj = proj
        else:
            self.proj = lambda x: x

    def step(self, gamma):
        self.x = self.proj(self.x - gamma*self.vx(self.x, self.y))
        self.y = self.proj(self.y + gamma*self.vy(self.x, self.y))
        self.x_his.append(self.x.copy())
        self.y_his.append(self.y.copy())

    def run(self, n_iters, gamma, dec_rate=0, dec=True, offset=1):
        for k in range(n_iters):
            if dec:
                self.step(gamma/(k+offset)**dec_rate)
            else:
                self.step(gamma)


class JacGDA(GsGDA):

    def step(self, gamma):
        vx = self.vx(self.x, self.y)
        vy = self.vy(self.x, self.y)
        self.x = self.proj(self.x - gamma*vx)
        self.y = self.proj(self.y + gamma*vy)
        self.x_his.append(self.x.copy())
        self.y_his.append(self.y.copy())


class TwoStep(object):

    def __init__(self, x1, y1, vx, vy, proj=None):
        self.x = x1.copy()
        self.y = y1.copy()
        self.x_his = [x1.copy()]
        self.y_his = [y1.copy()]
        self.vx = vx
        self.vy = vy
        if proj is None:
            self.proj = lambda x: x
        else:
            self.proj = proj

    def step(self, gamma, eta):
        raise NotImplementedError

    def run(self, n_iters, gamma, eta,
            dec_g=1/3, dec_e=2/3, with_log=False, dec=True, offset=1):
        for k in range(n_iters):
            if dec:
                gamma_div = (k+offset)**dec_g
                eta_div = (k+offset)**dec_e
                if with_log:
                    eta_div *= np.log(k+offset+1)
                self.step(gamma/gamma_div, eta/eta_div)
            else:
                self.step(gamma, eta)

    def distance2_his(self, sol):
        pt_his = np.hstack([self.x_his, self.y_his])
        dis = np.linalg.norm(pt_his-sol, axis=-1)
        return dis**2

    def norm2_his(self, grad_norm2):
        norm2 = np.array(
                    [grad_norm2(x, y)
                        for x, y in zip(self.x_his, self.y_his)])
        return norm2


class GsDssExtraGrad(TwoStep):

    def step(self, gamma, eta):
        x_inter = self.proj(self.x - gamma * self.vx(self.x, self.y))
        self.y = self.proj(self.y + eta * self.vy(x_inter, self.y))
        y_inter = self.proj(self.y + gamma * self.vy(self.x, self.y))
        self.x = self.proj(self.x - eta * self.vx(self.x, y_inter))
        self.x_his.append(self.x.copy())
        self.y_his.append(self.y.copy())


class JacDssExtraGrad(TwoStep):

    def step(self, gamma, eta):
        x_inter = self.proj(self.x - gamma * self.vx(self.x, self.y))
        y_inter = self.proj(self.y + gamma * self.vy(self.x, self.y))
        self.x = self.proj(self.x - eta * self.vx(x_inter, y_inter))
        self.y = self.proj(self.y + eta * self.vy(x_inter, y_inter))
        self.x_his.append(self.x.copy())
        self.y_his.append(self.y.copy())


class GsOG(TwoStep):

    def __init__(self, x1, y1, vx, vy, proj=None):
        super().__init__(x1, y1, vx, vy, proj)
        # Oracle call at past step, initializing using x1, y1
        self.last_vx = vx(x1, y1)
        self.last_vy = vy(x1, y1)
        self.xm_his = [x1.copy()]
        self.ym_his = [y1.copy()]

    def step(self, gamma, eta):
        vx = self.vx(self.x, self.y)
        self.x = self.proj(self.x - eta*vx - gamma*(vx-self.last_vx))
        self.last_vx = vx
        vy = self.vy(self.x, self.y)
        self.y = self.proj(self.y + eta*vy + gamma*(vy-self.last_vy))
        self.last_vy = vy
        self.x_his.append(self.x.copy())
        self.y_his.append(self.y.copy())
        self.xm_his.append(self.x + gamma*vx)
        self.ym_his.append(self.y - gamma*vy)

    def distance2_his(self, sol):
        pt_his = np.hstack([self.x_his, self.y_his])
        dis = np.linalg.norm(pt_his-sol, axis=-1)
        ptm_his = np.hstack([self.xm_his, self.ym_his])
        dism = np.linalg.norm(ptm_his-sol, axis=-1)
        return np.vstack([dis**2, dism**2])

    def norm2_his(self, grad_norm2):
        norm2 = np.array(
                    [grad_norm2(x, y)
                        for x, y in zip(self.x_his, self.y_his)])
        norm2m = np.array(
                    [grad_norm2(x, y)
                        for x, y in zip(self.xm_his, self.ym_his)])
        return np.vstack([norm2, norm2m])


class JacOG(TwoStep):

    def __init__(self, x1, y1, vx, vy, proj=None):
        super().__init__(x1, y1, vx, vy, proj)
        # Oracle call at past step, initializing using x1, y1
        self.last_vx = vx(x1, y1)
        self.last_vy = vy(x1, y1)
        self.xm_his = [x1.copy()]
        self.ym_his = [y1.copy()]

    def step(self, gamma, eta):
        vx = self.vx(self.x, self.y)
        vy = self.vy(self.x, self.y)
        self.x = self.proj(self.x - eta*vx - gamma*(vx-self.last_vx))
        self.y = self.proj(self.y + eta*vy + gamma*(vy-self.last_vy))
        self.last_vx = vx
        self.last_vy = vy
        self.x_his.append(self.x.copy())
        self.y_his.append(self.y.copy())
        self.xm_his.append(self.x + gamma*vx)
        self.ym_his.append(self.y - gamma*vy)

    def distance2_his(self, sol):
        pt_his = np.hstack([self.x_his, self.y_his])
        dis = np.linalg.norm(pt_his-sol, axis=-1)
        ptm_his = np.hstack([self.xm_his, self.ym_his])
        dism = np.linalg.norm(ptm_his-sol, axis=-1)
        return np.vstack([dis**2, dism**2])

    def norm2_his(self, grad_norm2):
        norm2 = np.array(
                    [grad_norm2(x, y)
                        for x, y in zip(self.x_his, self.y_his)])
        norm2m = np.array(
                    [grad_norm2(x, y)
                        for x, y in zip(self.xm_his, self.ym_his)])
        return np.vstack([norm2, norm2m])


class ChambollePock(TwoStep):

    def step(self, gamma, eta):
        vx = self.vx(self.x, self.y)
        self.x = self.proj(self.x - eta * vx)
        x_inter = self.x - gamma * vx
        self.y = self.proj(self.y + eta * self.vy(x_inter, self.y))
        self.x_his.append(self.x.copy())
        self.y_his.append(self.y.copy())


def grad_x_norm2_seq(qu, opt):
    return np.array([qu.grad_x_norm2(x, y)
                    for (x, y) in zip(opt.x_his, opt.y_his)])


def grad_y_norm2_seq(qu, opt):
    return np.array([qu.grad_y_norm2(x, y)
                    for (x, y) in zip(opt.x_his, opt.y_his)])
