import autograd.numpy as np
from autograd import grad
from scipy.stats import ortho_group


# def random_posdef(d, mu):
#     A = np.random.random([d, d])/d
#     return A@A.T + mu*np.eye(d)

def random_posdef(d, mu, L):
    Ort = ortho_group.rvs(d)
    D = np.diagflat(np.random.uniform(low=mu, high=L, size=d))
    return Ort@D@Ort.T


def random_mat(d, mu, L):
    Ort1 = ortho_group.rvs(d)
    Ort2 = ortho_group.rvs(d)
    D = np.diagflat(np.random.uniform(low=mu, high=L, size=d))
    return Ort1@D@Ort2.T


class QuadraticQuartic(object):

    def __init__(self, d, mu=1, L=2, mu_l=1, L_l=2,
                 lin_coef=1, q2_coef=0, q4_coef=0,
                 q2a_coef=None, q2b_coef=None,
                 q4a_coef=None, q4b_coef=None, sigma=0.1):
        self.A2 = random_posdef(d, mu, L)
        self.A4 = random_posdef(d, mu, L)
        self.B2 = random_posdef(d, mu, L)
        self.B4 = random_posdef(d, mu, L)
        # self.C = np.random.random([d, d])/d
        self.C = random_mat(d, mu_l, L_l)
        self.lin_coef = lin_coef
        self.q2a_coef = q2_coef if q2a_coef is None else q2a_coef
        self.q2b_coef = q2_coef if q2b_coef is None else q2b_coef
        self.q4a_coef = q4_coef if q4a_coef is None else q4a_coef
        self.q4b_coef = q4_coef if q4b_coef is None else q4b_coef
        self.d = d
        self.sigma = sigma

    def objective(self, x, y):
        obj = 0
        if self.q2a_coef != 0:
            obj += self.q2a_coef*x@self.A2@x/2
        if self.q2b_coef != 0:
            obj -= self.q2b_coef*y@self.B2@y/2
        if self.q4a_coef != 0:
            obj += self.q4a_coef*((x@self.A4@x)**2)/4
        if self.q4b_coef != 0:
            obj -= self.q4b_coef*((y@self.B4@y)**2)/4
        if self.lin_coef != 0:
            obj += self.lin_coef*x@self.C@y
        return obj

    def grad_x(self, x, y):
        return grad(self.objective, 0)(x, y)

    def grad_y(self, x, y):
        return grad(self.objective, 1)(x, y)

    # Only for bilinear games
    def Jv(self, X, jac_noise=False):
        mat_lin = self.C
        if jac_noise:
            mat_lin += np.random.randn(self.d, self.d) * self.sigma
        mat = np.block([
            [np.zeros([self.d, self.d]), mat_lin],
            [-mat_lin.T, np.zeros([self.d, self.d])]
        ])
        noise = np.random.randn(2*self.d) * self.sigma
        return mat, mat@X+noise

    def grad_x_gaussian_noise(self, x, y):
        noise = np.random.randn(self.d) * self.sigma
        return self.grad_x(x, y) + noise

    def grad_y_gaussian_noise(self, x, y):
        noise = np.random.randn(self.d) * self.sigma
        return self.grad_y(x, y) + noise


class SmoothL1Regression(object):
    """Ref: Linear Convergence of the Primal-Dual Gradient Method ...
    """

    def __init__(self, d, mu_l=1, L_l=2, lamb=0.1, a=10, sigma=0.1):
        self.M = random_mat(d, mu_l, L_l)
        self.v = np.random.random(d)
        self.d = d
        self.lamb = lamb
        self.a = a
        self.sigma = sigma
        self.grad_x = grad(self.objective, 0)
        self.grad_y = grad(self.objective, 1)

    def smooth_L1(self, x):
        l1_cord = (np.log(1+np.exp(self.a*x))
                   + np.log(1+np.exp(-self.a*x)))/self.a
        return np.sum(l1_cord)

    def objective(self, x, y):
        obj = 0
        obj += x@self.M@y - self.v@y
        obj -= y@y/2
        obj += self.smooth_L1(x)
        return obj

    def grad_x_gaussian_noise(self, x, y):
        noise = np.random.randn(self.d) * self.sigma
        return self.grad_x(x, y) + noise

    def grad_y_gaussian_noise(self, x, y):
        noise = np.random.randn(self.d) * self.sigma
        return self.grad_y(x, y) + noise


class LearnCovariance(object):

    def __init__(self, d, mu=0, L=1, batch_size=1):
        self.d = d
        self.cov = random_posdef(d, mu, L)
        self.z_v_used = True
        self.z_w_used = True
        self.batch_size = batch_size

    def grad_V(self, V, W):
        return -(W+W.T)@V

    def grad_W(self, V, W):
        return self.cov - V@V.T

    def grad_x_norm2(self, V, W):
        return np.sum(self.grad_V(V, W)**2)

    def grad_y_norm2(self, V, W):
        return np.sum(self.grad_V(V, W)**2)

    def grad_norm2(self, V, W):
        return self.grad_x_norm2(V, W) + self.grad_y_norm2(V, W)

    def grad_V_stoch(self, V, W):
        if self.z_v_used:
            self.z = np.random.randn(self.batch_size, self.d)
            self.z_w_used = False
        self.z_v_used = True
        return -(W+W.T)@V@(self.z.T@self.z)/self.batch_size

    def grad_W_stoch(self, V, W):
        x = np.random.multivariate_normal(
                np.zeros(self.d), self.cov, size=self.batch_size)
        if self.z_w_used:
            self.z = np.random.randn(self.batch_size, self.d)
            self.z_v_used = False
        self.z_w_used = True
        return (x.T@x - V@(self.z.T@self.z)@V.T)/self.batch_size


class FiniteSumBilinear(object):

    def __init__(self, n, d, mu=1, L=2, sigma=0.1, sigma_dis=1):
        self.n = n
        self.d = d
        self.A_avg = random_mat(d, mu, L)
        self.As = self.A_avg + sigma * np.random.randn(n, d, d)
        self.bs = np.random.randn(n, d) / d * sigma_dis
        self.cs = np.random.randn(n, d) / d * sigma_dis
        self.b_avg = np.mean(self.bs, axis=0)
        self.c_avg = np.mean(self.cs, axis=0)
        self.sol_x = np.linalg.solve(self.A_avg.T, -self.c_avg)
        self.sol_y = np.linalg.solve(self.A_avg, -self.b_avg)

    def get_perfect(self):
        self.A_curr = self.A_avg
        self.b_curr = self.b_avg
        self.c_curr = self.c_avg

    def draw_example(self):
        k = np.random.randint(self.n)
        self.A_curr = self.As[k]
        self.b_curr = self.bs[k]
        self.c_curr = self.cs[k]

    def grad_x(self, x, y):
        return self.A_curr@y + self.b_curr

    def grad_y(self, x, y):
        return self.A_curr.T@x + self.c_curr
