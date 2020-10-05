import os
import csv
from argparse import ArgumentParser

import numpy as np

from question_instances import QuadraticQuartic
from algorithms2 import StochHamiltonian


def main(params):

    qud = QuadraticQuartic(
            d=params.dim, lin_coef=1, q2_coef=0, q4_coef=0,
            mu_l=params.mu_l, L_l=params.L_l, sigma=params.noise_std)
    x1 = np.zeros(params.dim)
    x1[0] = 1
    y1 = x1.copy()
    init_g = params.init_stepsize_gamma
    iters = params.nb_iterations
    algo = StochHamiltonian
    offset = params.offset

    np.save(os.path.join(params.save_dir, 'C'), qud.C)

    # Execute one deterministic run
    qud.sigma = 0
    opt = algo(np.r_[x1, y1], qud.Jv)
    opt.run(iters, params.init_stepsize_gamma, dec=False)
    sol = np.zeros(2*params.dim)
    dis2 = opt.distance2_his(sol)
    np.save(os.path.join(params.save_dir, 'deter'), dis2)

    # Stochastic runs
    qud.sigma = params.noise_std
    multi_runs(qud, algo, x1, y1, sol, iters, offset,
               init_g, params.rp_times, params.save_dir)


def single_run(qud, algo, x1, y1, sol, iters, offset,
               init_g, rd_seed):
    np.random.seed(rd_seed)
    gamma = init_g*offset
    opt = algo(np.r_[x1, y1], qud.Jv)
    opt.run(iters, gamma, dec_rate=1, dec=True, offset=offset)
    dist2 = opt.distance2_his(sol)
    return dist2


def multi_runs(qud, algo, x1, y1, sol, iters, offset,
               init_g, rp_times, save_dir):
    hiss = []
    for k in range(rp_times):
        dist2 = single_run(
            qud, algo, x1, y1, sol, iters, offset, init_g, k)
        hiss.append(dist2)
    np.save(os.path.join(save_dir, 'stoch'), np.array(hiss))
    return np.array(hiss)


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument("--dim", type=int, default=50)
    parser.add_argument("--noise_std", type=float, default=0.5)

    parser.add_argument("--mu_l", type=float, default=1)
    parser.add_argument("--L_l", type=float, default=2)

    parser.add_argument("--offset", type=int, default=1)

    parser.add_argument("--nb_iterations", type=int, default=10000)
    parser.add_argument("--init_stepsize_gamma", type=float, default=0.4)
    parser.add_argument("--rp_times", type=int, default=10)

    parser.add_argument("--save_dir", type=str, default=".")

    params = parser.parse_args()

    if not os.path.exists(params.save_dir):
        os.makedirs(params.save_dir)
    csv_file = os.path.join(params.save_dir, 'params.csv')
    with open(csv_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in params.__dict__.items():
            writer.writerow([key, value])

    main(params)
