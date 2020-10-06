import os
import csv
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count

import numpy as np

from question_instances import Bilinear
from algorithms import JacDssExtraGrad
from algorithms_supp import StochHamiltonian, Anchoring


def main(params):

    qud = Bilinear(d=params.dim, mu=params.mu,
                   L=params.L, sigma=params.noise_std)
    x1 = np.zeros(params.dim)
    x1[0] = 1
    y1 = x1.copy()
    init_lr = 0.1
    iters = params.nb_iterations
    offset = 20
    sol = np.zeros(2*params.dim)

    np.save(os.path.join(params.save_dir, 'C'), qud.C)

    # Stochastic runs
    algos = ['Hamiltonian', 'DSEG', 'Anchoring']
    pool = Pool(cpu_count())
    args = []
    for algo in algos:
        for k in range(params.rp_times):
            args.append((qud, algo, x1, y1, sol, iters, offset, init_lr, k))

    results = pool.map(single_run_para, args)
    for k, algo in enumerate(algos):
        dis2s = results[k*params.rp_times: (k+1)*params.rp_times]
        np.save(os.path.join(
            params.save_dir, f'stoch-{algo}'), np.array(dis2s))


def single_run(qud, algo, x1, y1, sol, iters, offset, init_lr, rd_seed):
    np.random.seed(rd_seed)
    if algo == 'Hamiltonian':
        opt = StochHamiltonian(np.r_[x1, y1], qud.Jv)
        opt.run(iters, init_lr*offset, dec_rate=1, dec=True, offset=offset)
    if algo == 'DSEG':
        opt = JacDssExtraGrad(x1, y1, qud.grad_x, qud.grad_y)
        opt.run(iters, 1, init_lr*offset,
                dec_g=0, dec_e=1, dec=True, offset=offset)
    if algo == 'Anchoring':
        opt = Anchoring(np.r_[x1, y1], qud.vec)
        opt.run(iters, 0.3, 0.3, dec_gamma=0.7, dec_reg=0.9, offset=1)
    return opt.distance2_his(sol)


def single_run_para(args):
    return single_run(*args)


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

    parser.add_argument("--mu", type=float, default=1)
    parser.add_argument("--L", type=float, default=2)

    parser.add_argument("--nb_iterations", type=int, default=10000)
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
