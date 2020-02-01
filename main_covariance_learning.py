import os
import csv
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count

import numpy as np

from question_instances import LearnCovariance
from algorithms import JacDssExtraGrad, JacOG, ChambollePock


algos = {
    'EG': JacDssExtraGrad,
    'OG': JacOG,
    'CP': ChambollePock,
}


def main(params):

    d = params.dim
    lc = LearnCovariance(
            d=d, mu=params.mu, L=params.L,
            batch_size=params.batch_size)
    V1 = np.random.random([d, d])
    W1 = np.random.random([d, d])
    init_g = params.init_stepsize_gamma
    init_e = params.init_stepsize_eta
    init = init_e
    iters = params.nb_iterations
    algo = algos[params.algo]

    np.save(os.path.join(params.save_dir, 'COV'), lc.cov)

    # Execute one deterministic run
    opt = algo(V1, W1, lc.grad_V, lc.grad_W)
    opt.run(iters, init, init, dec=False)
    norm2 = opt.norm2_his(lc.grad_norm2)
    np.save(os.path.join(params.save_dir, 'deter'), norm2)

    # Stochastic runs
    dec_exponents = [
            (0, 0.7), (0.2, 0.7), (0.5, 0.7), (0.7, 0.7),
            (0.3, 0.6), (0.1, 0.8), (0, 0.9)]
    offset = params.offset

    if params.exe == 'seq':
        for dec_g, dec_e in dec_exponents:
            multi_runs(lc, algo, V1, W1, iters, offset,
                       init_g, init_e, dec_g, dec_e,
                       params.rp_times, params.save_dir)

    elif params.exe == 'para':
        pool = Pool(cpu_count())
        args = []
        for dec_g, dec_e in dec_exponents:
            args.append(
                (lc, algo, V1, W1, iters, offset,
                 init_g, init_e, dec_g, dec_e,
                 params.rp_times, params.save_dir))
        pool.map(multi_runs_para, args)

    elif params.exe == 'para_all':
        pool = Pool(cpu_count())
        args = []
        for dec_g, dec_e in dec_exponents:
            for k in range(params.rp_times):
                args.append(
                    (lc, algo, V1, W1, iters, offset,
                     init_g, init_e, dec_g, dec_e, k))
        results = pool.map(single_run_para, args)
        for k, (dec_g, dec_e) in enumerate(dec_exponents):
            norm2s = results[k*params.rp_times: (k+1)*params.rp_times]
            np.save(os.path.join(params.save_dir, f'stoch-{dec_g}-{dec_e}'),
                    np.array(norm2s))


def single_run(lc, algo, x1, y1, iters, offset,
               init_g, init_e, dec_g, dec_e, rd_seed):
    np.random.seed(rd_seed)
    gamma, eta = init_g*offset**dec_g, init_e*offset**dec_e
    opt = algo(
        x1, y1, lc.grad_V_stoch, lc.grad_W_stoch)
    opt.run(iters, gamma, eta,
            dec_g=dec_g, dec_e=dec_e, dec=True, offset=offset)
    norm2 = opt.norm2_his(lc.grad_norm2)
    return norm2, (dec_g, dec_e)


def single_run_para(args):
    return single_run(*args)[0]


def multi_runs(lc, algo, x1, y1, iters, offset,
               init_g, init_e, dec_g, dec_e, rp_times, save_dir):
    hiss = []
    for k in range(rp_times):
        norm2, _ = single_run(lc, algo, x1, y1, iters, offset,
                              init_g, init_e, dec_g, dec_e, k)
        hiss.append(norm2)
        np.save(os.path.join(save_dir, f'stoch-{dec_g}-{dec_e}'),
                np.array(hiss))
    return np.array(hiss)


def multi_runs_para(args):
    multi_runs(*args)


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument("--exe", type=str, default="para")

    parser.add_argument("--algo", type=str, default="EG")
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--mu", type=float, default=1)
    parser.add_argument("--L", type=float, default=2)

    parser.add_argument("--offset", type=int, default=50)

    parser.add_argument("--nb_iterations", type=int, default=10000)
    parser.add_argument("--init_stepsize_gamma", type=float, default=0.1)
    parser.add_argument("--init_stepsize_eta", type=float, default=0.05)
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
