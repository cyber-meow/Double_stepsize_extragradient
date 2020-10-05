import os
import csv
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count

import numpy as np

from question_instances import QuadraticQuartic
from algorithms import JacDssExtraGrad, JacOG, ChambollePock


algos = {
    'EG': JacDssExtraGrad,
    'OG': JacOG,
    'CP': ChambollePock,
}


def main(params):

    qud = QuadraticQuartic(
            d=params.dim, lin_coef=params.lin_coef,
            q2a_coef=params.q2a_coef, q4a_coef=params.q4a_coef,
            q2b_coef=params.q2b_coef, q4b_coef=params.q4b_coef,
            mu=params.mu_ho, L=params.L_ho,
            mu_l=params.mu_l, L_l=params.L_l, sigma=params.noise_std)
    x1 = np.zeros(params.dim)
    x1[0] = 1
    y1 = x1.copy()
    # proj = proj_players if params.constraint_type == 'normal' else None
    init_g = params.init_stepsize_gamma
    init_e = params.init_stepsize_eta
    init = init_e
    iters = params.nb_iterations
    algo = algos[params.algo]
    offset = params.offset

    np.save(os.path.join(params.save_dir, 'C'), qud.C)
    np.save(os.path.join(params.save_dir, 'A2'), qud.A2)
    np.save(os.path.join(params.save_dir, 'B2'), qud.B2)
    np.save(os.path.join(params.save_dir, 'A4'), qud.A4)
    np.save(os.path.join(params.save_dir, 'B4'), qud.B4)

    # Execute one deterministic run
    opt = algo(x1, y1, qud.grad_x, qud.grad_y)
    opt.run(iters, init, init, dec=False)
    pt_his = np.hstack([opt.x_his, opt.y_his])
    sol = pt_his[-1]
    if params.constraint_type == 'unconstrained':
        sol = np.zeros(2*params.dim)
    dis2 = opt.distance2_his(sol)
    np.save(os.path.join(params.save_dir, 'deter'), dis2)

    # Stochastic runs
    dec_exponents = [
            (0, 0.7), (0.2, 0.7), (0.5, 0.7), (0.7, 0.7),
            (0.1, 0.8), (0, 0.9), (0.3, 0.9), (0.9, 0.9)]

    if params.exe == 'seq':
        for dec_g, dec_e in dec_exponents:
            multi_runs(qud, algo, x1, y1, sol, iters, offset,
                       init_g, init_e, dec_g, dec_e,
                       params.rp_times, params.save_dir)

    elif params.exe == 'para':
        pool = Pool(cpu_count())
        args = []
        for dec_g, dec_e in dec_exponents:
            args.append(
                (qud, algo, x1, y1, sol, iters, offset,
                 init_g, init_e, dec_g, dec_e,
                 params.rp_times, params.save_dir))
        pool.map(multi_runs_para, args)

    elif params.exe == 'para_all':
        pool = Pool(cpu_count())
        args = []
        for dec_g, dec_e in dec_exponents:
            for k in range(params.rp_times):
                args.append(
                    (qud, algo, x1, y1, sol, iters, offset,
                     init_g, init_e, dec_g, dec_e, k))
        results = pool.map(single_run_para, args)
        for k, (dec_g, dec_e) in enumerate(dec_exponents):
            dis2s = results[k*params.rp_times: (k+1)*params.rp_times]
            np.save(os.path.join(params.save_dir, f'stoch-{dec_g}-{dec_e}'),
                    np.array(dis2s))


def single_run(qud, algo, x1, y1, sol, iters, offset,
               init_g, init_e, dec_g, dec_e, rd_seed):
    np.random.seed(rd_seed)
    gamma, eta = init_g*offset**dec_g, init_e*offset**dec_e
    opt = algo(
        x1, y1, qud.grad_x_gaussian_noise, qud.grad_y_gaussian_noise)
    opt.run(iters, gamma, eta,
            dec_g=dec_g, dec_e=dec_e, dec=True, offset=offset)
    dist2 = opt.distance2_his(sol)
    return dist2, (dec_g, dec_e)


def single_run_para(args):
    return single_run(*args)[0]


def multi_runs(qud, algo, x1, y1, sol, iters, offset,
               init_g, init_e, dec_g, dec_e, rp_times, save_dir):
    hiss = []
    for k in range(rp_times):
        dist2, _ = single_run(qud, algo, x1, y1, sol, iters, offset,
                              init_g, init_e, dec_g, dec_e, k)
        hiss.append(dist2)
    np.save(os.path.join(save_dir, f'stoch-{dec_g}-{dec_e}'), np.array(hiss))
    return np.array(hiss)


def multi_runs_para(args):
    multi_runs(*args)


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument("--exe", type=str, default="para_all")

    parser.add_argument("--algo", type=str, default="EG")
    parser.add_argument("--dim", type=int, default=50)
    parser.add_argument("--noise_std", type=float, default=0.5)

    parser.add_argument("--lin_coef", type=float, default=1)
    parser.add_argument("--q2a_coef", type=float, default=0)
    parser.add_argument("--q4a_coef", type=float, default=0)
    parser.add_argument("--q2b_coef", type=float, default=0)
    parser.add_argument("--q4b_coef", type=float, default=0)

    parser.add_argument("--mu_l", type=float, default=1)
    parser.add_argument("--L_l", type=float, default=2)
    parser.add_argument("--mu_ho", type=float, default=1)
    parser.add_argument("--L_ho", type=float, default=2)

    parser.add_argument("--offset", type=int, default=1)

    parser.add_argument("--nb_iterations", type=int, default=10000)
    parser.add_argument("--init_stepsize_gamma", type=float, default=0.4)
    parser.add_argument("--init_stepsize_eta", type=float, default=0.4)
    parser.add_argument("--rp_times", type=int, default=10)

    parser.add_argument("--constraint_type", type=str,
                        default="unconstrained")
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
