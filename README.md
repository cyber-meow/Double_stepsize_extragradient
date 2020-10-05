# Double_stepsize_extragradient

For reproducing experimental results of the paper [Explore Aggressively, Update Conservatively: Stochastic Extragradient Methods with Variable Stepsize Scaling](https://arxiv.org/abs/2003.10162).

## Requirements

Python3, NumPy, SciPy, Autograd

## Usage

To reproduce the results for

### Bilinear games

```
python(3) main_quadratic_quardric.py\
  --algo=[algo] --nb_iterations=100000\
  --init_stepsize_gamma=[gamma1] --init_stepsize_eta=[eta1] --offset=[offset]\
  --save_dir=[log_dir]
```

### Strongly convex-concave problem (Bilinear+Quadratic+Quadric)

```
python(3) main_quadratic_quardric.py\
  --algo=[algo] --nb_iterations=100000\
  --q2a_coef=1 --q4a_coef=1 --q2b_coef=1 --q4b_coef=1\
  --init_stepsize_gamma=[gamma1] --init_stepsize_eta=[eta1] --offset=[offset]\
  --save_dir=[log_dir]
```

### Covariance matrix learning problem (A toy GAN model)

```
python(3) main_covariance_learning.py\
  --algo=[algo] --nb_iterations=1000000\
  --init_stepsize_gamma=[gamma1] --init_stepsize_eta=[eta1] --offset=[offset]\
  --save_dir=[log_dir]
```

---
In the above, `algo` is either `EG` or `OG`.
The choice of `gamma1`, `eta1` and `offset` are as specified in the paper.
After the execution of the script, several `.npy` files containing the
necessary information to plot the figures are generated in `log_dir`.
In more details, the evolution of the relevant convergence measure for each
single run is recorded.

<!---
As an example, we provide script `generate_figure_ex.py` to generate
the EG bilinear figure when the data are saved in the directory `bilinear_EG/`.
-->
