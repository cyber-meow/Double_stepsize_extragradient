import numpy as np


def proj_simplex(x):
    """ Implement the paper
    Projection Onto A Simplex, Yunmei Chen, Xiaojin Ye, 2011
    https://arxiv.org/pdf/1101.6081.pdf
    """
    n = x.shape[0]
    # from big to small
    sorted_x = np.flip(np.sort(x))
    sum_x = np.cumsum(sorted_x)
    ts = (sum_x-1)/np.arange(1, n+1)
    t_idx = np.argmax(ts[:-1] - sorted_x[1:] >= 0)
    if t_idx == 0 and ts[0] < sorted_x[1]:
        t_idx = n-1
    return np.maximum(x-ts[t_idx], 0)


def proj_players(xy):
    x, y = np.split(xy, 2)
    return np.hstack([proj_simplex(x), proj_simplex(y)])


def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. sum_i w_i = s, w_i >= 0
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n))
    as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w
