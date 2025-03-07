import numpy as np
from tqdm import tqdm
from time import time

from ._units import *


def compute_normal_equations(
    cnm_mat,
    sph_coords,
    partials_func,
    r_0=1.0 * m,
    batch_size=1000,
    msr_noise=1.0,
    partials_scale=1.0,
    raw_msr_vec=None,
    perturb_msr=False,
    rng=None,
):
    """Computes the normal matrix N and the vector y (right hand side of the normal equations).

    Args:
        cnm_mat (np.array): matrix of coefficients of size (2,n_max+1, n_max+1)
        sph_coords (np.array): spherical coordinates of the msr points
        partials_func (func): function for computing the partials given cnm_mat, sph_coords, and r_0
        r_0: reference radius (m)
        batch_size (int): number of measurements to process per batch
        msr_noise (float or np.array): std of the measurements
        partials_scale (float): constant factor that multiplies the partials (usually gm)
        raw_msr_vec (np.array): vector of synthetic measurements. If None, msr will be computed from cnm_mat
        perturb_msr (bool): add white noise to the synthetic data
        rng (np.random.Generator): random number generator


    Returns:
        N_mat (np.array): normal matrix
        y (np.array): normal equations RHS
        msr_vec (np.array): simulated measurements (raw_msr_vec if it's not None)
    """

    # Run the partials function on the first point to get dimensions of the problem and Cnm matrix-to-vector indices
    msr_partials_dummy, cnm_idx = partials_func(
        cnm_mat,
        r_0,
        sph_coords[:1, :],
    )
    cnm_idx = cnm_idx.astype(int)

    n_max = cnm_mat.shape[-1] - 1  # deg max to be estimated
    n_params = msr_partials_dummy.shape[-1]  # no. of parameters
    msr_dim = msr_partials_dummy.shape[1]  # dimensions of each measurement
    n_pts = sph_coords.shape[0]  # number of measurement points
    n_msr = n_pts * msr_dim  # total number of equations
    n_batches = n_pts // batch_size + 1  # number of batches

    cnm_vec = cnm_mat[*cnm_idx.T]  # converrting Cnm matrix to vector

    if perturb_msr and rng is None:
        rng = np.random.default_rng()

    N_mat = np.zeros((n_params, n_params))
    y = np.zeros(n_params)

    # Copy raw_msr_vec to output if not None
    # if raw_msr_vec is not None:
    #     msr_vec = raw_msr_vec
    # else:
    msr_vec = np.zeros(n_msr)

    # Check if msr_noise is a vector: a different (faster) computation if all msr have the same std (i.e. msr_noise is a scalar)
    use_w_mat = False
    if hasattr(msr_noise, "__len__"):
        use_w_mat = True

    for batch_idx in tqdm(range(n_batches)):
        pt_idx = (batch_idx * batch_size, (batch_idx + 1) * batch_size)
        msr_idx = (pt_idx[0] * msr_dim, pt_idx[1] * msr_dim)

        msr_partials, cnm_idx = partials_func(
            cnm_mat,
            r_0,
            sph_coords[pt_idx[0] : pt_idx[1], :],
        )
        msr_partials = (msr_partials * partials_scale).reshape(
            (msr_partials.shape[0] * msr_dim, msr_partials.shape[-1])
        )

        # if raw_msr_vec is None, compute measurements
        if raw_msr_vec is None:
            msr = msr_partials @ cnm_vec
        else:
            msr = raw_msr_vec[msr_idx[0] : msr_idx[1]]

        # Add noise to the data if required
        if perturb_msr:
            # Single value if msr_noise is scalar, else vector
            scale = msr_noise if not use_w_mat else msr_noise[msr_idx[0] : msr_idx[1]]
            msr += rng.normal(scale=scale, size=msr.size)

        # Set of (perturbed) synthetic measurements for this batch
        msr_vec[msr_idx[0] : msr_idx[1]] = msr

        # Compute contributions from batch (faster if msr_noise is a scalar)
        if use_w_mat:
            w_mat = np.diag(1 / msr_noise[msr_idx[0] : msr_idx[1]] ** 2)
            N_mat += msr_partials.T @ w_mat @ msr_partials
            y += msr_partials.T @ w_mat @ msr
        else:
            N_mat += msr_partials.T @ msr_partials / (msr_noise**2)
            y += msr_partials.T @ msr / (msr_noise**2)
    return N_mat, y, msr_vec


def solve_normal_equations(N_mat, y, compute_covariance=True):
    """Inverts the system of normal equations

    Args:
        N_mat (np.array): normal matrix
        y (np.array): normal equations RHS
        compute_covariance (bool): whether to also compute the covariance of the estimated parameters


    Returns:
        x_sol (np.array): estimated parameters
        cov (np.array or None): covariance
    """
    cov = None  # cov will stay None if compute_covariance is False
    if not compute_covariance:
        print("Solving via lstsq...")
        ts = time()
        # compute least-squares solution via numpy lstsq (faster but gives no covariance)
        x_sol = np.linalg.lstsq(N_mat, y, rcond=None)[0]
        print("Took {:.2f} s".format(time() - ts))

    else:
        print("Solving via SVD...")
        ts = time()
        # compute least-squares solution via SVD
        U, S, Vh = np.linalg.svd(N_mat, full_matrices=False)
        cov = Vh.T @ np.diag(S ** (-1)) @ U.T
        x_sol = cov @ y
        print("Took {:.2f} s".format(time() - ts))

    return x_sol, cov
