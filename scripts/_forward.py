import numpy as np
from tqdm import tqdm
from time import time
import pyshtools as sh


def compute_pot_acc(coeffs, r0, v):
    """Computes the gravitational potentials and acceleration for the Stokes coefficients in the matrix coeffs and at the points of spherical coordinates in v

    Args:
        coeffs (np.array): matrix of coefficients of size (2,n_max+1, n_max+1)
        r_0: reference radius (m)
        v (np.array): spherical coordinates (r, theta, phi) of the msr points

    Returns:
        pot (np.array): potential divided by gm
        acc (np.array): acceleration divided by gm
    """

    n_max = coeffs.shape[1] - 1  # deg max of the SH

    # reading spherical coordinates
    r = v[:, 0]
    theta = v[:, 1]
    lbd = v[:, 2]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # matrix of 4-pi-normalized associated Legendre polynomials and their first derivatives at each point
    Pmn_full = np.array([sh.legendre.PlmBar_d1(n_max + 1, z) for z in cos_theta])
    # separating polynomials (Pmn_z) and their derivatives (dPmn_z)
    Pmn_z = Pmn_full[:, 0, ...]
    dPmn_z = Pmn_full[:, 1, ...]

    u = np.zeros(r.size)  # vector of potential msr
    g = np.zeros((r.size, 3))  # matrix of acceleration msr

    # loop over degree and order
    for n in tqdm(range(n_max + 1)):
        t1 = (r0 / r) ** n
        for m in range(n + 1):
            # index in last dimension of Pmn matrix (see PlmBar_d1 docs)
            pmn_idx = (n * (n + 1)) // 2 + m

            # contribution of Cnm to potential
            u += (
                t1
                * Pmn_z[:, pmn_idx]
                * (
                    coeffs[0, n, m] * np.cos(m * lbd)
                    + coeffs[1, n, m] * np.sin(m * lbd)
                )
            )

            # contribution of Cnm to accleration components (in spherical coordinates)
            g[:, 0] += (
                -t1
                * (n + 1)
                * Pmn_z[:, pmn_idx]
                * (
                    coeffs[0, n, m] * np.cos(m * lbd)
                    + coeffs[1, n, m] * np.sin(m * lbd)
                )
            )
            g[:, 1] += (
                t1
                * dPmn_z[:, pmn_idx]
                * (-sin_theta)
                * (
                    coeffs[0, n, m] * np.cos(m * lbd)
                    + coeffs[1, n, m] * np.sin(m * lbd)
                )
            )
            g[:, 2] += (
                t1
                / sin_theta
                * Pmn_z[:, pmn_idx]
                * m
                * (
                    -coeffs[0, n, m] * np.sin(m * lbd)
                    + coeffs[1, n, m] * np.cos(m * lbd)
                )
            )

    # divide by r or r^2 at the end (but not gm)
    return u / r, g / (r**2)[:, None]


def pot_cnm_partials(coeffs, r0, v):
    """Computes the partials of the gravitational potential wrt the Stokes coefficients in the matrix coeffs and at the points of spherical coordinates in v

    Args:
        coeffs (np.array): matrix of coefficients of size (2,n_max+1, n_max+1)
        r_0: reference radius (m)
        v (np.array): spherical coordinates (r, theta, phi) of the msr points

    Returns:
        par_mat (np.array): matrix of partials, size (n_pts, 1, (n_max+1)**2)
        par_lbl (np.array): index of conversion from matrix to vector
    """
    n_max = coeffs.shape[1] - 1

    r = v[:, 0]
    theta = v[:, 1]
    lbd = v[:, 2]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    Pmn_z = np.array([sh.legendre.PlmBar(n_max + 1, z) for z in cos_theta])

    n_params = (n_max + 1) ** 2
    par_mat = np.zeros((r.size, 1, n_params))
    par_idx = 0
    par_lbl = np.zeros((n_params, 3), dtype=int)
    for n in range(n_max + 1):
        t1 = (r0 / r) ** n
        for m in range(n + 1):
            pmn_idx = (n * (n + 1)) // 2 + m

            par_mat[:, 0, par_idx] = t1 * Pmn_z[:, pmn_idx] * (np.cos(m * lbd))
            par_lbl[par_idx, :] = [0, n, m]
            par_idx += 1

            if m == 0:
                continue
            par_mat[:, 0, par_idx] = t1 * Pmn_z[:, pmn_idx] * (np.sin(m * lbd))
            par_lbl[par_idx, :] = [1, n, m]
            par_idx += 1
    return par_mat / r[:, None, None], par_lbl


def acc_cnm_partials(coeffs, r0, v):
    """Computes the partials of the gravitational acceleration wrt the Stokes coefficients in the matrix coeffs and at the points of spherical coordinates in v

    Args:
        coeffs (np.array): matrix of coefficients of size (2,n_max+1, n_max+1)
        r_0: reference radius (m)
        v (np.array): spherical coordinates (r, theta, phi) of the msr points

    Returns:
        par_mat (np.array): matrix of partials, size (n_pts, 3, (n_max+1)**2)
        par_lbl (np.array): index of conversion from matrix to vector
    """
    n_max = coeffs.shape[1] - 1

    r = v[:, 0]
    theta = v[:, 1]
    lbd = v[:, 2]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    Pmn_full = np.array([sh.legendre.PlmBar_d1(n_max + 1, z) for z in cos_theta])
    if Pmn_full.ndim == 2:
        Pmn_full = Pmn_full[np.newaxis, ...]
    Pmn_z = Pmn_full[:, 0, ...]
    dPmn_z = Pmn_full[:, 1, ...]

    n_params = (n_max + 1) ** 2
    par_mat = np.zeros((r.size, 3, n_params))
    par_idx = 0
    par_lbl = np.zeros((n_params, 3), dtype=int)
    for n in range(n_max + 1):
        t1 = (r0 / r) ** n
        for m in range(n + 1):
            pmn_idx = (n * (n + 1)) // 2 + m
            par_mat[:, 0, par_idx] = (
                -t1 * (n + 1) * Pmn_z[:, pmn_idx] * (np.cos(m * lbd))
            )
            par_mat[:, 1, par_idx] = (
                t1 * dPmn_z[:, pmn_idx] * (-sin_theta) * (np.cos(m * lbd))
            )
            par_mat[:, 2, par_idx] = (
                t1 / (sin_theta) * Pmn_z[:, pmn_idx] * m * (-np.sin(m * lbd))
            )
            par_lbl[par_idx, :] = [0, n, m]
            par_idx += 1

            if m == 0:
                continue
            par_mat[:, 0, par_idx] = (
                -t1 * (n + 1) * Pmn_z[:, pmn_idx] * (np.sin(m * lbd))
            )
            par_mat[:, 1, par_idx] = (
                t1 * dPmn_z[:, pmn_idx] * (-sin_theta) * (np.sin(m * lbd))
            )
            par_mat[:, 2, par_idx] = (
                t1 / (sin_theta) * Pmn_z[:, pmn_idx] * m * (np.cos(m * lbd))
            )
            par_lbl[par_idx, :] = [1, n, m]
            par_idx += 1
    return par_mat / (r**2)[:, None, None], par_lbl
