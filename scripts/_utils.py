import numpy as np
import torch

import pyshtools as sh
import json
import pickle as pk

from ._units import *


def read_SHADR(file_path):
    """Read Stokes coefficients from SHADR text file

    Args:
        file_path (str): file to read

    Returns:
        c_mat (np.array): matrix of coefficients of size (2,n_max+1, n_max+1)
        ref_radius(float): reference radius from the file
        ref_gm (float): gravitational parameter from the file
    """
    with open(file_path) as f:
        data = f.read()

    data = data.split("\n")
    headerRow = [float(el) for el in data[0].split(",")]
    shadrMat = []
    ii = 0
    for row in data[1:-1]:
        a = row.split(",")
        if len(a) > 2:
            shadrMat.append([float(el) for el in a])
            ii += 1
    shadrMat = np.array(shadrMat)
    ref_radius = headerRow[0] * 1e3  # m
    ref_gm = headerRow[1] * 1e9  # m**3/s**2
    n_max = int(headerRow[3])

    c_mat = np.zeros((2, n_max + 1, n_max + 1))
    c_mat[0, 0, 0] = 1

    for shadrEl in shadrMat:
        c_mat[0, int(shadrEl[0]), int(shadrEl[1])] = shadrEl[2]
        c_mat[1, int(shadrEl[0]), int(shadrEl[1])] = shadrEl[3]

    return c_mat, ref_radius, ref_gm


def write_SHADR(file_path, cnm_vec, cnm_map, gm=1, r_0=1, cnm_err=None):
    """Store a vector of Stokes coefficients in a SHADR-formatted text file

    Args:
        file_path (str): file to write to
        cnm_vec (np.array): vector of coefficients
        cnm_map (dict): map of (0/1, n, m) matrix indices to cnm vector indices
        gm (float): gravitational parameter for file header
        r_0 (float): reference radius for file header
        cnm_err (np.array): standard deviations of the coefficients
    """
    cnm_map_keys = np.array(list(cnm_map.keys()))
    n_max = np.max(cnm_map_keys[:, 1])

    headerRow = [r_0, gm, 0.0, n_max, n_max]
    rows = [
        "{:23.16E},{:23.16E},{:23.16E},{:5d},{:5d},{:5d}".format(
            r_0, gm, 0.0, n_max, n_max, 1
        )
    ]
    for n in range(1, n_max + 1):
        for m in range(n + 1):
            csbar = np.array([0.0, 0.0])
            for cs in [0, 1]:
                cnm_idx = cnm_map.get((cs, n, m), None)
                if cnm_idx is None:
                    csbar[cs] = 0.0
                else:
                    csbar[cs] = cnm_vec[cnm_idx]
            rows.append("{:5d},{:5d},{:23.16E},{:23.16E}".format(n, m, *csbar))
    with open(file_path, "w+") as f:
        f.write("\n".join(rows))


def get_cnm_idx(n_max):
    """Compute mappings from vector to matrix of Stokes coefficients and vice-versa

    Args:
        n_max (int): maximum SH degree

    Returns:
        cnm_idx (np.array): mapping from vector to matrix indices
        cnm_map (dict): mapping from matrix to vector indices
    """
    n_coeffs = (n_max + 1) ** 2
    cnm_idx = np.zeros((n_coeffs, 3), dtype=int)
    cnm_map = {}
    par_idx = 0
    for n in range(n_max + 1):
        for m in range(n + 1):
            idx_tuple = (0, n, m)
            cnm_idx[par_idx, :] = idx_tuple
            cnm_map[idx_tuple] = par_idx
            par_idx += 1
            if m == 0:
                continue
            idx_tuple = (1, n, m)
            cnm_idx[par_idx, :] = idx_tuple
            cnm_map[idx_tuple] = par_idx
            par_idx += 1
    return cnm_idx, cnm_map


def save_cnm_json(file_path, cnm_vec, cnm_idx, gm=1, r_0=1):
    """Store vector of coefficients as a JSON file

    Args:
        file_path (str): destination file
        cnm_vec (np.array): vector of coefficients
        cnm_idx (np.array): map of cnm vector indices to (0/1, n, m) matrix indices
        gm (float): gravitational parameter for file header
        r_0 (float): reference radius for file header
    """

    coeffs_list = []
    for lmn, cnm in zip(cnm_idx.tolist(), cnm_vec.tolist()):
        coeffs_list.append([lmn, cnm])
    data = {"GM (m^3/s^2)": gm, "R0 (m)": r_0, "Stokes coefficients": coeffs_list}
    with open(file_path, "w+") as out_file:
        json.dump(data, out_file, indent=4)


def read_cnm_json(file_path):
    """Read vector of coefficients from a JSON file

    Args:
        file_path (str): file to read

    Returns:
        cnm_mat (np.array): coefficients matrix
        r_0 (float): reference radius
        gm (float): gravitational parameter
    """

    with open(file_path, "r") as f:
        data = json.load(f)
    cnm_idx = []
    cnm_vec = []
    for row in data["Stokes coefficients"]:
        cnm_idx.append(row[0])
        cnm_vec.append(row[1])
    cnm_idx = np.array(cnm_idx)
    cnm_vec = np.array(cnm_vec)

    n_max = np.max(cnm_idx[:, 1])
    cnm_mat = np.zeros((2, n_max + 1, n_max + 1))
    cnm_mat[*cnm_idx.T] = cnm_vec
    return cnm_mat, data["R0 (m)"], data["GM (m^3/s^2)"]


def save_msr_grid(file_path, coords, vals):
    """Stores coordinates and values of measurements over a grid in a pickle file

    Args:
        file_path (str): destination file
        coords (np.array): coordinates of grid points
        vals (np.array): measured values

    """
    with open(file_path, "wb+") as out_file:
        pk.dump((coords, vals), out_file)


def read_msr_grid(file_path):
    """Read coordinates and values of measurements over a grid from a pickle file

    Args:
        file_path (str): file to read

    Returns:
        coords (np.array): coordinates of grid points
        vals (np.array): measured values
    """
    with open(file_path, "rb") as f:
        coords, vals = pk.load(f)
    return coords, vals
