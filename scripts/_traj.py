import numpy as np

import spiceypy as sp
import matplotlib.pyplot as plt
import pyshtools as sh
from tqdm import tqdm

from ._units import *


def points_from_grid(n_max, r=1, use_GLQ_grid=True):
    """Return points for a spherical grid of resolution driven by n_max. Can be either a GLQ or a DH (uniformly-spaced) grid (see https://shtools.github.io/SHTOOLS/grid-formats.html)

    Args:
        n_max (int): maximum SH degree
        r (float): radius of the grid
        use_GLQ_grid (bool): whether to use a GLQ (True) or DH (False) grid


    Returns:
        state_mat (np.array): cartesian coords of grid points
        sph_coords (np.array): spherical coords of grid points
    """

    if use_GLQ_grid:  # Gauss-Legendre quadrature grid
        latVec, lonVec = sh.expand.GLQGridCoord(n_max, extend=False)
        latVec *= deg
        lonVec *= deg
        nlats = len(latVec)
        nlons = len(lonVec)
    else:  # Fall back to DH (regularly-spaced) grid
        nlats = 2 * n_max + 2
        nlons = 2 * nlats
        latVec = np.linspace(np.pi / 2, -np.pi / 2, nlats, endpoint=False)
        lonVec = np.linspace(0, 2 * np.pi, nlons, endpoint=False)
    latGrid, lonGrid = np.meshgrid(latVec, lonVec, indexing="ij")
    lon_lat = np.array([lonGrid.flatten(), latGrid.flatten()]).T.copy()

    # Store
    sph_coords = np.array(
        [
            np.ones(latGrid.size) * r,
            np.pi / 2 - latGrid.flatten(),
            lonGrid.flatten(),
        ]
    ).T
    state_mat = np.array([sp.sphrec(*el) for el in sph_coords])
    return state_mat, sph_coords


def old_points_from_keplerian_traj(
    gm, t_span=10 * day, coverage=1.0, dt=60.0 * sec, elts=None
):
    """Computes coordinates over a keplerian orbit for a given time span and sampling rate. The rate of variation of the RAAN is selected so as to reach the required coverage in the given time span.

    Args:
        gm (float): central body gravitational parameter
        t_span (float): length of propagation period
        coverage (float): how many times the orbital plane completes a full rotation around the body during t_span
        dt (float): rate of sampling of states along the orbit
        elts (dict): dictionary of keplerian elements for the orbit: a (semi-major axis), i (inclination), e (eccentricity), om (argument of periapsis), Om (right-ascension of the ascending node), M_0 (mean anomaly). Dafaults to MRO values

    Returns:
        state_mat (np.array): cartesian coords of grid points
        sph_coords (np.array): spherical coords of grid points
        t_vec (np.array): sampling times
    """
    ## Obital parameters
    if elts is None:
        elts = {}

    a = elts.get("a", 3630 * km)  # semi-major axis
    i = elts.get("i", 92.6 * deg)  # inclination
    e = elts.get("e", 0.00)  # eccentricity
    om = elts.get("om", 270 * deg)  # argument of perigee
    Om = elts.get("Om", 0.0 * deg)  # RAAN
    M_0 = elts.get("M_0", 0.0 * deg)  # mean anomaly

    n = np.sqrt(gm / a**3)  # mean motion

    ## SPICEY stuff
    METAKR = "spice/generic.tm"
    sp.furnsh(METAKR)

    # Keplerian elements for SPICE should be in km and rad
    kepel_0 = np.array(
        [
            a / km * (1 - e),
            e,
            i,
            Om,
            om,
            0.0,  # epoch 0
            M_0,
            gm / km**3,
        ]
    )

    # Desired "coverage", i.e. number of times it rotates around the body
    O_dot_from_coverage = 360.0 * deg / t_span

    t_vec = np.arange(0.0, t_span, dt)

    # Computing the state at each time-step from the keplerian elements and the RAAN rate
    state_mat = np.zeros((t_vec.shape[0], 3))
    sph_coords = np.zeros((t_vec.shape[0], 3))
    for t_idx, t in enumerate(t_vec):
        kepel = kepel_0.copy()
        kepel[3] += O_dot_from_coverage * t
        state = sp.conics(kepel, t)[:3]
        state_mat[t_idx, :] = state * km
        r, colat, slon = sp.recsph(state)
        sph_coords[t_idx, :] = [r * km, colat, slon]
    return state_mat, sph_coords, t_vec


def old_subsample_measurements(
    t_vec, passes_per_day=1, pass_length=8 * hour, random_track_start=True, rng=None
):
    """Construct a mask allowing to select measurement points from all trajectory points. Based on realistic tracking schedules

    Args:
        t_vec (np.array): sampling times of the trajectory
        passes_per_day (float): number of tracking passes each day
        pass_length (float): duration of each tracking pass
        random_track_start (bool): start tracking at random times each tracking day
        rng (np.random.Generator): random number generator

    Returns:
        fltr_idx (bool): mask for selection of the measurements from trajectory points
    """
    dt = np.min(np.abs(np.diff(t_vec)))

    # Length of tracking pass
    day_length = int(passes_per_day * day / dt)
    track_length = int(pass_length / dt)

    # Selecting each day only measurements within the tracking length
    fltr_idx = np.zeros(t_vec.size, dtype=bool)
    fltr_idx = fltr_idx.reshape((-1, day_length))

    if random_track_start:  # start at random times each day
        if rng is None:
            rng = np.random.default_rng()
        start_idx = rng.integers(0, day_length - track_length, size=fltr_idx.shape[0])
        for ii, idx in enumerate(start_idx):
            fltr_idx[ii, idx : idx + track_length] = True
    else:
        fltr_idx[:, :track_length] = True

    return fltr_idx.flatten()


def measurement_schedule(
    t_span,
    dt=60.0 * sec,
    passes_per_day=1,
    pass_length=8 * hour,
    random_track_start=True,
    rng=None,
):
    """Construct a mask allowing to select measurement points from all trajectory points. Based on realistic tracking schedules

    Args:
        t_vec (np.array): sampling times of the trajectory
        passes_per_day (float): number of tracking passes each day
        pass_length (float): duration of each tracking pass
        random_track_start (bool): start tracking at random times each tracking day
        rng (np.random.Generator): random number generator

    Returns:
        fltr_idx (bool): mask for selection of the measurements from trajectory points
    """

    # Length of tracking pass
    day_length = int(day / passes_per_day / dt)
    track_length = int(pass_length / dt)

    if day_length < track_length:
        day_length = track_length
    # Selecting each day only measurements within the tracking length
    fltr_idx = np.zeros(int(t_span / dt), dtype=bool)
    fltr_idx = fltr_idx.reshape((-1, day_length))

    if random_track_start:  # start at random times each day
        if rng is None:
            rng = np.random.default_rng()
        start_idx = rng.integers(0, day_length - track_length, size=fltr_idx.shape[0])
        for ii in tqdm(range(start_idx.shape[0])):
            idx = start_idx[ii]
            fltr_idx[ii, idx : idx + track_length] = True
    else:
        fltr_idx[:, :track_length] = True

    t_vec = np.nonzero(fltr_idx.flatten())[0] * dt
    return t_vec


def points_from_keplerian_traj(gm, t_vec, coverage=1.0, elts=None):
    """Computes coordinates over a keplerian orbit for a given a vector of times. The rate of variation of the RAAN is selected so as to reach the required coverage in the given time span.

    Args:
        gm (float): central body gravitational parameter
        t_vec (np.array): array of time stamps (seconds)
        coverage (float): how many times the orbital plane completes a full rotation around the body during t_span
        elts (dict): dictionary of keplerian elements for the orbit: a (semi-major axis), i (inclination), e (eccentricity), om (argument of periapsis), Om (right-ascension of the ascending node), M_0 (mean anomaly). Dafaults to MRO values

    Returns:
        state_mat (np.array): cartesian coords of grid points
        sph_coords (np.array): spherical coords of grid points
        t_vec (np.array): sampling times
    """
    ## Obital parameters
    if elts is None:
        elts = {}

    a = elts.get("a", 3630 * km)  # semi-major axis
    i = elts.get("i", 92.6 * deg)  # inclination
    e = elts.get("e", 0.00)  # eccentricity
    om = elts.get("om", 270 * deg)  # argument of perigee
    Om = elts.get("Om", 0.0 * deg)  # RAAN
    M_0 = elts.get("M_0", 0.0 * deg)  # mean anomaly

    n = np.sqrt(gm / a**3)  # mean motion

    ## SPICEY stuff
    METAKR = "spice/generic.tm"
    sp.furnsh(METAKR)

    # Keplerian elements for SPICE should be in km and rad
    kepel_0 = np.array(
        [
            a / km * (1 - e),
            e,
            i,
            Om,
            om,
            0.0,  # epoch 0
            M_0,
            gm / km**3,
        ]
    )

    # Desired "coverage", i.e. number of times it rotates around the body
    O_dot_from_coverage = 360.0 * deg / t_vec[-1]

    # Computing the state at each time-step from the keplerian elements and the RAAN rate
    state_mat = np.zeros((t_vec.shape[0], 3))
    sph_coords = np.zeros((t_vec.shape[0], 3))
    for t_idx, t in enumerate(t_vec):
        kepel = kepel_0.copy()
        kepel[3] += O_dot_from_coverage * t
        state = sp.conics(kepel, t)[:3]
        state_mat[t_idx, :] = state * km
        r, colat, slon = sp.recsph(state)
        sph_coords[t_idx, :] = [r * km, colat, slon]
    return state_mat, sph_coords, t_vec
