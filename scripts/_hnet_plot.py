import numpy as np 
import matplotlib.pyplot as plt

import cartopy
import cartopy.crs as ccrs


def plot_s2(data, fig=None, cmap="RdBu", projection = False, **kwargs):
    # Create a new figure if one is not provided
    if fig is None:
        fig = plt.figure()

    nlat = data.shape[-2]
    nlon = data.shape[-1]

    # Create default longitude and latitude arrays if not provided

    lon = np.linspace(0, 2 * np.pi, nlon)
    lat = np.linspace(np.pi / 2, -np.pi / 2, nlat)

    Lon, Lat = np.meshgrid(lon, lat)

    # Set up the desired projection

    if projection:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(central_longitude=0, central_latitude=0))
        transform = ccrs.PlateCarree()
    else:
        ax = fig.add_subplot(1, 1, 1)
        transform = None
    
    Lon = Lon * 180 / np.pi
    Lat = Lat * 180 / np.pi

    if transform:
        im = ax.imshow(data, extent=[-180, 180, -90, 90], cmap=cmap, transform=transform, **kwargs)
    else:
        im = ax.imshow(data, extent=[-180, 180, -90, 90], cmap=cmap, **kwargs)

    return fig, ax



def plot_shcoeffs(coeffs, fig=None, ticks_m=10, ticks_l=10, **kwargs):
    
    if fig is None:
        fig = plt.figure()
    
    Lmax = len(coeffs) - 1

    # Compute the log-magnitude, handling zeros.
    abs_coeffs = np.abs(coeffs)
    abs_coeffs[abs_coeffs == 0] = np.nan  # prevent log(0)
    log_coeffs = np.log(abs_coeffs)

    ax = fig.add_subplot(1, 1, 1)

    # Plot the data.
    im = ax.imshow(log_coeffs, origin='lower',
                   extent=[-Lmax - 0.5, Lmax + 0.5, -0.5, Lmax + 0.5],
                   aspect='equal', **kwargs)

    # Set tick marks.
    m_ticks = np.arange(-Lmax, Lmax + 1, ticks_m)
    ax.set_xticks(m_ticks)
    ax.set_xticklabels(m_ticks)

    l_ticks = np.arange(0, Lmax + 1, ticks_l)
    ax.set_yticks(l_ticks)
    ax.set_yticklabels(l_ticks)

    # Set axis labels.
    ax.set_xlabel("m", fontsize=16)
    ax.set_ylabel("ℓ", fontsize=16)

    # Place the m-axis on the top.
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    # Hide spines for a cleaner look.
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    return fig, ax



