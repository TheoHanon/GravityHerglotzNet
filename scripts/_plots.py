import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt


def plot_traj(xyz, val, fltr_idx=None, val_lbl="|a_grav| (m/s^2)", scatter_size=5):
    """Plot a trajectory in 3D with plotly

    Args:
        xyz (np.array): positions
        val (np.array): measurements
        fltr_idx (np.array): mask to apply to the positions
        val_lbl (str): colorbar label
        scatter_size (int): size of scatter markers

    Returns:
        fig (plotly Figure): interactive plotly figure. Can be displayed via fig.show() or save to html or png via the corresponding fig.write_html and fig.write_image methods
    """
    fig = go.Figure()
    if fltr_idx is None:
        fig.add_traces(
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="lines",
                line=dict(
                    color=np.linalg.norm(
                        val,
                        axis=1,
                    ),
                    colorscale="Magma",
                    colorbar={"title": val_lbl},
                    showscale=True,
                    width=5,
                ),
                opacity=1,
                showlegend=False,
            )
        )
        fig.add_traces(
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="markers",
                marker=dict(
                    size=scatter_size,
                    color=np.linalg.norm(
                        val,
                        axis=1,
                    ),
                    colorscale="Magma",
                    colorbar={"title": val_lbl},
                    showscale=False,
                ),
                opacity=1,
                showlegend=False,
            )
        )
    else:
        fig.add_traces(
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="lines",
                line=dict(
                    color="black",
                    showscale=False,
                    width=5,
                ),
                opacity=0.3,
                showlegend=False,
            )
        )
        fig.add_traces(
            go.Scatter3d(
                x=xyz[fltr_idx, 0],
                y=xyz[fltr_idx, 1],
                z=xyz[fltr_idx, 2],
                mode="markers",
                marker=dict(
                    size=scatter_size,
                    color=np.linalg.norm(
                        val,
                        axis=1,
                    ),
                    colorscale="Magma",
                    colorbar={"title": val_lbl},
                    opacity=1.0,
                    showscale=True,
                ),
                showlegend=False,
            )
        )
    fig.update_layout(
        scene=dict(
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
        )
    )
    return fig


def plot_val_2d(
    xy,
    val,
    fltr_idx=None,
    val_lbl="|a_grav| (m/s^2)",
    scatter_size=5,
    x_lbl=r"$\lambda$",
    y_lbl=r"$\theta$",
):
    """Plot a trajectory in 3D with plotly

    Args:
        xyz (np.array): positions (2 coordinates)
        val (np.array): measurements
        fltr_idx (np.array): mask to apply to the positions
        val_lbl (str): colorbar label
        scatter_size (int): size of scatter markers
        x_lbl (str): x axis label
        y_lbl (str): y axis label

    Returns:
        fig (plotly Figure): interactive plotly figure. Can be displayed via fig.show() or save to html or png via the corresponding fig.write_html and fig.write_image methods
    """
    fig = go.Figure()
    fig.add_traces(
        go.Scatter(
            x=xy[:, 0],
            y=xy[:, 1],
            mode="markers",
            opacity=1 if fltr_idx is None else 0.3,
            marker=dict(
                size=scatter_size,
                color=np.linalg.norm(
                    val,
                    axis=1,
                ),
                colorscale="Magma",
                colorbar={"title": val_lbl},
                showscale=True,
            ),
            showlegend=False,
        )
    )
    if fltr_idx is not None:
        fig.add_traces(
            go.Scatter(
                x=xy[fltr_idx, 0],
                y=xy[fltr_idx, 1],
                mode="markers",
                opacity=1.0,
                marker=dict(
                    size=scatter_size,
                    color=np.linalg.norm(
                        val,
                        axis=1,
                    ),
                    colorscale="Magma",
                    colorbar={"title": val_lbl},
                    showscale=False,
                ),
                showlegend=False,
            )
        )
    fig.update_layout(
        xaxis_title=x_lbl,
        yaxis_title=y_lbl,
    )
    return fig
