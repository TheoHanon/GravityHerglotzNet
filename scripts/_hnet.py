import os 

import torch
import torch.nn as nn
import numpy as np
import math

import spherical_inr as sph
import spherical_inr.differentiation as D

from typing import List, Tuple


class SReLU(nn.Module):
    def forward(self, x):
        return torch.where(x >= 0, x, torch.sin(x))
        
class NormalizedIrregularHerglotzPEv2(sph.NormalizedIrregularHerglotzPE):

    def __init__(self, normalize :bool=  True, **kwargs):
        super(NormalizedIrregularHerglotzPEv2, self).__init__(**kwargs)
        self.rref.data = torch.tensor(1.0)
        self.rref.requires_grad = False
        self.norm_const = torch.tensor(0.0) if not normalize else torch.tensor(1.0)

    def forward(self, x):
        A_rotated_real = self.quaternion_rotation(self.A_real)  
        A_rotated_imag = self.quaternion_rotation(self.A_imag)
        A_rotated = torch.complex(A_rotated_real, A_rotated_imag)
        
        x = x.to(A_rotated.dtype)
        r = torch.norm(x, dim=-1, keepdim=True, p = 2)
        ax = torch.matmul(x, A_rotated.t())
        
        ax_R = (ax.real / r) * (self.rref/r) 
        ax_I = (ax.imag / r) * (self.rref/r) 

        sin_term = torch.sin(self.w_R * (ax_I - ax_R))
        exp_term = torch.exp(self.w_R * ( (ax_R + ax_I - self.norm_const)))
        
        return 1/r * (sin_term * exp_term)
    
class ToCart(nn.Module):

    def forward(self, x):
        return sph.transforms.rtp_to_r3(x)


class GravityHNet(nn.Module):
    def __init__(self, 
            inr_sizes : List[int],
            activation : str = "relu", 
            init :bool = False,
            normalize :bool = True,
            ):
        
        super(GravityHNet, self).__init__()
        self.pe = NormalizedIrregularHerglotzPEv2(
            L = inr_sizes[0],
            init =init,
            normalize = normalize,
        )
        # self.pe.b_I.requires_grad = False
        self.pe.b_R.requires_grad = False
        
        if activation == "sin":
            self.mlp = sph.SineMLP(
                input_features=self.pe.num_atoms,
                hidden_sizes=inr_sizes[1:],
                output_features=1, 
                bias = False,
                omega0=1.0,
            )
        elif activation == "srelu":
            self.mlp = sph.MLP(
                input_features=self.pe.num_atoms,
                hidden_sizes=inr_sizes[1:],
                output_features=1, 
                bias = False,
                activation = "relu", 
            )
            self.mlp.activation = SReLU()
        else:
            self.mlp = sph.MLP(
                input_features=self.pe.num_atoms,
                hidden_sizes=inr_sizes[1:],
                output_features=1, 
                bias = False,
                activation = activation, 
            )

        self.to_cart = ToCart()
        
    
    def forward(self, x):
        r = x[..., 0].unsqueeze(-1)

        x = self.to_cart(x)
        x = self.pe(x)
        x = self.mlp(x)

        return  1/r + x
    
    def forward_pe(self, x):
        r = x[..., 0].unsqueeze(-1)
        x = self.to_cart(x)
        x = self.pe(x)
        return x


def sph_field_to_cart_field(f, sph_coords):
    """
    Convert a spherical field to a cartesian field.
    """

    f_r, f_theta, f_phi = f.unbind(dim = -1)
    r, theta, phi = sph_coords.unbind(dim = -1)

    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)

    f_x = (f_r * sin_theta * cos_phi +
           f_theta * cos_theta * cos_phi -
           f_phi * sin_phi)

    f_y = (f_r * sin_theta * sin_phi +
           f_theta * cos_theta * sin_phi +
           f_phi * cos_phi)

    f_z = (f_r * cos_theta -
           f_theta * sin_theta)

    return torch.stack([f_x, f_y, f_z], dim = -1)
    

def flux_regulizer(yreg1, yreg2, xreg1, xreg2, rad1, rad2, *args):
    """
    Imposes that the outward flux of 2 spheres of different radii is the same.
    """
    
    dr1 = D.spherical_gradient(yreg1, xreg1, track=True)[..., 0]
    dr2 = D.spherical_gradient(yreg2, xreg2, track=True)[..., 0]

    flux1 = rad1**2 * dr1.mean()
    flux2 = rad2**2 * dr2.mean()
    
    return (flux1 - flux2).pow(2)


def laplacian_regulizer(yreg, xreg, *args):
    """
    Imposes that the Laplacian of the output is zero.
    """

    lap = D.spherical_laplacian(yreg, xreg, track=True)    
    return lap.pow(2).mean()


def energy_regulizer(yreg, xreg, *args):
    """
    Minimizes the gradient energy.
    """
    grad = D.spherical_gradient(yreg, xreg, track=True)
    energy = torch.sum(grad**2, dim=-1).mean()
    return energy


def gradient_loss(target, output, input):
    """
    MSE loss between the gradient of the output and the acceleration field. 
    """

    grad = D.spherical_gradient(output, input, track=True)
    loss = torch.nn.functional.mse_loss(grad, target)

    return loss

def gradient_metric_loss(target, output, input):
    """
    MSE loss between the gradient of the output and the acceleration field. 
    """

    grad = D.spherical_gradient(output, input, track=True)

    grad_cart = sph_field_to_cart_field(grad, input)
    target_cart = sph_field_to_cart_field(target, input)

    loss = torch.nn.functional.mse_loss(grad_cart, target_cart)

    return loss


def s2_grid(R : float, L: int, sampling: str = "gl"):
    """
    Samples points on the 2-sphere of radius R for a given resolution L and sampling type.

    Parameters:
    L (int): Bandwidth of the spherical harmonics.
    sampling (str): Sampling scheme, default is 'gl' (Gauss-Legendre).

    Returns:
    tuple: A tuple containing:
        - phi (numpy.ndarray): Longitudinal angles (azimuth).
        - theta (numpy.ndarray): Latitudinal angles (colatitude).
        - (nlon, nlat) (tuple): Number of longitude and latitude points.
    """

    from s2fft.sampling.s2_samples import phis_equiang, thetas

    phi = phis_equiang(L, sampling=sampling)
    theta = thetas(L, sampling=sampling)
    nlon, nlat = phi.shape[0], theta.shape[0]

    phi, theta = np.meshgrid(phi, theta)
    phi, theta = torch.tensor(phi), torch.tensor(theta)
    radii = R * torch.ones_like(phi)

    return radii, phi, theta, (nlon, nlat)

def s2_sample(R : float, n_sample: int):
    """
    Sample 'n_sample' points uniformly on the 2-sphere of radius R.
    """

    radii = R * np.ones(n_sample)
    
    cos_theta = np.random.uniform(-1, 1, n_sample)
    theta = np.arccos(cos_theta)

    phi = np.random.uniform(0, 2*np.pi, n_sample)

    radii, theta, phi = torch.tensor(radii, dtype=torch.float32), torch.tensor(theta, dtype=torch.float32), torch.tensor(phi, dtype=torch.float32)
    xreg = torch.stack([radii, theta, phi], dim = -1).requires_grad_(True)

    return xreg


def two_sphere_sample(n_sample : int, Rref : float=10.0):
    """ 
    Sample points uniformly on two spheres of different radii.
    """

    r1, r2 = Rref * np.random.uniform(0.1, 1, size = (2, 1))
    xreg1 = s2_sample(r1, n_sample)
    xreg2 = s2_sample(r2, n_sample)

    r1, r2 = torch.tensor(r1, dtype=torch.float32), torch.tensor(r2, dtype=torch.float32)
    
    out_dict = {}
    out_dict["xregs"] = [xreg1, xreg2]
    out_dict["params"] = [r1, r2]
    
    return out_dict

def r3_sample(n_sample: int, Rref : float = 10.0):

    radii = Rref * np.random.uniform(.1, 1, n_sample)**(1/3)
    
    cos_theta = np.random.uniform(-1, 1, n_sample)
    theta = np.arccos(cos_theta)

    phi = np.random.uniform(0, 2*np.pi, n_sample)

    radii, theta, phi = torch.tensor(radii, dtype=torch.float32), torch.tensor(theta, dtype=torch.float32), torch.tensor(phi, dtype=torch.float32)
    xreg = torch.stack([radii, theta, phi], dim = -1).requires_grad_(True)

    out_dict = {}
    out_dict["xregs"] = [xreg]
    out_dict["params"] = []

    return out_dict


def train(
    model: nn.Module,
    optimizer,
    epoch: int,
    batch_size: int,
    N_reg: int,
    data_train: Tuple[torch.Tensor, torch.Tensor],
    data_val: Tuple[torch.Tensor, torch.Tensor],
    loss_fn : callable = gradient_loss,
    reg_fn: callable = None,
    sample_fn: callable = None,
    beta_reg : float = 1e-3,
    save_path: str = None
) -> None:
    """
    Train the model with optional regularization.

    Parameters:
        model (nn.Module): The model to train.
        optimizer: The optimizer instance.
        epoch (int): Number of epochs.
        batch_size (int): Batch size.
        N_reg (int): Number of regularization samples.
        data_train (Tuple[torch.Tensor, torch.Tensor]): Training dataset.
        data_val (Tuple[torch.Tensor, torch.Tensor]): Validation dataset.
        reg_fn (callable, optional): Regularization function.
        sample_fn (callable, optional): Sampling function for regularization.

    Raises:
        ValueError: If a regularization function is provided without a sampling function.
    """
    if reg_fn is not None and sample_fn is None:
        raise ValueError("Need to provide a sampling function for the regularization term.")

    xtrain, ytrain = data_train
    xval, yval = data_val

    dataset = torch.utils.data.TensorDataset(xtrain, ytrain)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    mse = nn.MSELoss()
    best_loss = float("inf")
    best_val_loss = float("inf")
    model.train()
    for i in range(epoch):
        epoch_loss = 0.0
        for xb, yb in dataloader:
            optimizer.zero_grad()

            if reg_fn is not None:

                reg_data = sample_fn(N_reg)
                xregs = reg_data["xregs"]
                yregs = [model(xreg) for xreg in xregs]
                params = reg_data["params"]
                Reg = reg_fn(*yregs, *xregs, *params)

            else : 
                Reg = 0.0

            ypred = model(xb)
            Loss = loss_fn(yb, ypred, xb) + beta_reg * Reg

            Loss.backward()
            optimizer.step()

            epoch_loss += Loss.item()

        epoch_loss /= len(dataloader)

        model.eval()
        with torch.no_grad():
            yval_pred = model(xval)
            val_loss = mse(yval, yval_pred).item()

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_val_loss = val_loss
            if save_path is not None:
                torch.save(model.state_dict(), os.path.join(save_path, f"best_model.pth"))

        model.train()

        print(f"Epoch {i+1}/{epoch} | Loss: {epoch_loss**(0.5):.8g} | Val Loss: {val_loss**(0.5):.8g}", end = "\r")

    print(
        "",
        "Training Done.",
        "-"*50,
        f"Best Loss: {best_loss**(0.5)}",
        f"Final Loss: {epoch_loss**(0.5)}",
        f"Best Val Loss: {best_val_loss**(0.5)}",
        sep="\n",
    )

    