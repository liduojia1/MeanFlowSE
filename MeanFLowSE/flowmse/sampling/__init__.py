# sampling/__init__.py
# Adapted from https://github.com/yang-song/score_sde_pytorch/.../sampling.py
from scipy import integrate
import torch

from .odesolvers import ODEsolver, ODEsolverRegistry

import numpy as np
import matplotlib.pyplot as plt


__all__ = [
    'ODEsolverRegistry', 'ODEsolver', 'get_white_box_solver'
]


def to_flattened_numpy(x):
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    return torch.from_numpy(x.reshape(shape))


def get_white_box_solver(
    odesolver_name,  ode, VF_fn, Y, Y_prior=None,
    T_rev=1.0, t_eps=0.03, N=30,  **kwargs
):
    odesolver_cls = ODEsolverRegistry.get_by_name(odesolver_name)
    odesolver = odesolver_cls(ode, VF_fn)

    def make_timesteps(device):
        return torch.linspace(T_rev, t_eps, N + 1, device=device)

    def ode_solver(Y_prior=Y_prior):
        with torch.no_grad():
            if Y_prior is None:
                Y_prior = Y

            xt, _ = ode.prior_sampling(Y_prior.shape, Y_prior)
            xt = xt.to(Y_prior.device)

            timesteps = make_timesteps(Y.device)

            for i in range(len(timesteps) - 1): 
                t = timesteps[i]
                stepsize = t - timesteps[i+1]   # Δ > 0
                vec_t = torch.ones(Y.shape[0], device=Y.device) * t
                xt = odesolver.update_fn(xt, vec_t, Y, stepsize)
            x_result = xt
            ns = len(timesteps) - 1  # = N
            return x_result, ns

    return ode_solver
