# !/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project    : SLA-VAE
@File       : pid.py
@Author     : boyue.ht
@Version    : 1.0
@CreateTime : 2023/2/15
@LastModifiedTime: 2023/2/15
@Description: PID Control to ensure the stability of KL convergence.
                refer to: ControlVAE: Controllable Variational Autoencoder
                https://proceedings.mlr.press/v119/shao20b/shao20b.pdf
"""

import numpy as np
import torch
from einops import rearrange

def time_to_timefreq(x, n_fft: int, C: int, norm:bool=True):
    """
    x: (B, C, L)
    """
    x = rearrange(x, 'b c l -> (b c) l')
    x = torch.stft(x, n_fft, normalized=norm, return_complex=True, window=torch.hann_window(window_length=n_fft, device=x.device))
    x = torch.view_as_real(x)  # (B, N, T, 2); 2: (real, imag)
    x = rearrange(x, '(b c) n t z -> b (c z) n t ', c=C)  # z=2 (real, imag)
    return x  # (B, C, H, W)


def timefreq_to_time(x, n_fft: int, C: int, norm:bool=True):
    x = rearrange(x, 'b (c z) n t -> (b c) n t z', c=C).contiguous()
    x = x.contiguous()
    x = torch.view_as_complex(x)
    x = torch.istft(x, n_fft, normalized=norm, window=torch.hann_window(window_length=n_fft, device=x.device))
    x = rearrange(x, '(b c) l -> b c l', c=C)
    return x

class PIControl:
    """Feedback System, PI Control"""

    def __init__(self):
        self.i_p = 0.0
        self.beta = 0.0
        self.error = 0.0

    def pi(self, err, beta_min=1, n=1, kp=1e-2, ki=1e-4):
        beta_i = None
        for i in range(n):
            p_i = kp / (1.0 + np.exp(err))
            i_i = self.i_p - ki * err

            if self.beta < 1.0:
                i_i = self.i_p
            beta_i = p_i + i_i + beta_min

            self.i_p = i_i
            self.beta = beta_i
            self.error = err

            if beta_i < beta_min:
                beta_i = beta_min
        return beta_i
