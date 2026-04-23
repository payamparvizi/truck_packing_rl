# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 13:02:08 2026

@author: payam
"""

import numpy as np
import torch
import torch.nn as nn


class GridCNNEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_size=256, grid_res=40):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),   # 40 -> 20
            nn.Tanh(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 20 -> 10
            nn.Tanh(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), # 10 -> 5
            nn.Tanh(),
        )

        conv_out_dim = 128 * (grid_res // 8) * (grid_res // 8)
        # for grid_res=40, this is 128 * 5 * 5 = 3200

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_dim, hidden_size),
            nn.Tanh(),
        )

        self.output_dim = hidden_size

    def forward(self, x):
        # x: [batch, channels, H, W]
        x = self.features(x)
        x = self.head(x)
        return x


class TripleGridNet(nn.Module):
    def __init__(
        self,
        state_shape,
        hidden_sizes=(256, 128),
        fusion_hidden_sizes=(256, 128),
        device="cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.state_shape = state_shape

        if len(state_shape) == 3:
            c, h, w = state_shape
            self.stack_num = 1
            assert c == 3, f"Expected 3 grids, got {c}"
        elif len(state_shape) == 4:
            s, c, h, w = state_shape
            self.stack_num = s
            assert c == 3, f"Expected 3 grids per frame, got {c}"
        else:
            raise ValueError(f"Unexpected state_shape: {state_shape}")

        encoder_hidden = hidden_sizes[0]

        self.xy_encoder = GridCNNEncoder(
            in_channels=self.stack_num, hidden_size=encoder_hidden, grid_res=h
        )
        self.yz_encoder = GridCNNEncoder(
            in_channels=self.stack_num, hidden_size=encoder_hidden, grid_res=h
        )
        self.xz_encoder = GridCNNEncoder(
            in_channels=self.stack_num, hidden_size=encoder_hidden, grid_res=h
        )

        fusion_in_dim = (
            self.xy_encoder.output_dim
            + self.yz_encoder.output_dim
            + self.xz_encoder.output_dim
        )

        layers = []
        last_dim = fusion_in_dim
        for hs in fusion_hidden_sizes:
            layers += [nn.Linear(last_dim, hs), nn.Tanh()]
            last_dim = hs

        self.fusion = nn.Sequential(*layers)
        self.output_dim = last_dim

        self.to(self.device)

    def forward(self, obs, state=None, info=None):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs = obs.to(self.device, dtype=torch.float32)

        if obs.ndim == 2:
            if len(self.state_shape) == 3:
                c, h, w = self.state_shape
                if obs.shape[1] == c * h * w:
                    obs = obs.view(obs.shape[0], c, h, w)
                else:
                    raise ValueError(
                        f"Expected flattened obs of size {c*h*w}, got shape {tuple(obs.shape)}"
                    )
            elif len(self.state_shape) == 4:
                s, c, h, w = self.state_shape
                if obs.shape[1] == s * c * h * w:
                    obs = obs.view(obs.shape[0], s, c, h, w)
                else:
                    raise ValueError(
                        f"Expected flattened obs of size {s*c*h*w}, got shape {tuple(obs.shape)}"
                    )
            else:
                raise ValueError(f"Unexpected state_shape: {self.state_shape}")

        elif obs.ndim == 3:
            if len(self.state_shape) != 3:
                raise ValueError(f"Expected stacked obs, got shape {tuple(obs.shape)}")
            obs = obs.unsqueeze(0)

        elif obs.ndim == 4:
            if len(self.state_shape) == 3:
                pass
            elif len(self.state_shape) == 4:
                obs = obs.unsqueeze(0)
            else:
                raise ValueError(f"Unexpected state_shape: {self.state_shape}")

        elif obs.ndim == 5:
            if len(self.state_shape) != 4:
                raise ValueError(f"Unexpected stacked batch obs shape: {tuple(obs.shape)}")
        else:
            raise ValueError(f"Unexpected obs shape: {tuple(obs.shape)}")

        if obs.ndim == 4:
            # (B, 3, H, W)
            xy = obs[:, 0:1, :, :]
            yz = obs[:, 1:2, :, :]
            xz = obs[:, 2:3, :, :]
        elif obs.ndim == 5:
            # (B, S, 3, H, W) -> (B, S, H, W)
            xy = obs[:, :, 0, :, :]
            yz = obs[:, :, 1, :, :]
            xz = obs[:, :, 2, :, :]
        else:
            raise ValueError(f"Unexpected obs shape after processing: {tuple(obs.shape)}")

        f_xy = self.xy_encoder(xy)
        f_yz = self.yz_encoder(yz)
        f_xz = self.xz_encoder(xz)

        fused = torch.cat([f_xy, f_yz, f_xz], dim=1)
        logits = self.fusion(fused)

        return logits, state