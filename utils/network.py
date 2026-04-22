import numpy as np
import torch
import torch.nn as nn


class GridEncoder(nn.Module):
    def __init__(self, in_channels=1, grid_res=100, hidden_sizes=(256, 128)):
        super().__init__()
        in_dim = in_channels * grid_res * grid_res

        layers = []
        last_dim = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last_dim, h), nn.Tanh()]
            last_dim = h

        self.model = nn.Sequential(*layers)
        self.output_dim = last_dim

    def forward(self, x):
        # x: [batch, channels, grid_res, grid_res]
        x = x.reshape(x.shape[0], -1)
        return self.model(x)


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

        self.xy_encoder = GridEncoder(
            in_channels=self.stack_num, grid_res=h, hidden_sizes=hidden_sizes
        )
        self.yz_encoder = GridEncoder(
            in_channels=self.stack_num, grid_res=h, hidden_sizes=hidden_sizes
        )
        self.xz_encoder = GridEncoder(
            in_channels=self.stack_num, grid_res=h, hidden_sizes=hidden_sizes
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
    
        # flattened input
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
    
        # single unstacked observation: (3, H, W)
        elif obs.ndim == 3:
            if len(self.state_shape) != 3:
                raise ValueError(f"Expected stacked obs, got shape {tuple(obs.shape)}")
            obs = obs.unsqueeze(0)   # -> (1, 3, H, W)
    
        # 4D case is ambiguous, so resolve it using self.state_shape
        elif obs.ndim == 4:
            if len(self.state_shape) == 3:
                # already batched unstacked: (B, 3, H, W)
                pass
            elif len(self.state_shape) == 4:
                # single stacked: (S, 3, H, W)
                obs = obs.unsqueeze(0)   # -> (1, S, 3, H, W)
            else:
                raise ValueError(f"Unexpected state_shape: {self.state_shape}")
    
        # batched stacked: (B, S, 3, H, W)
        elif obs.ndim == 5:
            if len(self.state_shape) != 4:
                raise ValueError(f"Unexpected stacked batch obs shape: {tuple(obs.shape)}")
    
        else:
            raise ValueError(f"Unexpected obs shape: {tuple(obs.shape)}")
    
        # split grids
        if obs.ndim == 4:
            # (B, 3, H, W)
            xy = obs[:, 0:1, :, :]
            yz = obs[:, 1:2, :, :]
            xz = obs[:, 2:3, :, :]
        elif obs.ndim == 5:
            # (B, S, 3, H, W)
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