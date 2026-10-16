"""
Continuous Flow Matching (CFM) Action Chunking Policy for Ultrasound Scanning.
Generates multi-step smooth continuous action trajectories conditioned on
PointNet latent reconstruction features and probe pose.
Based on Lipman et al. (ICLR 2023) and continuous robot flow matching paradigms.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowMatchingVectorFieldNet(nn.Module):
    """
    MLP vector field regressor v_theta(x_t, t | condition) predicting trajectory flow velocity.
    """
    def __init__(
        self,
        action_dim: int = 4,
        chunk_horizon: int = 8,
        condition_dim: int = 128 + 4, # 128 (PointNet) + 4 (probe pose)
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_horizon = chunk_horizon
        self.flat_action_dim = action_dim * chunk_horizon

        # Time embedding: sinusoidal embedding for flow time t in [0, 1]
        self.time_mlp = nn.Sequential(
            nn.Linear(32, 64),
            nn.Mish(),
            nn.Linear(64, 64)
        )

        # Condition projection
        self.cond_mlp = nn.Sequential(
            nn.Linear(condition_dim, 128),
            nn.Mish(),
            nn.Linear(128, 128)
        )

        # Vector field backbone
        in_dim = self.flat_action_dim + 64 + 128
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, self.flat_action_dim)
        )

    def _sinusoidal_embedding(self, t: torch.Tensor, dim: int = 32) -> torch.Tensor:
        """Sinusoidal positional encoding for continuous time t in [0, 1]."""
        half_dim = dim // 2
        freqs = torch.exp(-torch.arange(half_dim, device=t.device) * (torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_t: (B, horizon * action_dim) noisy action chunk.
            t: (B,) continuous integration time in [0, 1].
            condition: (B, condition_dim) combined context vector.

        Returns:
            v_t: (B, horizon * action_dim) predicted velocity field.
        """
        t_emb = self.time_mlp(self._sinusoidal_embedding(t))
        c_emb = self.cond_mlp(condition)

        feat = torch.cat([x_t, t_emb, c_emb], dim=-1)
        return self.backbone(feat)


class FlowMatchingPolicy(nn.Module):
    """
    Conditional Optimal Transport Flow Matching policy for robot action chunk generation.
    Generates action sequences via forward Euler ODE integration: dx/dt = v_theta(x, t | c).
    """
    def __init__(
        self,
        action_dim: int = 4,
        chunk_horizon: int = 8,
        condition_dim: int = 132,
        hidden_dim: int = 256,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_dim = action_dim
        self.chunk_horizon = chunk_horizon
        self.model = FlowMatchingVectorFieldNet(
            action_dim=action_dim,
            chunk_horizon=chunk_horizon,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim
        ).to(self.device)

    def to(self, *args, **kwargs):
        parsed = torch._C._nn._parse_to(*args, **kwargs)
        if parsed[0] is not None:
            self.device = parsed[0]
        return super().to(*args, **kwargs)

    def compute_loss(
        self,
        expert_actions: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes Conditional Flow Matching loss:
        L_CFM = E_{t, x0, x1} || v_theta(x_t, t | c) - (x1 - x0) ||^2

        Args:
            expert_actions: (B, horizon, action_dim) ground truth trajectory chunks.
            condition: (B, condition_dim) context features.
        """
        batch_size = expert_actions.shape[0]
        x1 = expert_actions.view(batch_size, -1).to(self.device) # Target action chunk

        # Sample prior noise x0 ~ N(0, I)
        x0 = torch.randn_like(x1)

        # Sample uniform time t in [0, 1]
        t = torch.rand(batch_size, device=self.device)

        # Optimal Transport straight interpolation path: x_t = (1 - t) * x0 + t * x1
        t_expand = t.unsqueeze(-1)
        x_t = (1.0 - t_expand) * x0 + t_expand * x1
        target_velocity = x1 - x0 # u_t = d(x_t)/dt

        # Predict vector field
        pred_velocity = self.model(x_t, t, condition.to(self.device))

        # Mean squared error loss
        loss = F.mse_loss(pred_velocity, target_velocity)
        return loss

    @torch.no_grad()
    def generate_action_chunk(
        self,
        condition: torch.Tensor,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """
        Integrates the learned velocity field from t=0 to t=1 using Euler method.

        Args:
            condition: (1, condition_dim) or (B, condition_dim) context features.
            num_steps: Number of ODE discretization steps.

        Returns:
            action_chunk: (B, horizon, action_dim) planned actions.
        """
        self.model.eval()
        cond = condition.to(self.device)
        if cond.ndim == 1:
            cond = cond.unsqueeze(0)
        batch_size = cond.shape[0]

        # Initial sample from standard Gaussian prior x(0) ~ N(0, I)
        flat_dim = self.chunk_horizon * self.action_dim
        x = torch.randn(batch_size, flat_dim, device=self.device)

        dt = 1.0 / float(num_steps)
        # Numerical Euler integration
        for step_idx in range(num_steps):
            t_val = step_idx * dt
            t_tensor = torch.full((batch_size,), t_val, device=self.device)
            v = self.model(x, t_tensor, cond)
            x = x + v * dt

        # Reshape to (B, chunk_horizon, action_dim) and clamp to valid [-1, 1]
        actions = torch.clamp(x.view(batch_size, self.chunk_horizon, self.action_dim), -1.0, 1.0)
        return actions
