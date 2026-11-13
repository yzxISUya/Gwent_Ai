"""
Neural network architectures for Gwent RL (Masked PPO).
Supports unified ONNX export wrapper with signature:
  Inputs:
    - obs: float32[batch, 689]
    - mask: bool[batch, 91]
  Outputs:
    - probs: float32[batch, 91]   (masked action probability distribution)
    - value: float32[batch, 1]    (estimated win value in [-1, 1], convertible to win rate)
"""
from __future__ import annotations
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from gwent_rl.engine import OBS_DIM, ACTION_DIM


class ResMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.gelu(self.ln1(self.fc1(x)))
        out = self.dropout(out)
        out = self.ln2(self.fc2(out))
        return F.gelu(residual + out)


class GwentPolicyValueNet(nn.Module):
    """
    Residual MLP Actor-Critic network for Masked PPO.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = 512,
        num_blocks: int = 3,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([ResMLPBlock(hidden_dim) for _ in range(num_blocks)])

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(obs)
        for block in self.blocks:
            h = block(h)
        return h

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning (probs, value) suitable for direct ONNX export.
        """
        h = self.encode(obs)
        logits = self.policy_head(h)
        bool_mask = mask.bool()
        mask_val = -1e4 if obs.dtype == torch.float16 else -1e8
        masked_logits = logits.masked_fill(~bool_mask, mask_val)
        probs = F.softmax(masked_logits, dim=-1)
        value = self.value_head(h)
        return probs, value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encode(obs)
        logits = self.policy_head(h)
        bool_mask = mask.bool()
        mask_val = -1e4 if obs.dtype == torch.float16 else -1e8
        masked_logits = logits.masked_fill(~bool_mask, mask_val)
        dist = Categorical(logits=masked_logits)
        if action is None:
            if deterministic:
                action = torch.argmax(masked_logits, dim=-1)
            else:
                action = dist.sample()
        log_prob = dist.log_prob(action)
        # Numerically stable masked entropy: only sum entropy over legal actions
        probs = F.softmax(masked_logits, dim=-1)
        log_probs = torch.log(probs + 1e-12)
        entropy = -torch.sum(torch.where(bool_mask, probs * log_probs, torch.zeros_like(probs)), dim=-1)
        value = self.value_head(h).squeeze(-1)
        return action, log_prob, entropy, value
