from __future__ import annotations

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from bigconv.ops import group_norm


class DistributedGroupNorm(nn.Module):
    """``nn.Module`` wrapper for :func:`bigconv.ops.group_norm`.

    Args:
        num_groups: Number of channel groups.
        num_channels: Number of channels in the input tensor.
        eps: Numerical stability epsilon.
        affine: Whether to learn per-channel affine parameters.
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        *,
        eps: float = 1e-5,
        affine: bool = True,
    ) -> None:
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError(
                f"num_channels={num_channels} must be divisible by num_groups={num_groups}"
            )
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def reset_parameters(self) -> None:
        """Reset affine parameters when present."""
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, mesh: DeviceMesh | None = None) -> torch.Tensor:
        """Run distributed group normalization."""
        return group_norm(
            x,
            self.num_groups,
            self.weight,
            self.bias,
            self.eps,
            mesh=mesh,
        )
