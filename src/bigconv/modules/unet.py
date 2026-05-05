from __future__ import annotations

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from bigconv.modules.conv3d import DistributedConv3d
from bigconv.modules.unet_block import UNetConvBlock3d, UNetUpBlock3d


class UNet3d(nn.Module):
    """Classic encoder-decoder 3D U-Net for channels-first tensors.

    The model uses stride-2 convolutions for encoder downsampling and
    transposed convolutions for decoder upsampling. Inputs and outputs use the
    project layout ``(C, X_local, Y, Z)``. Pass ``mesh`` to ``forward`` for
    distributed X-sharded execution.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        channels: Channel schedule from first encoder stage through bottleneck.
            Must contain at least two entries.
        num_groups: Number of groups for group normalization in all blocks.
        kernel_size: Scalar or per-axis convolution kernel size for conv blocks.
        padding: Scalar or per-axis convolution padding. Defaults to same
            padding for odd kernels.
        padding_mode: Padding mode for regular convolutions.
        bias: Whether convolution layers use bias parameters.
        eps: Numerical stability epsilon for group normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        *,
        num_groups: int = 8,
        kernel_size: int | tuple[int, int, int] = 3,
        padding: int | tuple[int, int, int] | None = None,
        padding_mode: str = "zeros",
        bias: bool = True,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if len(channels) < 2:
            raise ValueError(f"channels must contain at least two entries, got {channels!r}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels

        self.input_block = UNetConvBlock3d(
            in_channels,
            channels[0],
            num_groups=num_groups,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            padding_mode=padding_mode,
            bias=bias,
            eps=eps,
        )

        self.down_blocks = nn.ModuleList(
            UNetConvBlock3d(
                channels[i],
                channels[i + 1],
                num_groups=num_groups,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                padding_mode=padding_mode,
                bias=bias,
                eps=eps,
            )
            for i in range(len(channels) - 1)
        )

        self.up_blocks = nn.ModuleList(
            UNetUpBlock3d(
                channels[i + 1],
                skip_channels=channels[i],
                out_channels=channels[i],
                num_groups=num_groups,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode=padding_mode,
                bias=bias,
                eps=eps,
            )
            for i in range(len(channels) - 2, -1, -1)
        )

        self.output = DistributedConv3d(
            channels[0],
            out_channels,
            kernel_size=1,
            padding=0,
            bias=True,
        )

    def forward(self, x: torch.Tensor, mesh: DeviceMesh | None = None) -> torch.Tensor:
        """Run the U-Net.

        Args:
            x: Channels-first tensor with shape ``(C_in, X_local, Y, Z)``.
            mesh: Optional 1D ``DeviceMesh`` along the sharded X axis.

        Returns:
            Channels-first tensor with shape ``(C_out, X_local, Y, Z)``.
        """
        skips = [self.input_block(x, mesh=mesh)]
        x = skips[-1]

        for block in self.down_blocks:
            x = block(x, mesh=mesh)
            skips.append(x)

        x = skips.pop()
        for block in self.up_blocks:
            skip = skips.pop()
            x = block(x, skip, mesh=mesh)

        return self.output(x, mesh=mesh)
