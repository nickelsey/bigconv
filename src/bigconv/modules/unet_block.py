from __future__ import annotations

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from bigconv.modules.conv3d import DistributedConv3d, DistributedConvTranspose3d
from bigconv.modules.norm import DistributedGroupNorm


class ConvNormAct3d(nn.Module):
    """Apply distributed convolution, distributed group norm, and activation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Scalar or per-axis convolution kernel size.
        stride: Scalar or per-axis convolution stride.
        padding: Scalar or per-axis convolution padding. Defaults to same
            padding for odd kernels.
        num_groups: Number of groups for distributed group normalization.
        padding_mode: Convolution padding mode.
        bias: Whether the convolution uses a bias parameter.
        eps: Numerical stability epsilon for group normalization.
        activation: Activation module. Defaults to ``nn.ReLU()``. Pass
            ``nn.Identity()`` to disable activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] | None = None,
        num_groups: int = 8,
        padding_mode: str = "zeros",
        bias: bool = True,
        eps: float = 1e-5,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv = DistributedConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            bias=bias,
        )
        self.norm = DistributedGroupNorm(num_groups, out_channels, eps=eps)
        self.activation = nn.ReLU() if activation is None else activation

    def reset_parameters(self) -> None:
        """Initialize convolution and normalization parameters."""
        self.conv.reset_parameters()
        self.norm.reset_parameters()

    def forward(self, x: torch.Tensor, mesh: DeviceMesh | None = None) -> torch.Tensor:
        """Run the block.

        Args:
            x: Channels-first tensor with shape ``(C_in, X_local, Y, Z)``.
            mesh: Optional 1D ``DeviceMesh`` along the sharded X axis.

        Returns:
            Channels-first tensor with shape ``(C_out, X_out, Y_out, Z_out)``.
        """
        x = self.conv(x, mesh=mesh)
        x = self.norm(x, mesh=mesh)
        return self.activation(x)


class UNetConvBlock3d(nn.Module):
    """Two-convolution U-Net block for channels-first 3D tensors.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        num_groups: Number of groups for both group normalization layers.
        kernel_size: Scalar or per-axis convolution kernel size.
        stride: Stride for the first convolution. Use ``2`` for stride-based
            downsampling.
        padding: Scalar or per-axis convolution padding. Defaults to same
            padding for odd kernels.
        padding_mode: Convolution padding mode.
        bias: Whether convolution layers use bias parameters.
        eps: Numerical stability epsilon for group normalization.
        activation: Activation module. A new ``nn.ReLU()`` is used for each
            layer when omitted.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_groups: int = 8,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] | None = None,
        padding_mode: str = "zeros",
        bias: bool = True,
        eps: float = 1e-5,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = ConvNormAct3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            num_groups=num_groups,
            padding_mode=padding_mode,
            bias=bias,
            eps=eps,
            activation=nn.ReLU() if activation is None else activation,
        )
        self.conv2 = ConvNormAct3d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            num_groups=num_groups,
            padding_mode=padding_mode,
            bias=bias,
            eps=eps,
            activation=nn.ReLU() if activation is None else activation,
        )

    def forward(self, x: torch.Tensor, mesh: DeviceMesh | None = None) -> torch.Tensor:
        """Run the two-convolution block."""
        x = self.conv1(x, mesh=mesh)
        return self.conv2(x, mesh=mesh)


class UNetUpBlock3d(nn.Module):
    """Transposed-convolution upsampling block with skip concatenation.

    Args:
        in_channels: Number of decoder input channels.
        skip_channels: Number of skip-connection channels.
        out_channels: Number of output channels.
        num_groups: Number of groups for group normalization in the conv block.
        up_kernel_size: Scalar or per-axis transposed-convolution kernel size.
        up_stride: Scalar or per-axis transposed-convolution stride.
        up_padding: Scalar or per-axis transposed-convolution padding. Defaults
            to same padding for odd kernels.
        up_output_padding: Scalar or per-axis output padding. Defaults to
            ``stride - 1`` per axis.
        kernel_size: Scalar or per-axis convolution kernel size for the
            post-concatenation block.
        padding: Scalar or per-axis convolution padding for the
            post-concatenation block. Defaults to same padding for odd kernels.
        padding_mode: Padding mode for the post-concatenation convolutions.
        bias: Whether convolution layers use bias parameters.
        eps: Numerical stability epsilon for group normalization.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        num_groups: int = 8,
        up_kernel_size: int | tuple[int, int, int] = 3,
        up_stride: int | tuple[int, int, int] = 2,
        up_padding: int | tuple[int, int, int] | None = None,
        up_output_padding: int | tuple[int, int, int] | None = None,
        kernel_size: int | tuple[int, int, int] = 3,
        padding: int | tuple[int, int, int] | None = None,
        padding_mode: str = "zeros",
        bias: bool = True,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.up = DistributedConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=up_kernel_size,
            stride=up_stride,
            padding=up_padding,
            output_padding=up_output_padding,
            bias=bias,
        )
        self.block = UNetConvBlock3d(
            out_channels + skip_channels,
            out_channels,
            num_groups=num_groups,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            padding_mode=padding_mode,
            bias=bias,
            eps=eps,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize transposed-convolution parameters and the conv block."""
        self.up.reset_parameters()
        self.block.conv1.reset_parameters()
        self.block.conv2.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        mesh: DeviceMesh | None = None,
    ) -> torch.Tensor:
        """Upsample ``x``, concatenate ``skip`` on channels, then run the conv block.

        Args:
            x: Decoder tensor with shape ``(C_in, X_local, Y, Z)``.
            skip: Encoder skip tensor with shape compatible with the upsampled
                output except for channels.
            mesh: Optional 1D ``DeviceMesh`` along the sharded X axis.

        Returns:
            Channels-first tensor with shape ``(C_out, X_out, Y_out, Z_out)``.

        Raises:
            ValueError: If upsampled and skip spatial shapes differ.
        """
        x = self.up(x, mesh=mesh)
        if x.shape[1:] != skip.shape[1:]:
            raise ValueError(
                f"upsampled spatial shape {tuple(x.shape[1:])} does not match "
                f"skip spatial shape {tuple(skip.shape[1:])}"
            )
        return self.block(torch.cat((x, skip), dim=0), mesh=mesh)
