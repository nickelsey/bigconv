from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from bigconv.ops import conv3d, conv_transpose3d
from bigconv.ops.conv3d import _normalize_tuple


def _default_padding(kernel_size: int | tuple[int, int, int]) -> tuple[int, int, int]:
    kernel_size_t = _normalize_tuple(kernel_size, "kernel_size")
    return (kernel_size_t[0] // 2, kernel_size_t[1] // 2, kernel_size_t[2] // 2)


class DistributedConv3d(nn.Module):
    """``nn.Module`` wrapper for :func:`bigconv.ops.conv3d`.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Scalar or per-axis convolution kernel size.
        stride: Scalar or per-axis convolution stride.
        padding: Scalar or per-axis convolution padding. Defaults to same
            padding for odd kernels.
        padding_mode: Convolution padding mode.
        bias: Whether to learn an additive bias.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] | None = None,
        padding_mode: str = "zeros",
        bias: bool = True,
    ) -> None:
        super().__init__()
        kernel_size_t = _normalize_tuple(kernel_size, "kernel_size")
        self.stride = _normalize_tuple(stride, "stride")
        self.padding = (
            _default_padding(kernel_size_t)
            if padding is None
            else _normalize_tuple(
                padding,
                "padding",
            )
        )
        self.padding_mode = padding_mode

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size_t))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights with PyTorch's convolution defaults."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = nn.init._calculate_correct_fan(self.weight, "fan_in")
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, mesh: DeviceMesh | None = None) -> torch.Tensor:
        """Run distributed convolution."""
        return conv3d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            padding_mode=self.padding_mode,
            mesh=mesh,
        )


class DistributedConvTranspose3d(nn.Module):
    """``nn.Module`` wrapper for :func:`bigconv.ops.conv_transpose3d`.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Scalar or per-axis transposed-convolution kernel size.
        stride: Scalar or per-axis transposed-convolution stride.
        padding: Scalar or per-axis transposed-convolution padding. Defaults to
            same padding for odd kernels.
        output_padding: Scalar or per-axis output padding. Defaults to
            ``stride - 1`` per axis.
        bias: Whether to learn an additive bias.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 2,
        padding: int | tuple[int, int, int] | None = None,
        output_padding: int | tuple[int, int, int] | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        kernel_size_t = _normalize_tuple(kernel_size, "kernel_size")
        self.stride = _normalize_tuple(stride, "stride")
        self.padding = (
            _default_padding(kernel_size_t)
            if padding is None
            else _normalize_tuple(
                padding,
                "padding",
            )
        )
        self.output_padding: tuple[int, int, int] = (
            (self.stride[0] - 1, self.stride[1] - 1, self.stride[2] - 1)
            if output_padding is None
            else _normalize_tuple(output_padding, "output_padding")
        )

        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, *kernel_size_t))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights with PyTorch's convolution defaults."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = nn.init._calculate_correct_fan(self.weight, "fan_in")
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, mesh: DeviceMesh | None = None) -> torch.Tensor:
        """Run distributed transposed convolution."""
        return conv_transpose3d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            mesh=mesh,
        )
