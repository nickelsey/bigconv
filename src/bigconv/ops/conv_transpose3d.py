from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh

from bigconv.ops._dist_utils import group_name_from_mesh, resolve_group
from bigconv.ops.conv3d import _MAX_KERNEL, _SUPPORTED_STRIDES, _normalize_tuple


def _validate_conv_transpose_args(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
    output_padding: tuple[int, int, int],
    mesh: DeviceMesh | None,
) -> None:
    """Validate distributed transposed convolution arguments.

    Args:
        x: Local channels-first input tensor with shape ``(C_in, X_local, Y, Z)``.
        weight: Transposed-convolution weight with shape
            ``(C_in, C_out, K_x, K_y, K_z)``.
        bias: Optional bias tensor with shape ``(C_out,)``.
        stride: Per-axis transposed-convolution stride.
        padding: Per-axis transposed-convolution padding.
        output_padding: Per-axis output padding.
        mesh: Optional 1D device mesh along the sharded X axis.

    Raises:
        ValueError: If tensor shapes, scalar constraints, mesh rank, or shard
            sizes are unsupported.
    """
    if x.ndim != 4:
        raise ValueError(
            f"x must be 4D (C_in, X_local, Y, Z), got {x.ndim}D with shape {tuple(x.shape)}"
        )
    C_in, X_local, _Y, _Z = x.shape

    if weight.ndim != 5:
        raise ValueError(
            f"weight must be 5D (C_in, C_out, K_x, K_y, K_z), got {weight.ndim}D "
            f"with shape {tuple(weight.shape)}"
        )
    C_in_w, C_out, K_x, K_y, K_z = weight.shape
    if C_in != C_in_w:
        raise ValueError(f"x C_in={C_in} doesn't match weight C_in={C_in_w}")

    if bias is not None and (bias.ndim != 1 or bias.shape[0] != C_out):
        raise ValueError(f"bias must have shape ({C_out},), got {tuple(bias.shape)}")

    for name, k in (("K_x", K_x), ("K_y", K_y), ("K_z", K_z)):
        if k % 2 == 0:
            raise ValueError(f"{name}={k} must be odd (even kernels are not supported)")
        if k < 1 or k > _MAX_KERNEL:
            raise ValueError(f"{name}={k} must be in [1, {_MAX_KERNEL}]")

    for axis_name, s in zip(("stride_x", "stride_y", "stride_z"), stride):
        if s not in _SUPPORTED_STRIDES:
            raise ValueError(
                f"{axis_name}={s} not supported; stride must be in {_SUPPORTED_STRIDES}"
            )

    for axis_name, p, k in zip(
        ("padding_x", "padding_y", "padding_z"),
        padding,
        (K_x, K_y, K_z),
    ):
        if p < 0:
            raise ValueError(f"{axis_name}={p} must be non-negative")
        half = (k - 1) // 2
        if p > half:
            raise ValueError(f"{axis_name}={p} exceeds (K-1)/2={half}")

    for axis_name, op, s in zip(
        ("output_padding_x", "output_padding_y", "output_padding_z"),
        output_padding,
        stride,
    ):
        if op < 0:
            raise ValueError(f"{axis_name}={op} must be non-negative")
        if op >= s:
            raise ValueError(f"{axis_name}={op} must be smaller than stride={s}")

    P_x_required = (K_x - 1) // 2
    if padding[0] != P_x_required:
        raise ValueError(
            f"padding_x={padding[0]} must equal (K_x - 1) // 2 = {P_x_required} "
            f"for uniform X output sharding"
        )

    output_padding_x_required = stride[0] - 1
    if output_padding[0] != output_padding_x_required:
        raise ValueError(
            f"output_padding_x={output_padding[0]} must equal stride_x - 1 = "
            f"{output_padding_x_required} for uniform X output sharding"
        )

    if mesh is not None and mesh.ndim != 1:
        raise ValueError(
            f"conv_transpose3d expects a 1D DeviceMesh along X; got {mesh.ndim}D mesh "
            f"with shape {tuple(mesh.mesh.shape)}. Pass a sliced sub-mesh like mesh['x']."
        )

    if mesh is not None and dist.get_world_size(mesh.get_group()) > 1:
        pg = mesh.get_group()
        extrema = torch.tensor([X_local, -X_local], dtype=torch.int64, device=x.device)
        dist.all_reduce(extrema, op=dist.ReduceOp.MAX, group=pg)
        max_x = int(extrema[0].item())
        min_x = -int(extrema[1].item())
        if max_x != min_x:
            raise ValueError(
                f"X_local must be uniform across ranks; got min={min_x} max={max_x} "
                f"(this rank has X_local={X_local})"
            )


def _reference_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    stride: int | tuple[int, int, int] = 1,
    padding: int | tuple[int, int, int] = 0,
    output_padding: int | tuple[int, int, int] = 0,
) -> torch.Tensor:
    """Compute the single-process reference transposed convolution used by tests."""
    stride_t = _normalize_tuple(stride, "stride")
    padding_t = _normalize_tuple(padding, "padding")
    output_padding_t = _normalize_tuple(output_padding, "output_padding")
    _validate_conv_transpose_args(
        x,
        weight,
        bias,
        stride_t,
        padding_t,
        output_padding_t,
        mesh=None,
    )
    out = F.conv_transpose3d(
        x.unsqueeze(0),
        weight,
        bias,
        stride=stride_t,
        padding=padding_t,
        output_padding=output_padding_t,
    )
    return out.squeeze(0)


def _rank_info(pg: ProcessGroup | None) -> tuple[int, int]:
    if dist.is_initialized():
        return dist.get_rank(pg), dist.get_world_size(pg)
    return 0, 1


def _exchange_output_contribs_pg(
    full_out: torch.Tensor,
    own_x: int,
    pad_x: int,
    pg: ProcessGroup | None,
) -> torch.Tensor:
    """Crop local transposed-conv output and add neighbor boundary contributions."""
    if pad_x == 0:
        return full_out[:, :own_x].contiguous()

    rank, world_size = _rank_info(pg)
    has_left = rank > 0
    has_right = rank < world_size - 1

    own = full_out[:, pad_x : pad_x + own_x].contiguous()
    left_send = full_out[:, :pad_x].contiguous()
    right_send = full_out[:, pad_x + own_x :].contiguous()

    send_bufs: list[torch.Tensor] = []
    left_recv: torch.Tensor | None = None
    right_recv: torch.Tensor | None = None
    reqs: list[Any] = []

    if has_left:
        send_bufs.append(left_send)
        reqs.append(dist.isend(send_bufs[-1], rank - 1, group=pg))
        left_recv = torch.empty_like(left_send)
        reqs.append(dist.irecv(left_recv, rank - 1, group=pg))
    if has_right:
        send_bufs.append(right_send)
        reqs.append(dist.isend(send_bufs[-1], rank + 1, group=pg))
        right_recv = torch.empty_like(right_send)
        reqs.append(dist.irecv(right_recv, rank + 1, group=pg))

    for req in reqs:
        req.wait()

    if left_recv is not None:
        own[:, :pad_x] += left_recv
    if right_recv is not None:
        own[:, -pad_x:] += right_recv

    return own


def _exchange_output_grads_pg(
    grad_out: torch.Tensor,
    full_x: int,
    pad_x: int,
    pg: ProcessGroup | None,
) -> torch.Tensor:
    """Reverse ``_exchange_output_contribs_pg`` for transposed-conv backward."""
    if pad_x == 0:
        return grad_out[:, :full_x].contiguous()

    rank, world_size = _rank_info(pg)
    has_left = rank > 0
    has_right = rank < world_size - 1

    left_send = grad_out[:, :pad_x].contiguous()
    right_send = grad_out[:, -pad_x:].contiguous()

    send_bufs: list[torch.Tensor] = []
    left_recv: torch.Tensor | None = None
    right_recv: torch.Tensor | None = None
    reqs: list[Any] = []

    if has_left:
        send_bufs.append(left_send)
        reqs.append(dist.isend(send_bufs[-1], rank - 1, group=pg))
        left_recv = torch.empty_like(left_send)
        reqs.append(dist.irecv(left_recv, rank - 1, group=pg))
    if has_right:
        send_bufs.append(right_send)
        reqs.append(dist.isend(send_bufs[-1], rank + 1, group=pg))
        right_recv = torch.empty_like(right_send)
        reqs.append(dist.irecv(right_recv, rank + 1, group=pg))

    for req in reqs:
        req.wait()

    grad_full = grad_out.new_zeros(
        (grad_out.shape[0], full_x, grad_out.shape[2], grad_out.shape[3])
    )
    own_x = grad_out.shape[1]
    grad_full[:, pad_x : pad_x + own_x].copy_(grad_out)
    if left_recv is not None:
        grad_full[:, :pad_x].copy_(left_recv)
    if right_recv is not None:
        grad_full[:, pad_x + own_x :].copy_(right_recv)
    return grad_full


@torch.library.custom_op("bigconv::conv_transpose3d", mutates_args=())
def _conv_transpose3d_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: list[int],
    padding: list[int],
    output_padding: list[int],
    group_name: str,
) -> torch.Tensor:
    pg = resolve_group(group_name)
    stride_t = (stride[0], stride[1], stride[2])
    yz_padding_t = (0, padding[1], padding[2])
    output_padding_t = (output_padding[0], output_padding[1], output_padding[2])

    full_out = F.conv_transpose3d(
        x.unsqueeze(0),
        weight,
        None,
        stride=stride_t,
        padding=yz_padding_t,
        output_padding=output_padding_t,
    ).squeeze(0)
    own_x = x.shape[1] * stride[0]
    out = _exchange_output_contribs_pg(full_out, own_x, padding[0], pg)
    if bias is not None:
        out = out + bias.view(bias.shape[0], 1, 1, 1)
    return out


def _fake_conv_transpose3d_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: list[int],
    padding: list[int],
    output_padding: list[int],
    group_name: str,
):
    C_out = weight.shape[1]
    K_y, K_z = weight.shape[3], weight.shape[4]
    X_out = x.shape[1] * stride[0]
    Y_out = (x.shape[2] - 1) * stride[1] - 2 * padding[1] + K_y + output_padding[1]
    Z_out = (x.shape[3] - 1) * stride[2] - 2 * padding[2] + K_z + output_padding[2]
    return x.new_empty((C_out, X_out, Y_out, Z_out))


def _setup_context(ctx, inputs, output):
    x, weight, bias, stride, padding, output_padding, group_name = inputs
    ctx.save_for_backward(x, weight)
    ctx.has_bias = bias is not None
    ctx.stride = tuple(stride)
    ctx.padding = tuple(padding)
    ctx.output_padding = tuple(output_padding)
    ctx.group_name = group_name


def _conv_transpose3d_backward(ctx, grad_out):
    x, weight = ctx.saved_tensors
    stride = ctx.stride
    padding = ctx.padding
    output_padding = ctx.output_padding
    pg = resolve_group(ctx.group_name)

    full_x = (x.shape[1] - 1) * stride[0] + weight.shape[2] + output_padding[0]
    grad_full = _exchange_output_grads_pg(grad_out, full_x, padding[0], pg)

    grad_input, grad_weight, _grad_bias = torch.ops.aten.convolution_backward(
        grad_full.unsqueeze(0).contiguous(),
        x.unsqueeze(0),
        weight,
        None,
        list(stride),
        [0, padding[1], padding[2]],
        [1, 1, 1],
        True,
        list(output_padding),
        1,
        [True, True, False],
    )
    grad_bias = grad_out.sum(dim=(1, 2, 3)) if ctx.has_bias else None

    return (
        grad_input.squeeze(0).contiguous(),
        grad_weight,
        grad_bias,
        None,
        None,
        None,
        None,
    )


_conv_transpose3d_op.register_fake(_fake_conv_transpose3d_op)
_conv_transpose3d_op.register_autograd(
    _conv_transpose3d_backward,
    setup_context=_setup_context,
)


def conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    stride: int | tuple[int, int, int] = 1,
    padding: int | tuple[int, int, int] = 0,
    output_padding: int | tuple[int, int, int] = 0,
    mesh: DeviceMesh | None = None,
) -> torch.Tensor:
    """Run 1D-sharded distributed 3D transposed convolution with autograd.

    Args:
        x: Channels-first input tensor with shape ``(C_in, X_local, Y, Z)``.
        weight: Weight tensor with shape ``(C_in, C_out, K_x, K_y, K_z)``.
        bias: Optional bias tensor with shape ``(C_out,)``.
        stride: Scalar or per-axis transposed-convolution stride.
        padding: Scalar or per-axis transposed-convolution padding.
        output_padding: Scalar or per-axis output padding.
        mesh: Optional 1D ``DeviceMesh`` along the sharded X axis.

    Returns:
        Channels-first output tensor with shape
        ``(C_out, X_local * stride_x, Y_out, Z_out)``.

    Raises:
        TypeError: If scalar-or-tuple parameters have unsupported types.
        ValueError: If any transposed-convolution constraint is unsupported.
    """
    stride_t = _normalize_tuple(stride, "stride")
    padding_t = _normalize_tuple(padding, "padding")
    output_padding_t = _normalize_tuple(output_padding, "output_padding")
    _validate_conv_transpose_args(x, weight, bias, stride_t, padding_t, output_padding_t, mesh)

    return _conv_transpose3d_op(
        x,
        weight,
        bias,
        list(stride_t),
        list(padding_t),
        list(output_padding_t),
        group_name_from_mesh(mesh),
    )
