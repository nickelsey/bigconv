"""Distributed 3D convolution op (1D sharded along X).

Public entry point is ``conv3d``. It's registered as a ``torch.library.custom_op``
internally with a separate forward and backward. Forward composes a halo
exchange (X-axis) with a local ``F.conv3d``; backward reverses both — it runs
``aten.convolution_backward`` on the saved halo'd input, then routes halo
gradients back to their source ranks.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh

from bigconv.ops._dist_utils import group_name_from_mesh, resolve_group


_SUPPORTED_PADDING_MODES = ("zeros", "reflect", "replicate")
_SUPPORTED_STRIDES = (1, 2)
_MAX_KERNEL = 7


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _normalize_tuple(v: int | tuple[int, int, int], name: str) -> tuple[int, int, int]:
    if isinstance(v, int):
        return (v, v, v)
    if isinstance(v, tuple) and len(v) == 3 and all(isinstance(i, int) for i in v):
        return v
    raise TypeError(f"{name} must be an int or 3-tuple of ints, got {v!r}")


# ---------------------------------------------------------------------------
# Argument validator
# ---------------------------------------------------------------------------


def _validate_conv_args(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
    padding_mode: str,
    mesh: DeviceMesh | None,
) -> None:
    """Validate distributed convolution arguments.

    Checks cheap local shape and scalar constraints before the one collective,
    so early validation failures do not hang other ranks.

    Args:
        x: Local channels-first input tensor with shape ``(C_in, X_local, Y, Z)``.
        weight: Convolution weight with shape ``(C_out, C_in, K_x, K_y, K_z)``.
        bias: Optional bias tensor with shape ``(C_out,)``.
        stride: Per-axis convolution stride.
        padding: Per-axis convolution padding.
        padding_mode: Padding mode name.
        mesh: Optional 1D device mesh along the sharded X axis.

    Raises:
        ValueError: If shapes, scalar constraints, mesh rank, or distributed
            shard sizes are unsupported.
    """

    if x.ndim != 4:
        raise ValueError(
            f"x must be 4D (C_in, X_local, Y, Z), got {x.ndim}D with shape {tuple(x.shape)}"
        )
    C_in, X_local, _Y, _Z = x.shape

    if weight.ndim != 5:
        raise ValueError(
            f"weight must be 5D (C_out, C_in, K_x, K_y, K_z), got {weight.ndim}D "
            f"with shape {tuple(weight.shape)}"
        )
    C_out, C_in_w, K_x, K_y, K_z = weight.shape
    if C_in != C_in_w:
        raise ValueError(f"x C_in={C_in} doesn't match weight C_in={C_in_w}")

    if bias is not None:
        if bias.ndim != 1 or bias.shape[0] != C_out:
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
            raise ValueError(
                f"{axis_name}={p} exceeds (K-1)/2={half}; larger padding would require "
                f"a halo bigger than the neighbor's X_local can supply"
            )

    # Uniform output sharding requires same-padding along the sharded axis.
    P_x_required = (K_x - 1) // 2
    if padding[0] != P_x_required:
        raise ValueError(
            f"padding_x={padding[0]} must equal (K_x - 1) // 2 = {P_x_required} "
            f"(same-padding along the sharded X axis). Other X-paddings would "
            f"produce non-uniform output shards across ranks, which is currently "
            f"unsupported."
        )

    if padding_mode not in _SUPPORTED_PADDING_MODES:
        raise ValueError(f"padding_mode={padding_mode!r} must be one of {_SUPPORTED_PADDING_MODES}")

    if padding_mode == "reflect":
        for axis_name, size, p, k in (
            ("X", X_local, padding[0], K_x),
            ("Y", _Y, padding[1], K_y),
            ("Z", _Z, padding[2], K_z),
        ):
            halo = max(p, k - 1 - p) if axis_name == "X" else p
            if halo >= size:
                raise ValueError(
                    f"reflect padding on {axis_name} requires size > halo; "
                    f"{axis_name}={size}, halo={halo}"
                )

    if mesh is not None and mesh.ndim != 1:
        raise ValueError(
            f"conv3d expects a 1D DeviceMesh along X; "
            f"got {mesh.ndim}D mesh with shape {tuple(mesh.mesh.shape)}. "
            f"Pass a sliced sub-mesh like mesh['x']."
        )

    stride_x = stride[0]
    if X_local % stride_x != 0:
        raise ValueError(f"X_local={X_local} must be divisible by stride_x={stride_x}")

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


# ---------------------------------------------------------------------------
# Single-process reference (for testing)
# ---------------------------------------------------------------------------


def _reference_conv3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    stride: int | tuple[int, int, int] = 1,
    padding: int | tuple[int, int, int] = 0,
    padding_mode: str = "zeros",
) -> torch.Tensor:
    """Compute the single-process reference convolution used by tests.

    Args:
        x: Channels-first input tensor with shape ``(C_in, X, Y, Z)``.
        weight: Weight tensor with shape ``(C_out, C_in, K_x, K_y, K_z)``.
        bias: Optional bias tensor with shape ``(C_out,)``.
        stride: Scalar or per-axis convolution stride.
        padding: Scalar or per-axis convolution padding.
        padding_mode: Padding mode to use.

    Returns:
        Channels-first output tensor. The validator runs with ``mesh=None``.
    """
    stride_t = _normalize_tuple(stride, "stride")
    padding_t = _normalize_tuple(padding, "padding")
    _validate_conv_args(x, weight, bias, stride_t, padding_t, padding_mode, mesh=None)

    x_ncdhw = x.unsqueeze(0)

    if padding_mode == "zeros":
        out_ncdhw = F.conv3d(x_ncdhw, weight, bias, stride=stride_t, padding=padding_t)
    else:
        pad_amounts = [
            padding_t[2],
            padding_t[2],
            padding_t[1],
            padding_t[1],
            padding_t[0],
            padding_t[0],
        ]
        x_padded = F.pad(x_ncdhw, pad_amounts, mode=padding_mode)
        out_ncdhw = F.conv3d(x_padded, weight, bias, stride=stride_t, padding=0)

    return out_ncdhw.squeeze(0)


# ---------------------------------------------------------------------------
# Edge padding — forward + backward
# ---------------------------------------------------------------------------


def _apply_left_edge_padding(
    out: torch.Tensor, x: torch.Tensor, halo: int, padding_mode: str
) -> None:
    if padding_mode == "zeros":
        out[:, :halo].zero_()
    elif padding_mode == "replicate":
        out[:, :halo] = x[:, 0:1]
    elif padding_mode == "reflect":
        out[:, :halo] = x[:, 1 : halo + 1].flip(1)
    else:  # pragma: no cover
        raise ValueError(f"unknown padding_mode {padding_mode!r}")


def _apply_right_edge_padding(
    out: torch.Tensor, x: torch.Tensor, halo: int, padding_mode: str
) -> None:
    X_local = x.shape[1]
    if padding_mode == "zeros":
        out[:, -halo:].zero_()
    elif padding_mode == "replicate":
        out[:, -halo:] = x[:, -1:]
    elif padding_mode == "reflect":
        out[:, -halo:] = x[:, X_local - halo - 1 : X_local - 1].flip(1)
    else:  # pragma: no cover
        raise ValueError(f"unknown padding_mode {padding_mode!r}")


def _apply_left_edge_padding_backward(
    grad_x: torch.Tensor, grad_halo: torch.Tensor, padding_mode: str
) -> None:
    """Route left-edge halo gradients back to interior voxels.

    Forward computes ``out[:, :halo] = f(x)``. Backward adds ``grad_halo`` to
    the corresponding positions of ``grad_x``.

    Args:
        grad_x: Gradient tensor for the unpadded local input, mutated in place.
        grad_halo: Gradient tensor for the left halo slice.
        padding_mode: Padding mode used in the forward pass.
    """
    halo = grad_halo.shape[1]
    if padding_mode == "zeros":
        pass  # edge halo was a constant, no grad flows to x
    elif padding_mode == "replicate":
        grad_x[:, 0] += grad_halo.sum(dim=1)
    elif padding_mode == "reflect":
        # out[:, :halo] = x[:, 1:halo+1].flip(1)
        grad_x[:, 1 : halo + 1] += grad_halo.flip(1)
    else:  # pragma: no cover
        raise ValueError(f"unknown padding_mode {padding_mode!r}")


def _apply_right_edge_padding_backward(
    grad_x: torch.Tensor, grad_halo: torch.Tensor, padding_mode: str
) -> None:
    halo = grad_halo.shape[1]
    X_local = grad_x.shape[1]
    if padding_mode == "zeros":
        pass
    elif padding_mode == "replicate":
        grad_x[:, -1] += grad_halo.sum(dim=1)
    elif padding_mode == "reflect":
        # out[:, -halo:] = x[:, X_local-halo-1:X_local-1].flip(1)
        grad_x[:, X_local - halo - 1 : X_local - 1] += grad_halo.flip(1)
    else:  # pragma: no cover
        raise ValueError(f"unknown padding_mode {padding_mode!r}")


# ---------------------------------------------------------------------------
# Halo exchange — forward + backward
# ---------------------------------------------------------------------------


def _halo_exchange_pg(
    x: torch.Tensor,
    halo_left: int,
    halo_right: int,
    padding_mode: str,
    pg: ProcessGroup | None,
) -> torch.Tensor:
    """Exchange halo slices along the sharded X axis.

    Args:
        x: Local channels-first tensor with shape ``(C, X_local, Y, Z)``.
        halo_left: Number of X slices required from the left neighbor.
        halo_right: Number of X slices required from the right neighbor.
        padding_mode: Edge padding mode for boundary ranks.
        pg: Process group to exchange within. ``None`` means WORLD when
            distributed is initialized and single-process behavior otherwise.

    Returns:
        Tensor with left and right halos included along X.
    """
    if halo_left == 0 and halo_right == 0:
        # Custom op outputs cannot alias inputs. The no-halo path is used by
        # 1x1x1 convolutions, so return a distinct tensor even when x is already
        # contiguous.
        return x.contiguous().clone()

    C, X_local, Y, Z = x.shape

    # pg=None means "default / WORLD" when distributed is up; "no distributed"
    # when it isn't. Distinguishing them matters here — encoder hit the same
    # thing. See _halo_exchange_backward_pg for the mirror case.
    if dist.is_initialized():
        local_rank = dist.get_rank(pg)
        world_size = dist.get_world_size(pg)
    else:
        local_rank, world_size = 0, 1

    has_left = local_rank > 0
    has_right = local_rank < world_size - 1

    out = x.new_empty((C, X_local + halo_left + halo_right, Y, Z))
    out[:, halo_left : halo_left + X_local].copy_(x)

    # Recv sizes match *my* halo widths; send sizes match the *peer's* halo
    # widths (same, by uniform-X_local). My send to the left fills their right
    # halo (halo_right); my send to the right fills their left halo (halo_left).
    send_bufs: list[torch.Tensor] = []
    left_recv_buf: torch.Tensor | None = None
    right_recv_buf: torch.Tensor | None = None
    reqs: list[Any] = []

    if has_left:
        if halo_left > 0:
            left_recv_buf = torch.empty((C, halo_left, Y, Z), dtype=x.dtype, device=x.device)
            reqs.append(dist.irecv(left_recv_buf, local_rank - 1, group=pg))
        if halo_right > 0:
            send_bufs.append(x[:, :halo_right].contiguous())
            reqs.append(dist.isend(send_bufs[-1], local_rank - 1, group=pg))

    if has_right:
        if halo_left > 0:
            send_bufs.append(x[:, -halo_left:].contiguous())
            reqs.append(dist.isend(send_bufs[-1], local_rank + 1, group=pg))
        if halo_right > 0:
            right_recv_buf = torch.empty((C, halo_right, Y, Z), dtype=x.dtype, device=x.device)
            reqs.append(dist.irecv(right_recv_buf, local_rank + 1, group=pg))

    for req in reqs:
        req.wait()

    if left_recv_buf is not None:
        out[:, :halo_left].copy_(left_recv_buf)
    if right_recv_buf is not None:
        out[:, -halo_right:].copy_(right_recv_buf)

    if halo_left > 0 and not has_left:
        _apply_left_edge_padding(out, x, halo_left, padding_mode)
    if halo_right > 0 and not has_right:
        _apply_right_edge_padding(out, x, halo_right, padding_mode)

    return out


def _halo_exchange(
    x: torch.Tensor,
    halo_left: int,
    halo_right: int,
    padding_mode: str,
    mesh: DeviceMesh | None,
) -> torch.Tensor:
    """Exchange halo slices using a mesh wrapper.

    Args:
        x: Local channels-first tensor.
        halo_left: Number of X slices required from the left neighbor.
        halo_right: Number of X slices required from the right neighbor.
        padding_mode: Edge padding mode for boundary ranks.
        mesh: Optional 1D device mesh.

    Returns:
        Tensor with X-axis halos included.
    """
    pg = None if mesh is None else mesh.get_group()
    return _halo_exchange_pg(x, halo_left, halo_right, padding_mode, pg)


def _halo_exchange_backward_pg(
    grad_halo: torch.Tensor,
    halo_left: int,
    halo_right: int,
    padding_mode: str,
    pg: ProcessGroup | None,
) -> torch.Tensor:
    """Reverse ``_halo_exchange_pg`` for autograd.

    Accumulates three gradient sources: the interior slice of ``grad_halo``,
    remote halo gradients returned by neighboring ranks, and local edge
    padding gradients for boundary ranks.

    Args:
        grad_halo: Gradient on the haloed tensor with shape
            ``(C, X_local + halo_left + halo_right, Y, Z)``.
        halo_left: Number of left halo slices used in the forward pass.
        halo_right: Number of right halo slices used in the forward pass.
        padding_mode: Edge padding mode used in the forward pass.
        pg: Process group used for the original halo exchange.

    Returns:
        Gradient tensor with shape ``(C, X_local, Y, Z)``.
    """
    if halo_left == 0 and halo_right == 0:
        return grad_halo.contiguous() if not grad_halo.is_contiguous() else grad_halo

    C, X_padded, Y, Z = grad_halo.shape
    X_local = X_padded - halo_left - halo_right

    grad_x = grad_halo[:, halo_left : halo_left + X_local].contiguous()

    if dist.is_initialized():
        local_rank = dist.get_rank(pg)
        world_size = dist.get_world_size(pg)
    else:
        local_rank, world_size = 0, 1

    has_left = local_rank > 0
    has_right = local_rank < world_size - 1

    send_bufs: list[torch.Tensor] = []
    left_recv_buf: torch.Tensor | None = None  # adds into grad_x[:halo_right]
    right_recv_buf: torch.Tensor | None = None  # adds into grad_x[-halo_left:]
    reqs: list[Any] = []

    # Send/recv sizes mirror the forward exchange: if I sent halo_right to the
    # left in forward, the left neighbor now sends halo_right back to me.
    if has_left:
        if halo_left > 0:
            send_bufs.append(grad_halo[:, :halo_left].contiguous())
            reqs.append(dist.isend(send_bufs[-1], local_rank - 1, group=pg))
        if halo_right > 0:
            left_recv_buf = torch.empty(
                (C, halo_right, Y, Z), dtype=grad_x.dtype, device=grad_x.device
            )
            reqs.append(dist.irecv(left_recv_buf, local_rank - 1, group=pg))

    if has_right:
        if halo_right > 0:
            send_bufs.append(grad_halo[:, -halo_right:].contiguous())
            reqs.append(dist.isend(send_bufs[-1], local_rank + 1, group=pg))
        if halo_left > 0:
            right_recv_buf = torch.empty(
                (C, halo_left, Y, Z), dtype=grad_x.dtype, device=grad_x.device
            )
            reqs.append(dist.irecv(right_recv_buf, local_rank + 1, group=pg))

    for req in reqs:
        req.wait()

    if left_recv_buf is not None:
        grad_x[:, :halo_right] += left_recv_buf
    if right_recv_buf is not None:
        grad_x[:, -halo_left:] += right_recv_buf

    if halo_left > 0 and not has_left:
        _apply_left_edge_padding_backward(grad_x, grad_halo[:, :halo_left], padding_mode)
    if halo_right > 0 and not has_right:
        _apply_right_edge_padding_backward(grad_x, grad_halo[:, -halo_right:], padding_mode)

    return grad_x


# ---------------------------------------------------------------------------
# Local (post halo-exchange) conv on the halo'd tensor
# ---------------------------------------------------------------------------


def _local_conv_on_halo(
    x_halo: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
    padding_mode: str,
) -> torch.Tensor:
    """Run local convolution on a halo-exchanged tensor.

    ``x_halo`` already includes X-axis padding from the halo exchange, so this
    function handles only Y/Z padding and the convolution itself.

    Args:
        x_halo: Channels-first tensor with X-axis halos included.
        weight: Convolution weight tensor.
        bias: Optional bias tensor.
        stride: Per-axis convolution stride.
        padding: Per-axis convolution padding.
        padding_mode: Padding mode for Y/Z padding.

    Returns:
        Local channels-first convolution output.
    """
    x_ncdhw = x_halo.unsqueeze(0)

    P_y, P_z = padding[1], padding[2]
    if P_y == 0 and P_z == 0:
        x_padded = x_ncdhw
    else:
        pad_vals = [P_z, P_z, P_y, P_y, 0, 0]
        if padding_mode == "zeros":
            x_padded = F.pad(x_ncdhw, pad_vals, mode="constant", value=0)
        else:
            x_padded = F.pad(x_ncdhw, pad_vals, mode=padding_mode)

    out_ncdhw = F.conv3d(x_padded, weight, bias, stride=stride, padding=0)
    return out_ncdhw.squeeze(0)


# ---------------------------------------------------------------------------
# Custom op + autograd registration
# ---------------------------------------------------------------------------


@torch.library.custom_op("bigconv::conv3d", mutates_args=())
def _conv3d_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: list[int],
    padding: list[int],
    padding_mode: str,
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the distributed convolution custom op forward pass.

    Args:
        x: Local channels-first input tensor.
        weight: Convolution weight tensor.
        bias: Optional bias tensor.
        stride: Per-axis stride encoded as a list for the custom-op schema.
        padding: Per-axis padding encoded as a list for the custom-op schema.
        padding_mode: Padding mode to use.
        group_name: Process-group name used to resolve the process group.

    Returns:
        Tuple of ``(output, x_halo)``. ``x_halo`` is saved for backward so
        autograd setup does not repeat the collective.
    """
    pg = resolve_group(group_name)
    K_x = weight.shape[2]
    halo_left = padding[0]
    halo_right = K_x - stride[0] - padding[0]

    x_halo = _halo_exchange_pg(x, halo_left, halo_right, padding_mode, pg)
    stride_t = (stride[0], stride[1], stride[2])
    padding_t = (padding[0], padding[1], padding[2])
    out = _local_conv_on_halo(x_halo, weight, bias, stride_t, padding_t, padding_mode)
    return out, x_halo


def _fake_conv3d_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: list[int],
    padding: list[int],
    padding_mode: str,
    group_name: str,
):
    C_in, X_local, Y, Z = x.shape
    C_out, _, K_x, K_y, K_z = weight.shape
    halo_left = padding[0]
    halo_right = K_x - stride[0] - padding[0]
    X_padded = X_local + halo_left + halo_right

    X_out = X_local // stride[0]
    Y_out = (Y + 2 * padding[1] - K_y) // stride[1] + 1
    Z_out = (Z + 2 * padding[2] - K_z) // stride[2] + 1

    out = x.new_empty((C_out, X_out, Y_out, Z_out))
    x_halo = x.new_empty((C_in, X_padded, Y, Z))
    return out, x_halo


def _pad_backward_yz(
    grad_padded: torch.Tensor,
    x_nchw: torch.Tensor,
    P_y: int,
    P_z: int,
    padding_mode: str,
) -> torch.Tensor:
    """Run native backward for the Y/Z padding in ``_local_conv_on_halo``.

    Args:
        grad_padded: Gradient tensor with shape
            ``(1, C, X_halo, Y + 2 * P_y, Z + 2 * P_z)``.
        x_nchw: Pre-pad tensor whose shape is used for the returned gradient.
        P_y: Padding amount on the Y axis.
        P_z: Padding amount on the Z axis.
        padding_mode: Padding mode used in the forward pass.

    Returns:
        Gradient tensor with the same shape as ``x_nchw``. For zeros padding,
        this is a slice. For reflect and replicate padding, this dispatches to
        native aten backward kernels so padded contributions accumulate into
        interior positions.
    """
    if P_y == 0 and P_z == 0:
        return grad_padded

    if padding_mode == "zeros":
        y_slice = slice(P_y, -P_y if P_y > 0 else None)
        z_slice = slice(P_z, -P_z if P_z > 0 else None)
        return grad_padded[:, :, :, y_slice, z_slice].contiguous()

    pad = [P_z, P_z, P_y, P_y, 0, 0]
    if padding_mode == "reflect":
        return torch.ops.aten.reflection_pad3d_backward(grad_padded, x_nchw, pad)
    if padding_mode == "replicate":
        return torch.ops.aten.replication_pad3d_backward(grad_padded, x_nchw, pad)
    raise ValueError(f"unknown padding_mode {padding_mode!r}")  # pragma: no cover


def _setup_context(ctx, inputs, output):
    x, weight, bias, stride, padding, padding_mode, group_name = inputs
    _out, x_halo = output
    ctx.mark_non_differentiable(x_halo)
    if bias is not None:
        ctx.save_for_backward(x_halo, weight, bias)
    else:
        ctx.save_for_backward(x_halo, weight)
    ctx.stride = tuple(stride)
    ctx.padding = tuple(padding)
    ctx.padding_mode = padding_mode
    ctx.group_name = group_name
    ctx.has_bias = bias is not None
    K_x = weight.shape[2]
    ctx.halo_left = padding[0]
    ctx.halo_right = K_x - stride[0] - padding[0]


def _conv3d_backward(ctx, grad_out, _grad_x_halo_ignored):
    if ctx.has_bias:
        x_halo, weight, _bias = ctx.saved_tensors
    else:
        x_halo, weight = ctx.saved_tensors
    stride = ctx.stride
    padding = ctx.padding
    padding_mode = ctx.padding_mode
    group_name = ctx.group_name
    has_bias = ctx.has_bias
    halo_left = ctx.halo_left
    halo_right = ctx.halo_right

    # NCDHW view of x_halo — serves both as the pre-Y/Z-pad input to the conv
    # and as shape info for the pad backward.
    x_ncdhw = x_halo.unsqueeze(0)

    # Rebuild x_padded with Y/Z padding (no collectives here).
    P_y, P_z = padding[1], padding[2]
    if P_y > 0 or P_z > 0:
        pad_vals = [P_z, P_z, P_y, P_y, 0, 0]
        if padding_mode == "zeros":
            x_padded = F.pad(x_ncdhw, pad_vals, mode="constant", value=0)
        else:
            x_padded = F.pad(x_ncdhw, pad_vals, mode=padding_mode)
    else:
        x_padded = x_ncdhw

    grad_out_ncdhw = grad_out.unsqueeze(0).contiguous()

    C_out = weight.shape[0]
    grad_x_padded_nchw, grad_weight, grad_bias = torch.ops.aten.convolution_backward(
        grad_out_ncdhw,
        x_padded,
        weight,
        [C_out] if has_bias else None,
        list(stride),
        [0, 0, 0],
        [1, 1, 1],
        False,
        [0, 0, 0],
        1,
        [True, True, has_bias],
    )

    # Native Y/Z pad backward — for reflect/replicate this dispatches to the
    # aten kernel that accumulates padded grads into the reflected/replicated
    # interior positions, instead of dropping them.
    grad_x_halo_ncdhw = _pad_backward_yz(grad_x_padded_nchw, x_ncdhw, P_y, P_z, padding_mode)

    grad_x_halo = grad_x_halo_ncdhw.squeeze(0).contiguous()

    # Halo exchange backward: send halo grads back to source, accumulate
    # neighbor contributions, handle edge reflections/replications along X.
    pg = resolve_group(group_name)
    grad_x = _halo_exchange_backward_pg(grad_x_halo, halo_left, halo_right, padding_mode, pg)

    # Inputs: x, weight, bias, stride, padding, padding_mode, group_name
    return (
        grad_x,
        grad_weight,
        grad_bias if has_bias else None,
        None,
        None,
        None,
        None,
    )


_conv3d_op.register_fake(_fake_conv3d_op)
_conv3d_op.register_autograd(_conv3d_backward, setup_context=_setup_context)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def conv3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    stride: int | tuple[int, int, int] = 1,
    padding: int | tuple[int, int, int] = 0,
    padding_mode: str = "zeros",
    mesh: DeviceMesh | None = None,
) -> torch.Tensor:
    """Run 1D-sharded distributed 3D convolution with autograd.

    Args:
        x: Channels-first input tensor with shape ``(C_in, X_local, Y, Z)``.
        weight: Weight tensor with shape ``(C_out, C_in, K_x, K_y, K_z)``.
        bias: Optional bias tensor with shape ``(C_out,)``.
        stride: Scalar or per-axis convolution stride.
        padding: Scalar or per-axis convolution padding.
        padding_mode: Padding mode. Supported values are ``"zeros"``,
            ``"reflect"``, and ``"replicate"``.
        mesh: Optional 1D ``DeviceMesh`` along the sharded X axis.

    Returns:
        Channels-first output tensor with shape
        ``(C_out, X_local / stride_x, Y_out, Z_out)``.

    Raises:
        TypeError: If scalar-or-tuple parameters have unsupported types.
        ValueError: If any convolution constraint is unsupported. Constraints
            include uniform ``X_local`` across ranks, ``X_local % stride_x == 0``,
            odd kernels in ``[1, 7]``, per-axis stride in ``{1, 2}``,
            ``padding_x == (K_x - 1) // 2``, supported padding modes, and a 1D
            mesh when ``mesh`` is provided.
    """
    stride_t = _normalize_tuple(stride, "stride")
    padding_t = _normalize_tuple(padding, "padding")
    _validate_conv_args(x, weight, bias, stride_t, padding_t, padding_mode, mesh)

    group_name = group_name_from_mesh(mesh)
    out, _x_halo = _conv3d_op(
        x,
        weight,
        bias,
        list(stride_t),
        list(padding_t),
        padding_mode,
        group_name,
    )
    return out
