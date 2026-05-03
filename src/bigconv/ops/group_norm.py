from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh

from bigconv.ops._dist_utils import group_name_from_mesh, resolve_group


def _stats_dtype(x: torch.Tensor) -> torch.dtype:
    if x.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return x.dtype


def _maybe_all_reduce(t: torch.Tensor, pg: ProcessGroup | None) -> torch.Tensor:
    if dist.is_initialized() and dist.get_world_size(pg) > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group=pg)
    return t


def _validate_group_norm_args(
    x: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
    mesh: DeviceMesh | None,
) -> None:
    """Validate distributed group normalization arguments.

    Args:
        x: Channels-first input tensor with shape ``(C, X_local, Y, Z)``.
        num_groups: Number of channel groups.
        weight: Optional affine weight with shape ``(C,)``.
        bias: Optional affine bias with shape ``(C,)``.
        eps: Numerical stability epsilon.
        mesh: Optional 1D device mesh for distributed statistics.

    Raises:
        TypeError: If scalar argument types are unsupported.
        ValueError: If tensor ranks, shapes, devices, or mesh rank are invalid.
    """
    if x.ndim != 4:
        raise ValueError(
            f"x must be 4D (C, X_local, Y, Z), got {x.ndim}D with shape {tuple(x.shape)}"
        )
    C = x.shape[0]
    if C <= 0:
        raise ValueError(f"x must have C > 0, got C={C}")

    if not isinstance(num_groups, int):
        raise TypeError(f"num_groups must be an int, got {type(num_groups).__name__}")
    if num_groups <= 0:
        raise ValueError(f"num_groups must be positive, got {num_groups}")
    if C % num_groups != 0:
        raise ValueError(f"C={C} must be divisible by num_groups={num_groups}")

    if not isinstance(eps, float):
        raise TypeError(f"eps must be a float, got {type(eps).__name__}")
    if eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps}")

    for name, tensor in (("weight", weight), ("bias", bias)):
        if tensor is None:
            continue
        if tensor.ndim != 1 or tensor.shape[0] != C:
            raise ValueError(f"{name} must have shape ({C},), got {tuple(tensor.shape)}")
        if tensor.device != x.device:
            raise ValueError(f"{name} device={tensor.device} must match x device={x.device}")

    if mesh is not None and mesh.ndim != 1:
        raise ValueError(
            f"group_norm expects a 1D DeviceMesh; got {mesh.ndim}D mesh with "
            f"shape {tuple(mesh.mesh.shape)}. Pass a sliced sub-mesh like mesh['x']."
        )


@torch.library.custom_op("bigconv::group_norm", mutates_args=())
def _group_norm_op(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    num_groups: int,
    eps: float,
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pg = resolve_group(group_name)
    C, X_local, Y, Z = x.shape
    channels_per_group = C // num_groups
    stats_dtype = _stats_dtype(x)

    x_grouped = x.reshape(num_groups, channels_per_group, X_local, Y, Z).to(stats_dtype)
    reduce_dims = (1, 2, 3, 4)
    group_sum = x_grouped.sum(dim=reduce_dims)
    group_sumsq = (x_grouped * x_grouped).sum(dim=reduce_dims)
    count = torch.full(
        (num_groups,),
        channels_per_group * X_local * Y * Z,
        dtype=stats_dtype,
        device=x.device,
    )

    _maybe_all_reduce(group_sum, pg)
    _maybe_all_reduce(group_sumsq, pg)
    _maybe_all_reduce(count, pg)

    mean = group_sum / count
    var = torch.clamp(group_sumsq / count - mean * mean, min=0)
    inv_std = torch.rsqrt(var + eps)

    x_hat = (x_grouped - mean.view(num_groups, 1, 1, 1, 1)) * inv_std.view(num_groups, 1, 1, 1, 1)
    x_hat = x_hat.reshape_as(x).to(x.dtype)

    out = x_hat.clone()
    if weight is not None:
        out = out * weight.view(C, 1, 1, 1)
    if bias is not None:
        out = out + bias.view(C, 1, 1, 1)

    return out, x_hat, inv_std.to(x.dtype)


def _fake_group_norm_op(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    num_groups: int,
    eps: float,
    group_name: str,
):
    out = x.new_empty(x.shape)
    x_hat = x.new_empty(x.shape)
    inv_std = x.new_empty((num_groups,))
    return out, x_hat, inv_std


def _setup_context(ctx, inputs, output):
    _x, weight, bias, num_groups, _eps, group_name = inputs
    _out, x_hat, inv_std = output
    ctx.mark_non_differentiable(x_hat, inv_std)
    if weight is not None:
        ctx.save_for_backward(x_hat, inv_std, weight)
    else:
        ctx.save_for_backward(x_hat, inv_std)
    ctx.has_weight = weight is not None
    ctx.has_bias = bias is not None
    ctx.num_groups = num_groups
    ctx.group_name = group_name


def _group_norm_backward(ctx, grad_out, _grad_x_hat_ignored, _grad_inv_std_ignored):
    if ctx.has_weight:
        x_hat, inv_std, weight = ctx.saved_tensors
    else:
        x_hat, inv_std = ctx.saved_tensors
        weight = None

    pg = resolve_group(ctx.group_name)
    num_groups = ctx.num_groups
    C, X_local, Y, Z = grad_out.shape
    channels_per_group = C // num_groups
    stats_dtype = _stats_dtype(grad_out)

    grad_norm = grad_out if weight is None else grad_out * weight.view(C, 1, 1, 1)
    grad_grouped = grad_norm.reshape(num_groups, channels_per_group, X_local, Y, Z).to(stats_dtype)
    x_hat_grouped = x_hat.reshape(num_groups, channels_per_group, X_local, Y, Z).to(stats_dtype)

    reduce_dims = (1, 2, 3, 4)
    grad_sum = grad_grouped.sum(dim=reduce_dims)
    grad_xhat_sum = (grad_grouped * x_hat_grouped).sum(dim=reduce_dims)
    count = torch.full(
        (num_groups,),
        channels_per_group * X_local * Y * Z,
        dtype=stats_dtype,
        device=grad_out.device,
    )

    _maybe_all_reduce(grad_sum, pg)
    _maybe_all_reduce(grad_xhat_sum, pg)
    _maybe_all_reduce(count, pg)

    grad_mean = grad_sum / count
    grad_xhat_mean = grad_xhat_sum / count
    grad_x_grouped = (
        grad_grouped
        - grad_mean.view(num_groups, 1, 1, 1, 1)
        - x_hat_grouped * grad_xhat_mean.view(num_groups, 1, 1, 1, 1)
    ) * inv_std.to(stats_dtype).view(num_groups, 1, 1, 1, 1)
    grad_x = grad_x_grouped.reshape_as(grad_out).to(grad_out.dtype)

    grad_weight = None
    if weight is not None:
        grad_weight = (grad_out * x_hat).sum(dim=(1, 2, 3))
    grad_bias = grad_out.sum(dim=(1, 2, 3)) if ctx.has_bias else None

    return grad_x, grad_weight, grad_bias, None, None, None


_group_norm_op.register_fake(_fake_group_norm_op)
_group_norm_op.register_autograd(_group_norm_backward, setup_context=_setup_context)


def group_norm(
    x: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
    mesh: DeviceMesh | None = None,
) -> torch.Tensor:
    """Run distributed group normalization over an X-sharded 3D tensor.

    Statistics are computed over each channel group and all spatial positions
    in the full distributed X extent. The public tensor layout is
    channels-first ``(C, X_local, Y, Z)``.

    Args:
        x: Channels-first local input tensor with shape ``(C, X_local, Y, Z)``.
        num_groups: Number of channel groups.
        weight: Optional affine weight with shape ``(C,)``.
        bias: Optional affine bias with shape ``(C,)``.
        eps: Numerical stability epsilon.
        mesh: Optional 1D device mesh for the sharded X axis.

    Returns:
        Local normalized tensor with shape ``(C, X_local, Y, Z)``.

    Raises:
        TypeError: If scalar argument types are unsupported.
        ValueError: If tensor ranks, shapes, devices, or mesh rank are invalid.
    """
    _validate_group_norm_args(x, num_groups, weight, bias, eps, mesh)
    out, _x_hat, _inv_std = _group_norm_op(
        x,
        weight,
        bias,
        num_groups,
        eps,
        group_name_from_mesh(mesh),
    )
    return out


def _reference_group_norm(
    x: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute a single-process group norm reference for tests."""
    _validate_group_norm_args(x, num_groups, weight, bias, eps, mesh=None)
    return F.group_norm(x.unsqueeze(0), num_groups, weight, bias, eps).squeeze(0)
