import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh

from bigconv.ops._dist_utils import group_name_from_mesh, resolve_group


_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


def _validate_encoder_scatter_args(
    feat: torch.Tensor,
    dest_rank: torch.Tensor,
    xyz: torch.Tensor,
    X_local: int,
    Y: int,
    Z: int,
    mesh: DeviceMesh | None,
) -> None:
    """Validate encoder scatter inputs before launching collectives.

    Args:
        feat: Local feature tensor with shape ``(N, C)``.
        dest_rank: Destination rank for each feature with shape ``(N,)``.
        xyz: Local voxel coordinates for each feature with shape ``(N, 3)``.
        X_local: Local voxel grid size along X.
        Y: Voxel grid size along Y.
        Z: Voxel grid size along Z.
        mesh: Optional 1D device mesh used for rank bounds.

    Raises:
        RuntimeError: If torch.distributed is not initialized.
        TypeError: If scalar dimensions or index dtypes are unsupported.
        ValueError: If shapes, devices, rank bounds, coordinate bounds, or mesh
            rank are invalid.
    """
    if not dist.is_initialized():
        raise RuntimeError("encoder_scatter_to_voxel requires an initialized process group")

    if feat.ndim != 2:
        raise ValueError(f"feat must be 2D (N, C), got {feat.ndim}D with shape {tuple(feat.shape)}")
    N, C = feat.shape
    if C <= 0:
        raise ValueError(f"feat must have C > 0, got C={C}")

    if dest_rank.ndim != 1:
        raise ValueError(
            f"dest_rank must be 1D (N,), got {dest_rank.ndim}D with shape {tuple(dest_rank.shape)}"
        )
    if dest_rank.shape[0] != N:
        raise ValueError(f"dest_rank length={dest_rank.shape[0]} must match feat N={N}")
    if dest_rank.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"dest_rank must have an integer dtype, got {dest_rank.dtype}")

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape (N, 3), got {tuple(xyz.shape)}")
    if xyz.shape[0] != N:
        raise ValueError(f"xyz length={xyz.shape[0]} must match feat N={N}")
    if xyz.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"xyz must have an integer dtype, got {xyz.dtype}")

    if dest_rank.device != feat.device:
        raise ValueError(
            f"dest_rank device={dest_rank.device} must match feat device={feat.device}"
        )
    if xyz.device != feat.device:
        raise ValueError(f"xyz device={xyz.device} must match feat device={feat.device}")

    for name, value in (("X_local", X_local), ("Y", Y), ("Z", Z)):
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an int, got {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    if mesh is not None and mesh.ndim != 1:
        raise ValueError(
            f"encoder_scatter_to_voxel expects a 1D DeviceMesh; got {mesh.ndim}D "
            f"mesh with shape {tuple(mesh.mesh.shape)}. Pass a sliced sub-mesh like mesh['x']."
        )

    pg = None if mesh is None else mesh.get_group()
    world_size = dist.get_world_size(pg)
    if N == 0:
        return

    min_dest = int(dest_rank.min().item())
    max_dest = int(dest_rank.max().item())
    if min_dest < 0 or max_dest >= world_size:
        raise ValueError(
            f"dest_rank values must be in [0, {world_size}); got min={min_dest}, max={max_dest}"
        )

    mins = xyz.min(dim=0).values
    maxs = xyz.max(dim=0).values
    bounds = (X_local, Y, Z)
    names = ("x", "y", "z")
    for axis, size, min_val_t, max_val_t in zip(names, bounds, mins, maxs):
        min_val = int(min_val_t.item())
        max_val = int(max_val_t.item())
        if min_val < 0 or max_val >= size:
            raise ValueError(
                f"xyz {axis} coordinates must be in [0, {size}); got min={min_val}, max={max_val}"
            )


def _pack_for_scatter(
    feat: torch.Tensor,
    dest_rank: torch.Tensor,
    xyz: torch.Tensor,
    world_size: int,
    pg: ProcessGroup | None,
):
    N = feat.shape[0]
    device = feat.device

    sort_perm = torch.argsort(dest_rank, stable=True)
    unsort_perm = torch.empty_like(sort_perm)
    unsort_perm[sort_perm] = torch.arange(N, device=device)

    packed_feat = feat.index_select(0, sort_perm)
    packed_dest = dest_rank.index_select(0, sort_perm)
    packed_xyz = xyz.index_select(0, sort_perm)

    send_counts = torch.bincount(packed_dest.to(torch.long), minlength=world_size)
    recv_counts = torch.empty_like(send_counts)

    dist.all_to_all_single(recv_counts, send_counts, group=pg)

    return (
        packed_feat,
        packed_xyz,
        sort_perm,
        unsort_perm,
        send_counts,
        recv_counts,
    )


def _distribute_tensor_by_rank(
    packed_tensor: torch.Tensor,
    send_counts: torch.Tensor,
    recv_counts: torch.Tensor,
    pg: ProcessGroup | None,
):
    out_shape = (int(recv_counts.sum().item()),) + tuple(packed_tensor.shape[1:])
    out = torch.empty(out_shape, dtype=packed_tensor.dtype, device=packed_tensor.device)

    dist.all_to_all_single(
        out,
        packed_tensor.contiguous(),
        output_split_sizes=recv_counts.tolist(),
        input_split_sizes=send_counts.tolist(),
        group=pg,
    )

    return out


def _local_scatter(
    recv_feat: torch.Tensor,
    recv_xyz: torch.Tensor,
    X_local: int,
    Y: int,
    Z: int,
):
    C = recv_feat.shape[1]
    out = torch.zeros(
        (C, X_local * Y * Z),
        dtype=recv_feat.dtype,
        device=recv_feat.device,
    )
    lin = ((recv_xyz[:, 0] * Y) + recv_xyz[:, 1]) * Z + recv_xyz[:, 2]
    out.index_add_(1, lin, recv_feat.transpose(0, 1))
    return out.view(C, X_local, Y, Z)


def _local_gather_grad(
    grad_out: torch.Tensor,
    recv_xyz: torch.Tensor,
):
    C, X_local, Y, Z = grad_out.shape
    grad_flat = grad_out.contiguous().view(C, X_local * Y * Z)
    lin = ((recv_xyz[:, 0] * Y) + recv_xyz[:, 1]) * Z + recv_xyz[:, 2]
    return grad_flat.index_select(1, lin).transpose(0, 1)


# the op returns the voxel grid plus the four tensors backward needs
# (unsort_perm, send_counts, recv_counts, recv_xyz)
@torch.library.custom_op("bigconv::encoder_scatter_to_voxel", mutates_args=())
def _encoder_scatter_to_voxel_op(
    feat: torch.Tensor,
    dest_rank: torch.Tensor,
    xyz: torch.Tensor,
    X_local: int,
    Y: int,
    Z: int,
    group_name: str = "",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pg = resolve_group(group_name)
    world_size = dist.get_world_size(pg)

    (
        packed_feat,
        packed_xyz,
        _sort_perm,
        unsort_perm,
        send_counts,
        recv_counts,
    ) = _pack_for_scatter(feat, dest_rank, xyz, world_size, pg)

    recv_feat = _distribute_tensor_by_rank(packed_feat, send_counts, recv_counts, pg)
    recv_xyz = _distribute_tensor_by_rank(packed_xyz, send_counts, recv_counts, pg)

    voxel = _local_scatter(recv_feat, recv_xyz, X_local, Y, Z)

    return voxel, unsort_perm, send_counts, recv_counts, recv_xyz


def _fake(
    feat: torch.Tensor,
    dest_rank: torch.Tensor,
    xyz: torch.Tensor,
    X_local: int,
    Y: int,
    Z: int,
    group_name: str = "",
):
    N, C = feat.shape
    world_size = dist.get_world_size(resolve_group(group_name))

    voxel = feat.new_empty((C, X_local, Y, Z))
    unsort_perm = torch.empty((N,), dtype=torch.int64, device=feat.device)
    send_counts = torch.empty((world_size,), dtype=torch.int64, device=feat.device)
    recv_counts = torch.empty((world_size,), dtype=torch.int64, device=feat.device)

    # recv_xyz's length is data-dependent (depends on what other ranks send us)
    M = torch.library.get_ctx().new_dynamic_size()
    recv_xyz = torch.empty((M, 3), dtype=xyz.dtype, device=xyz.device)

    return voxel, unsort_perm, send_counts, recv_counts, recv_xyz


def _setup_context(ctx, inputs, output):
    _voxel, unsort_perm, send_counts, recv_counts, recv_xyz = output
    _feat, _dest_rank, _xyz, X_local, Y, Z, group_name = inputs
    ctx.mark_non_differentiable(unsort_perm, send_counts, recv_counts, recv_xyz)
    ctx.save_for_backward(unsort_perm, send_counts, recv_counts, recv_xyz)
    ctx.X_local = X_local
    ctx.Y = Y
    ctx.Z = Z
    ctx.group_name = group_name


def _backward(ctx, grad_voxel, *_non_diff_grads):
    unsort_perm, send_counts, recv_counts, recv_xyz = ctx.saved_tensors
    pg = resolve_group(ctx.group_name)
    C = grad_voxel.shape[0]

    grad_recv_feat = _local_gather_grad(grad_voxel, recv_xyz)

    grad_packed = torch.empty(
        (int(send_counts.sum().item()), C),
        dtype=grad_recv_feat.dtype,
        device=grad_recv_feat.device,
    )
    dist.all_to_all_single(
        grad_packed,
        grad_recv_feat.contiguous(),
        output_split_sizes=send_counts.tolist(),
        input_split_sizes=recv_counts.tolist(),
        group=pg,
    )

    grad_feat = grad_packed.index_select(0, unsort_perm)

    return grad_feat, None, None, None, None, None, None


_encoder_scatter_to_voxel_op.register_fake(_fake)
_encoder_scatter_to_voxel_op.register_autograd(_backward, setup_context=_setup_context)


def encoder_scatter_to_voxel(
    feat: torch.Tensor,
    dest_rank: torch.Tensor,
    xyz: torch.Tensor,
    X_local: int,
    Y: int,
    Z: int,
    mesh: DeviceMesh | None = None,
) -> torch.Tensor:
    """Scatter point features into destination-rank voxel grids.

    Args:
        feat: Local feature tensor with shape ``(N, C)``.
        dest_rank: Destination rank for each feature with shape ``(N,)``.
        xyz: Local voxel coordinates on the destination rank with shape
            ``(N, 3)``. Coordinates are ordered as ``(x, y, z)``.
        X_local: Local voxel grid size along X.
        Y: Voxel grid size along Y.
        Z: Voxel grid size along Z.
        mesh: Optional 1D device mesh. If omitted, the default process group is
            used.

    Returns:
        Local voxel grid tensor with shape ``(C, X_local, Y, Z)``.

    Raises:
        RuntimeError: If torch.distributed is not initialized.
        TypeError: If scalar dimensions or index dtypes are unsupported.
        ValueError: If shapes, devices, rank bounds, coordinate bounds, or mesh
            rank are invalid.
    """
    _validate_encoder_scatter_args(feat, dest_rank, xyz, X_local, Y, Z, mesh)
    voxel, *_ = _encoder_scatter_to_voxel_op(
        feat, dest_rank, xyz, X_local, Y, Z, group_name_from_mesh(mesh)
    )
    return voxel
