import torch
import torch.distributed as dist


def _resolve_pg(pg: dist.GroupName | None) -> dist.ProcessGroup:
    if pg is None:
        return dist.group.WORLD
    return dist.distributed_c10d._resolve_process_group(pg)


def _pack_for_scatter(
    feat: torch.Tensor,
    dest_rank: torch.Tensor,
    xyz: torch.Tensor,
    world_size: int,
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

    dist.all_to_all_single(recv_counts, send_counts)

    return (
        packed_feat,
        packed_xyz,
        sort_perm,
        unsort_perm,
        send_counts,
        recv_counts,
    )


def _distribute_tensor_by_rank(
    packed_tensor: torch.Tensor, send_counts: torch.Tensor, recv_counts: torch.Tensor
):
    out_shape = (int(recv_counts.sum().item()),) + tuple(packed_tensor.shape[1:])
    out = torch.empty(out_shape, dtype=packed_tensor.dtype, device=packed_tensor.device)

    dist.all_to_all_single(
        out,
        packed_tensor.contiguous(),
        output_split_sizes=recv_counts.tolist(),
        input_split_sizes=send_counts.tolist(),
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
        (X_local * Y * Z, C),
        dtype=recv_feat.dtype,
        device=recv_feat.device,
    )
    lin = ((recv_xyz[:, 0] * Y) + recv_xyz[:, 1]) * Z + recv_xyz[:, 2]
    out.index_add_(0, lin, recv_feat)
    return out.view(X_local, Y, Z, C)


def _local_gather_grad(
    grad_out: torch.Tensor,
    recv_xyz: torch.Tensor,
):
    X_local, Y, Z, C = grad_out.shape
    grad_flat = grad_out.contiguous().view(X_local * Y * Z, C)
    lin = ((recv_xyz[:, 0] * Y) + recv_xyz[:, 1]) * Z + recv_xyz[:, 2]
    return grad_flat.index_select(0, lin)


@torch.library.custom_op("bigconv::encoder_scatter_to_voxel", mutates_args=())
def encoder_scatter_to_voxel(
    feat: torch.Tensor,
    dest_rank: torch.Tensor,
    xyz: torch.Tensor,
    X_local: int,
    Y: int,
    Z: int,
    pg: dist.GroupName | None = None,
) -> torch.Tensor:
    world_size = dist.get_world_size(_resolve_pg(pg))

    (
        packed_feat,
        packed_xyz,
        _sort_perm,
        _unsort_perm,
        send_counts,
        recv_counts,
    ) = _pack_for_scatter(feat, dest_rank, xyz, world_size)

    recv_feat = _distribute_tensor_by_rank(packed_feat, send_counts, recv_counts)
    recv_xyz = _distribute_tensor_by_rank(packed_xyz, send_counts, recv_counts)

    return _local_scatter(recv_feat, recv_xyz, X_local, Y, Z)


def _fake(
    feat: torch.Tensor,
    dest_rank: torch.Tensor,
    xyz: torch.Tensor,
    X_local: int,
    Y: int,
    Z: int,
    pg: dist.GroupName | None = None,
):
    return feat.new_empty((X_local, Y, Z, feat.shape[1]))


def _setup_context(ctx, inputs, output):
    feat, dest_rank, xyz, X_local, Y, Z, pg = inputs
    world_size = dist.get_world_size(_resolve_pg(pg))

    (
        _packed_feat,
        packed_xyz,
        _sort_perm,
        unsort_perm,
        send_counts,
        recv_counts,
    ) = _pack_for_scatter(feat.detach(), dest_rank, xyz, world_size)

    recv_xyz = _distribute_tensor_by_rank(packed_xyz, send_counts, recv_counts)

    ctx.save_for_backward(
        unsort_perm,
        send_counts,
        recv_counts,
        recv_xyz,
    )
    ctx.X_local = X_local
    ctx.Y = Y
    ctx.Z = Z


def _backward(ctx, grad_out):
    unsort_perm, send_counts, recv_counts, recv_xyz = ctx.saved_tensors
    X_local, Y, Z = ctx.X_local, ctx.Y, ctx.Z
    C = grad_out.shape[-1]

    grad_recv_feat = _local_gather_grad(grad_out, recv_xyz)

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
    )

    grad_feat = grad_packed.index_select(0, unsort_perm)

    return grad_feat, None, None, None, None, None, None


encoder_scatter_to_voxel.register_fake(_fake)
encoder_scatter_to_voxel.register_autograd(_backward, setup_context=_setup_context)
