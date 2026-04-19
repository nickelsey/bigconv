"""Tests for bigconv.ops.encoder_distribute."""

import torch
import torch.distributed as dist

from bigconv.ops.encoder_distribute import encoder_scatter_to_voxel
from bigconv.testing import run_distributed


# ---------------------------------------------------------------------------
# Test 1: analytical forward + backward on a tiny (4,1,1) voxel grid.
#
# 4 ranks, each with 1 point (C=1). Each rank sends to its next neighbor
# (round-robin), placing value rank+1 into voxel (0,0,0).
#
# After scatter the voxel values are [4, 1, 2, 3] on ranks [0, 1, 2, 3].
# Loss = product of all voxels = 24.
# ---------------------------------------------------------------------------


def _analytical_worker(rank, world_size):
    feat = torch.tensor([[float(rank + 1)]], requires_grad=True)       # (1, 1)
    dest_rank = torch.tensor([(rank + 1) % world_size], dtype=torch.int32)  # next neighbor
    xyz = torch.tensor([[0, 0, 0]], dtype=torch.int64)

    voxel = encoder_scatter_to_voxel(feat, dest_rank, xyz, X_local=1, Y=1, Z=1)
    # voxel shape: (1, 1, 1, 1)

    # Gather all local voxels (detached) to compute the product of *other*
    # ranks' values; multiply by the local value to get a differentiable loss
    # whose gradient equals product-of-others.
    local_val = voxel.view(1)
    gathered = [torch.zeros(1) for _ in range(world_size)]
    dist.all_gather(gathered, local_val.detach())
    others_prod = torch.stack(gathered).prod() / gathered[rank]
    loss = (local_val * others_prod).squeeze()

    loss.backward()

    return {
        "voxel": voxel.detach(),
        "loss": loss.detach(),
        "feat_grad": feat.grad.detach(),
    }


def test_scatter_to_voxel_analytical():
    world_size = 4
    results = run_distributed(_analytical_worker, world_size=world_size)

    # --- forward checks ---
    # Rank r receives from rank (r-1+W)%W whose feature is (r-1+W)%W + 1.
    expected_voxels = {
        0: 4.0,  # from rank 3
        1: 1.0,  # from rank 0
        2: 2.0,  # from rank 1
        3: 3.0,  # from rank 2
    }
    for r in range(world_size):
        assert results[r]["voxel"].item() == expected_voxels[r], (
            f"rank {r}: expected voxel {expected_voxels[r]}, got {results[r]['voxel'].item()}"
        )
        assert results[r]["loss"].item() == 24.0

    # --- backward checks ---
    # d(loss)/d(feat_r) = d(loss)/d(voxel at dest rank) where dest = (r+1)%W.
    # d(loss)/d(voxel_s) = loss / voxel_s.
    # So feat_grad[r] = 24 / voxel_{(r+1)%W} = 24 / expected_voxels[(r+1)%W].
    expected_feat_grads = {
        0: 24.0 / expected_voxels[1],  # 24/1 = 24
        1: 24.0 / expected_voxels[2],  # 24/2 = 12
        2: 24.0 / expected_voxels[3],  # 24/3 = 8
        3: 24.0 / expected_voxels[0],  # 24/4 = 6
    }
    for r in range(world_size):
        assert results[r]["feat_grad"].item() == expected_feat_grads[r], (
            f"rank {r}: expected grad {expected_feat_grads[r]}, got {results[r]['feat_grad'].item()}"
        )


# ---------------------------------------------------------------------------
# Test 2: compare distributed scatter against a local single-process
# reference on a larger problem (world_size=2, 64 points/rank, C=8, 4x4x4).
# ---------------------------------------------------------------------------

N_PER_RANK = 64
C = 8
X_LOCAL, Y, Z = 4, 4, 4


def _reference_scatter(all_feat, all_dest, all_xyz, world_size):
    """Single-process reference: scatter all points into per-rank voxel grids."""
    grids = []
    for dest_r in range(world_size):
        mask = all_dest == dest_r
        f = all_feat[mask]
        c = all_xyz[mask]
        grid = torch.zeros(X_LOCAL * Y * Z, C)
        lin = ((c[:, 0] * Y) + c[:, 1]) * Z + c[:, 2]
        grid.index_add_(0, lin, f)
        grids.append(grid.view(X_LOCAL, Y, Z, C))
    return grids


def _large_scale_worker(rank, world_size, all_feat, all_dest, all_xyz):
    # Slice this rank's portion.
    start = rank * N_PER_RANK
    end = start + N_PER_RANK
    feat = all_feat[start:end].clone().requires_grad_(True)
    dest_rank = all_dest[start:end]
    xyz = all_xyz[start:end]

    voxel = encoder_scatter_to_voxel(feat, dest_rank, xyz, X_LOCAL, Y, Z)
    loss = voxel.sum()
    loss.backward()

    return {
        "voxel": voxel.detach(),
        "feat_grad": feat.grad.detach(),
    }


def test_scatter_to_voxel_vs_local_reference():
    world_size = 2
    torch.manual_seed(42)

    # Pre-generate all data so distributed and reference use the same inputs.
    all_feat = torch.randn(N_PER_RANK * world_size, C)
    all_dest = torch.randint(0, world_size, (N_PER_RANK * world_size,), dtype=torch.int32)
    all_xyz = torch.stack([
        torch.randint(0, X_LOCAL, (N_PER_RANK * world_size,)),
        torch.randint(0, Y, (N_PER_RANK * world_size,)),
        torch.randint(0, Z, (N_PER_RANK * world_size,)),
    ], dim=1)

    # --- local reference ---
    ref_feat = all_feat.clone().requires_grad_(True)
    ref_grids = _reference_scatter(ref_feat, all_dest, all_xyz, world_size)
    ref_loss = sum(g.sum() for g in ref_grids)
    ref_loss.backward()

    # --- distributed ---
    results = run_distributed(
        _large_scale_worker,
        world_size=world_size,
        all_feat=all_feat,
        all_dest=all_dest,
        all_xyz=all_xyz,
    )

    # Compare voxel grids.
    for r in range(world_size):
        torch.testing.assert_close(
            results[r]["voxel"], ref_grids[r],
            msg=lambda m: f"rank {r} voxel mismatch: {m}",
        )

    # Compare gradients: gather per-rank grads back into the original order.
    dist_grad = torch.cat([results[r]["feat_grad"] for r in range(world_size)], dim=0)
    torch.testing.assert_close(
        dist_grad, ref_feat.grad,
        msg=lambda m: f"feat grad mismatch: {m}",
    )
