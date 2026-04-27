import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from bigconv.ops.encoder_distribute import encoder_scatter_to_voxel
from bigconv.testing import (
    assert_close_per_rank,
    distributed,
    gather_per_rank,
    run_distributed,
)


def _valid_scatter_inputs(world_size):
    feat = torch.ones((2, 3), requires_grad=True)
    dest_rank = torch.tensor([0, world_size - 1], dtype=torch.int64)
    xyz = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.int64)
    return feat, dest_rank, xyz


# ---------------------------------------------------------------------------
# Test 1: analytical forward + backward on a tiny (1,1,1) voxel grid.
#
# 4 ranks, each with 1 point (C=1). Each rank sends to its next neighbor
# (round-robin), placing value rank+1 into voxel (0,0,0).
#
# After scatter the voxel values are [4, 1, 2, 3] on ranks [0, 1, 2, 3].
# Loss = product of all voxels = 24.
# ---------------------------------------------------------------------------


@distributed(world_size=4)
def test_scatter_to_voxel_analytical(rank, world_size):
    feat = torch.tensor([[float(rank + 1)]], requires_grad=True)  # (1, 1)
    dest_rank = torch.tensor([(rank + 1) % world_size], dtype=torch.int32)
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

    # --- forward checks ---
    # Rank r receives from rank (r-1+W)%W whose feature is (r-1+W)%W + 1.
    expected_voxels = {0: 4.0, 1: 1.0, 2: 2.0, 3: 3.0}
    assert voxel.item() == expected_voxels[rank]
    assert loss.item() == 24.0

    # --- backward checks ---
    # d(loss)/d(feat_r) = d(loss)/d(voxel at dest rank) = loss / voxel_dest.
    expected_grad = 24.0 / expected_voxels[(rank + 1) % world_size]
    assert feat.grad is not None
    assert feat.grad.item() == expected_grad


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
        grid = torch.zeros(C, X_LOCAL * Y * Z)
        lin = ((c[:, 0] * Y) + c[:, 1]) * Z + c[:, 2]
        grid.index_add_(1, lin, f.transpose(0, 1))
        grids.append(grid.view(C, X_LOCAL, Y, Z))
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
    all_xyz = torch.stack(
        [
            torch.randint(0, X_LOCAL, (N_PER_RANK * world_size,)),
            torch.randint(0, Y, (N_PER_RANK * world_size,)),
            torch.randint(0, Z, (N_PER_RANK * world_size,)),
        ],
        dim=1,
    )

    # --- local reference ---
    ref_feat = all_feat.clone().requires_grad_(True)
    ref_grids = _reference_scatter(ref_feat, all_dest, all_xyz, world_size)
    ref_loss = torch.stack([g.sum() for g in ref_grids]).sum()
    ref_loss.backward()

    # --- distributed ---
    results = run_distributed(
        _large_scale_worker,
        world_size=world_size,
        all_feat=all_feat,
        all_dest=all_dest,
        all_xyz=all_xyz,
    )

    # Compare voxel grids per rank.
    assert_close_per_rank(results, ref_grids, key="voxel")

    # Compare gradients: gather per-rank grads back into the original order.
    dist_grad = gather_per_rank(results, "feat_grad")
    torch.testing.assert_close(dist_grad, ref_feat.grad)


# ---------------------------------------------------------------------------
# Test 3: torch.library.opcheck — validates fake impl and autograd registration
# for the custom op. Runs inside a distributed worker because opcheck calls the
# op, which requires a live process group.
#
# The other opcheck utilities are skipped because of torch-internal limitations:
#   - test_schema: schema_check_mode walks args of the underlying
#     alltoall_base and trips on ProcessGroup (no __len__ on torchbind).
#   - test_aot_dispatch_{static,dynamic}: torch.compile the op, which
#     doesn't play well with in-flight collectives.
# ---------------------------------------------------------------------------


@distributed(world_size=2)
def test_encoder_scatter_to_voxel_opcheck(rank, world_size):
    torch.manual_seed(rank)
    N, C_op = 8, 4
    X_local, Y_op, Z_op = 2, 2, 2

    feat = torch.randn(N, C_op, requires_grad=True)
    dest_rank = torch.randint(0, world_size, (N,), dtype=torch.int32)
    xyz = torch.stack(
        [
            torch.randint(0, X_local, (N,)),
            torch.randint(0, Y_op, (N,)),
            torch.randint(0, Z_op, (N,)),
        ],
        dim=1,
    ).to(torch.int64)

    torch.library.opcheck(
        torch.ops.bigconv.encoder_scatter_to_voxel.default,
        (feat, dest_rank, xyz, X_local, Y_op, Z_op, ""),
        test_utils=("test_faketensor", "test_autograd_registration"),
    )


# ---------------------------------------------------------------------------
# Test 4: DeviceMesh plumbing — passing an explicit 1D mesh should produce the
# same result as the WORLD default, and a multi-dim mesh should raise.
# ---------------------------------------------------------------------------


@distributed(world_size=4)
def test_scatter_with_explicit_1d_mesh(rank, world_size):
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("x",))
    feat = torch.tensor([[float(rank + 1)]], requires_grad=True)
    dest_rank = torch.tensor([(rank + 1) % world_size], dtype=torch.int32)
    xyz = torch.tensor([[0, 0, 0]], dtype=torch.int64)

    voxel = encoder_scatter_to_voxel(feat, dest_rank, xyz, 1, 1, 1, mesh=mesh)

    # rank r receives from (r-1+W)%W whose feature is (r-1+W)%W + 1
    expected = {0: 4.0, 1: 1.0, 2: 2.0, 3: 3.0}[rank]
    assert voxel.item() == expected


@distributed(world_size=4)
def test_scatter_rejects_multi_dim_mesh(rank, world_size):
    mesh_2d = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "sp"))
    feat = torch.zeros((1, 1), requires_grad=True)
    dest_rank = torch.tensor([0], dtype=torch.int32)
    xyz = torch.tensor([[0, 0, 0]], dtype=torch.int64)

    with pytest.raises(ValueError, match="1D DeviceMesh"):
        encoder_scatter_to_voxel(feat, dest_rank, xyz, 1, 1, 1, mesh=mesh_2d)

    # Sliced sub-mesh should be accepted.
    sub = mesh_2d["sp"]
    encoder_scatter_to_voxel(feat, dest_rank, xyz, 1, 1, 1, mesh=sub)


# ---------------------------------------------------------------------------
# Input validation — invalid local inputs should fail before collectives.
# ---------------------------------------------------------------------------


@distributed(world_size=2)
def test_scatter_validation_rejects_bad_feat_shape(rank, world_size):
    _feat, dest_rank, xyz = _valid_scatter_inputs(world_size)
    with pytest.raises(ValueError, match="feat must be 2D"):
        encoder_scatter_to_voxel(torch.ones((2, 3, 1)), dest_rank, xyz, 2, 2, 2)


@distributed(world_size=2)
def test_scatter_validation_rejects_bad_dest_rank_shape(rank, world_size):
    feat, _dest_rank, xyz = _valid_scatter_inputs(world_size)
    with pytest.raises(ValueError, match="dest_rank must be 1D"):
        encoder_scatter_to_voxel(feat, torch.zeros((2, 1), dtype=torch.int64), xyz, 2, 2, 2)


@distributed(world_size=2)
def test_scatter_validation_rejects_mismatched_lengths(rank, world_size):
    feat, _dest_rank, xyz = _valid_scatter_inputs(world_size)
    with pytest.raises(ValueError, match="dest_rank length=1 must match feat N=2"):
        encoder_scatter_to_voxel(feat, torch.zeros((1,), dtype=torch.int64), xyz, 2, 2, 2)

    with pytest.raises(ValueError, match="xyz length=1 must match feat N=2"):
        encoder_scatter_to_voxel(
            feat,
            torch.zeros((2,), dtype=torch.int64),
            torch.zeros((1, 3), dtype=torch.int64),
            2,
            2,
            2,
        )


@distributed(world_size=2)
def test_scatter_validation_rejects_bad_xyz_shape(rank, world_size):
    feat, dest_rank, _xyz = _valid_scatter_inputs(world_size)
    with pytest.raises(ValueError, match=r"xyz must have shape \(N, 3\)"):
        encoder_scatter_to_voxel(feat, dest_rank, torch.zeros((2, 2), dtype=torch.int64), 2, 2, 2)


@distributed(world_size=2)
def test_scatter_validation_rejects_non_integer_indices(rank, world_size):
    feat, dest_rank, xyz = _valid_scatter_inputs(world_size)
    with pytest.raises(TypeError, match="dest_rank must have an integer dtype"):
        encoder_scatter_to_voxel(feat, dest_rank.to(torch.float32), xyz, 2, 2, 2)

    with pytest.raises(TypeError, match="xyz must have an integer dtype"):
        encoder_scatter_to_voxel(feat, dest_rank, xyz.to(torch.float32), 2, 2, 2)


@distributed(world_size=2)
def test_scatter_validation_rejects_bad_grid_sizes(rank, world_size):
    feat, dest_rank, xyz = _valid_scatter_inputs(world_size)
    with pytest.raises(ValueError, match="X_local must be positive"):
        encoder_scatter_to_voxel(feat, dest_rank, xyz, 0, 2, 2)

    with pytest.raises(TypeError, match="Y must be an int"):
        encoder_scatter_to_voxel(feat, dest_rank, xyz, 2, 2.0, 2)  # pyrefly: ignore[bad-argument-type]


@distributed(world_size=2)
def test_scatter_validation_rejects_dest_rank_out_of_range(rank, world_size):
    feat, _dest_rank, xyz = _valid_scatter_inputs(world_size)
    with pytest.raises(ValueError, match=r"dest_rank values must be in \[0, 2\)"):
        encoder_scatter_to_voxel(feat, torch.tensor([0, 2], dtype=torch.int64), xyz, 2, 2, 2)

    with pytest.raises(ValueError, match=r"dest_rank values must be in \[0, 2\)"):
        encoder_scatter_to_voxel(feat, torch.tensor([-1, 0], dtype=torch.int64), xyz, 2, 2, 2)


@distributed(world_size=2)
def test_scatter_validation_rejects_xyz_out_of_range(rank, world_size):
    feat, dest_rank, _xyz = _valid_scatter_inputs(world_size)
    with pytest.raises(ValueError, match=r"xyz x coordinates must be in \[0, 2\)"):
        encoder_scatter_to_voxel(
            feat,
            dest_rank,
            torch.tensor([[0, 0, 0], [2, 1, 1]], dtype=torch.int64),
            2,
            2,
            2,
        )

    with pytest.raises(ValueError, match=r"xyz z coordinates must be in \[0, 2\)"):
        encoder_scatter_to_voxel(
            feat,
            dest_rank,
            torch.tensor([[0, 0, 0], [1, 1, -1]], dtype=torch.int64),
            2,
            2,
            2,
        )


@distributed(world_size=2)
def test_scatter_validation_accepts_empty_inputs(rank, world_size):
    feat = torch.empty((0, 3), requires_grad=True)
    dest_rank = torch.empty((0,), dtype=torch.int64)
    xyz = torch.empty((0, 3), dtype=torch.int64)

    voxel = encoder_scatter_to_voxel(feat, dest_rank, xyz, 2, 2, 2)
    assert voxel.shape == (3, 2, 2, 2)
    assert torch.all(voxel == 0)
