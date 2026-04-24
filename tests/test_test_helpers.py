"""
Example pytest tests using the distributed helper.

Run with:   pytest -xvs test_distributed_example.py
"""

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from bigconv.testing import DistributedTestError, distributed, run_distributed


# ---------------------------------------------------------------------------
# Simple per-rank collective tests using the @distributed decorator.
# ---------------------------------------------------------------------------


@distributed(world_size=2)
def test_all_reduce_sum(rank, world_size):
    t = torch.tensor([float(rank + 1)])
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    # every rank should see 1+2 = 3
    assert torch.equal(t, torch.tensor([3.0])), f"rank {rank} got {t}"


@distributed(world_size=3)
def test_all_gather(rank, world_size):
    t = torch.tensor([float(rank)])
    gathered = [torch.zeros(1) for _ in range(world_size)]
    dist.all_gather(gathered, t)
    expected = [torch.tensor([float(i)]) for i in range(world_size)]
    for got, exp in zip(gathered, expected):
        assert torch.equal(got, exp)


@distributed(world_size=[1, 2])
def test_world_size_list_parametrizes(rank, world_size):
    t = torch.tensor([float(rank + 1)])
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = float(world_size * (world_size + 1) // 2)
    assert torch.equal(t, torch.tensor([expected]))


@distributed(mesh_shape=(2, 2))
def test_mesh_shape_derives_world_size(rank, world_size, mesh_shape):
    assert mesh_shape == (2, 2)
    assert world_size == 4
    t = torch.tensor([1.0])
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    assert torch.equal(t, torch.tensor([4.0]))


@distributed(mesh_shape=[(1,), (2,)])
def test_mesh_shape_list_parametrizes(rank, world_size, mesh_shape):
    assert world_size == mesh_shape[0]
    t = torch.tensor([1.0])
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    assert torch.equal(t, torch.tensor([float(world_size)]))


def test_distributed_rejects_invalid_topology_args():
    with pytest.raises(ValueError, match="exactly one"):
        distributed()

    with pytest.raises(ValueError, match="exactly one"):
        distributed(world_size=2, mesh_shape=(2,))

    with pytest.raises(ValueError, match="world_size values must be positive"):
        distributed(world_size=[1, 0])

    with pytest.raises(ValueError, match="mesh_shape dimensions must be positive"):
        distributed(mesh_shape=(2, 0))


@distributed(world_size=2)
def test_broadcast(rank, world_size):
    if rank == 0:
        t = torch.tensor([42.0, 43.0])
    else:
        t = torch.zeros(2)
    dist.broadcast(t, src=0)
    assert torch.equal(t, torch.tensor([42.0, 43.0]))


# ---------------------------------------------------------------------------
# Testing a distributed module: DDP gradient averaging.
# This one still needs main-process comparison against a single-process
# reference, so it uses run_distributed directly.
# ---------------------------------------------------------------------------


def _ddp_grad_matches_single(rank, world_size):
    """Each rank runs forward/backward on its shard; DDP should average grads."""
    from torch.nn.parallel import DistributedDataParallel as DDP

    torch.manual_seed(0)
    model = nn.Linear(4, 1, bias=False)
    ddp_model = DDP(model)

    # per-rank input
    x = torch.full((2, 4), float(rank + 1))
    y = ddp_model(x).sum()
    y.backward()
    assert model.weight.grad is not None

    return {
        "weight": model.weight.detach().clone(),
        "grad": model.weight.grad.detach().clone(),
    }


def test_ddp_grads_are_averaged():
    world_size = 2
    results = run_distributed(_ddp_grad_matches_single, world_size=world_size)

    # DDP averages grads across ranks, so every rank should see the same grad.
    g0 = results[0]["grad"]
    for r in results[1:]:
        assert torch.allclose(r["grad"], g0)

    # Compare to what a single-process run would produce with the concatenated
    # batch (the "equivalent" non-distributed computation).
    torch.manual_seed(0)
    ref = nn.Linear(4, 1, bias=False)
    x_full = torch.cat([torch.full((2, 4), float(r + 1)) for r in range(world_size)])
    ref(x_full).sum().backward()
    assert ref.weight.grad is not None
    # DDP averages, plain backward sums -> divide by world_size.
    assert torch.allclose(results[0]["grad"], ref.weight.grad / world_size)


# ---------------------------------------------------------------------------
# Verifying that worker exceptions propagate with a useful traceback.
# ---------------------------------------------------------------------------


def _rank1_raises(rank, world_size):
    if rank == 1:
        raise ValueError("boom from rank 1")
    # other ranks would normally participate in collectives; for this test
    # we just return immediately so the parent doesn't hang.
    return rank


def test_worker_exception_is_raised():
    with pytest.raises(DistributedTestError) as excinfo:
        run_distributed(_rank1_raises, world_size=2, timeout=10)
    # the original error info should be attached
    assert any(e.rank == 1 and "boom" in e.message for e in excinfo.value.errors)


# ---------------------------------------------------------------------------
# CUDA tests — gloo supports CUDA tensors and allows multiple ranks to share
# a single GPU (nccl does not).
# ---------------------------------------------------------------------------


@distributed(world_size=2, device=["cpu", "cuda"])
def test_reduce_scatter(rank, world_size, device):
    # Each rank contributes a world_size-sized tensor; reduce_scatter sums
    # them element-wise and scatters the result across ranks.
    inputs = [torch.tensor([float(rank * 10 + i)], device=device) for i in range(world_size)]
    output = torch.zeros(1, device=device)
    dist.reduce_scatter(output, inputs, op=dist.ReduceOp.SUM)
    # rank r receives the r-th column summed across ranks:
    # col 0: 0 + 10 = 10;  col 1: 1 + 11 = 12
    expected = torch.tensor([10.0, 12.0], device=device)[rank : rank + 1]
    assert torch.equal(output, expected), f"rank {rank}: {output} vs {expected}"


@distributed(world_size=2, device="cuda")
def test_all_reduce_sum_cuda(rank, world_size, device):
    t = torch.tensor([float(rank + 1)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    assert torch.equal(t.cpu(), torch.tensor([3.0]))
