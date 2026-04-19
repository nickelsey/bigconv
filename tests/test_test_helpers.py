"""
Example pytest tests using the distributed helper.

Run with:   pytest -xvs test_distributed_example.py
"""

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from bigconv.testing import DistributedTestError, run_distributed


# ---------------------------------------------------------------------------
# Style 1: call run_distributed() directly from the test.
# Best when you want per-test control over world_size or backend.
# ---------------------------------------------------------------------------


def _all_reduce_sum(rank, world_size):
    t = torch.tensor([float(rank + 1)])
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def test_all_reduce_sum():
    world_size = 2
    results = run_distributed(_all_reduce_sum, world_size=world_size)
    # every rank should see 1+2 = 3
    expected = torch.tensor([3.0])
    assert len(results) == world_size
    for r, out in enumerate(results):
        assert torch.equal(out, expected), f"rank {r} got {out}"


def _all_gather(rank, world_size):
    t = torch.tensor([float(rank)])
    gathered = [torch.zeros(1) for _ in range(world_size)]
    dist.all_gather(gathered, t)
    return gathered


def test_all_gather():
    world_size = 3
    results = run_distributed(_all_gather, world_size=world_size)
    expected = [torch.tensor([float(i)]) for i in range(world_size)]
    for rank_out in results:
        for got, exp in zip(rank_out, expected):
            assert torch.equal(got, exp)


# ---------------------------------------------------------------------------
# Broadcast: another simple collective.
# ---------------------------------------------------------------------------


def _broadcast_from_rank0(rank, world_size):
    if rank == 0:
        t = torch.tensor([42.0, 43.0])
    else:
        t = torch.zeros(2)
    dist.broadcast(t, src=0)
    return t


def test_broadcast():
    results = run_distributed(_broadcast_from_rank0, world_size=2)
    expected = torch.tensor([42.0, 43.0])
    for out in results:
        assert torch.equal(out, expected)


# ---------------------------------------------------------------------------
# Testing a distributed module: DDP gradient averaging.
# ---------------------------------------------------------------------------


def _ddp_grad_matches_single(rank, world_size):
    """Each rank runs forward/backward on its shard; DDP should average grads."""
    from torch.nn.parallel import DistributedDataParallel as DDP

    torch.manual_seed(0)
    model = nn.Linear(4, 1, bias=False)
    ddp_model = DDP(model)

    # Per-rank input
    x = torch.full((2, 4), float(rank + 1))
    y = ddp_model(x).sum()
    y.backward()

    return {
        "weight": model.weight.detach().clone(),
        "grad": model.weight.grad.detach().clone(),
    }


def test_ddp_grads_are_averaged():
    world_size = 2
    results = run_distributed(
        _ddp_grad_matches_single, world_size=world_size
    )

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


def _reduce_scatter_fn(rank, world_size, device):
    # Each rank contributes a world_size-sized tensor; reduce_scatter sums
    # them element-wise and scatters the result across ranks.
    inputs = [
        torch.tensor([float(rank * 10 + i)], device=device) for i in range(world_size)
    ]
    output = torch.zeros(1, device=device)
    dist.reduce_scatter(output, inputs, op=dist.ReduceOp.SUM)
    return output


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"))])
def test_reduce_scatter(device):
    world_size = 2
    results = run_distributed(
        _reduce_scatter_fn,
        world_size=world_size,
        device=device,
    )
    # rank r receives the r-th column summed across ranks:
    # col 0: 0 + 10 = 10;  col 1: 1 + 11 = 12
    expected = [torch.tensor([10.0]), torch.tensor([12.0])]
    for r, out in enumerate(results):
        assert torch.equal(out, expected[r]), f"rank {r}: {out} vs {expected[r]}"


def _all_reduce_sum_cuda(rank, world_size):
    t = torch.tensor([float(rank + 1)], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_all_reduce_sum_cuda():
    results = run_distributed(_all_reduce_sum_cuda, world_size=2)
    expected = torch.tensor([3.0])  # 1 + 2
    for out in results:
        assert torch.equal(out, expected)
