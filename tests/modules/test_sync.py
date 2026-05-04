import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from bigconv.distributed import sync_module_parameters
from bigconv.testing import run_distributed


class _ModuleWithBuffer(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.register_buffer("running", torch.full((2,), value))


def _sync_module_parameters_worker(rank, world_size):
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("x",))
    module = _ModuleWithBuffer(float(rank))

    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(rank + 1.0)

    sync_module_parameters(module, mesh)

    return {
        "weight": module.linear.weight.detach().clone(),
        "bias": module.linear.bias.detach().clone(),
        "running": module.get_buffer("running").detach().clone(),
    }


def test_sync_module_parameters_broadcasts_from_mesh_rank_zero():
    results = run_distributed(_sync_module_parameters_worker, world_size=3)

    for result in results:
        torch.testing.assert_close(result["weight"], torch.ones_like(result["weight"]))
        torch.testing.assert_close(result["bias"], torch.ones_like(result["bias"]))
        torch.testing.assert_close(result["running"], torch.zeros_like(result["running"]))


def _sync_module_parameters_rejects_multi_dim_mesh_worker(rank, world_size, shape):
    mesh = init_device_mesh("cpu", shape, mesh_dim_names=("x", "y"))
    module = nn.Linear(1, 1)

    with pytest.raises(ValueError, match="expects a 1D DeviceMesh"):
        sync_module_parameters(module, mesh)


def test_sync_module_parameters_rejects_multi_dim_mesh():
    run_distributed(
        _sync_module_parameters_rejects_multi_dim_mesh_worker,
        world_size=2,
        shape=(1, 2),
    )
