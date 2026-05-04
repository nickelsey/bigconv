from __future__ import annotations

import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh


def sync_module_parameters(module: nn.Module, mesh: DeviceMesh) -> None:
    """Broadcast module parameters and buffers from mesh rank 0.

    This is intended for single distributed replicas that use BigConv ops
    directly without wrapping the model in ``DistributedDataParallel``. The
    distributed ops assume replicated weights are identical on every rank; this
    helper makes that invariant explicit after local module construction.

    Args:
        module: Module whose parameters and buffers should be synchronized.
        mesh: 1D device mesh defining the ranks that share replicated module
            state. Source is rank 0 within this mesh, not necessarily global
            rank 0.

    Raises:
        RuntimeError: If torch distributed is not initialized.
        ValueError: If ``mesh`` is not 1D.
    """
    if not dist.is_initialized():
        raise RuntimeError("sync_module_parameters requires torch.distributed to be initialized")
    if mesh.ndim != 1:
        raise ValueError(
            f"sync_module_parameters expects a 1D DeviceMesh; got {mesh.ndim}D mesh "
            f"with shape {tuple(mesh.mesh.shape)}"
        )

    group = mesh.get_group()
    src = dist.get_global_rank(group, 0)

    for parameter in module.parameters():
        dist.broadcast(parameter.data, src=src, group=group)
    for buffer in module.buffers():
        dist.broadcast(buffer.data, src=src, group=group)
