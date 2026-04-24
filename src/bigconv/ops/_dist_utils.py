"""Shared distribution plumbing for ops in ``bigconv.ops``.

Ops here take a ``DeviceMesh`` in the public API but register as
``torch.library.custom_op`` internally — and custom-op schemas don't accept
``ProcessGroup`` or ``DeviceMesh`` (both are torchbind). So we convert at the
wrapper boundary: ``mesh`` → ``group_name`` (str) going in, ``group_name`` →
``ProcessGroup`` going out.
"""

from __future__ import annotations

from typing import Any, cast

import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh


def resolve_group(group_name: str) -> ProcessGroup | None:
    """Look up a process group by its registered name.

    An empty string maps to ``None`` (default / WORLD in collectives).
    Uses ``dist.distributed_c10d._resolve_process_group``, which is a private
    API but the only way to reverse ``ProcessGroup.group_name``. PyTorch's own
    functional collectives do the same thing.

    Args:
        group_name: Registered process-group name, or an empty string for the
            default group.

    Returns:
        Resolved process group, or ``None`` for the default group.
    """
    if not group_name:
        return None
    resolver = cast(Any, dist.distributed_c10d._resolve_process_group)
    return cast(ProcessGroup, resolver(group_name))


def group_name_from_mesh(mesh: DeviceMesh | None) -> str:
    """Extract a custom-op-safe process-group name from a mesh.

    Args:
        mesh: Optional 1D device mesh.

    Returns:
        Process-group name for custom-op schemas, or an empty string for the
        default group.

    Raises:
        ValueError: If ``mesh`` is not 1D.
    """
    if mesh is None:
        return ""
    if mesh.ndim != 1:
        raise ValueError(
            f"expected a 1D DeviceMesh (one sharded axis); got {mesh.ndim}D "
            f"mesh with shape {tuple(mesh.mesh.shape)}. "
            f"Pass a sliced sub-mesh like mesh['x']."
        )
    pg = mesh.get_group()
    if pg is None or pg is dist.group.WORLD:
        return ""
    return pg.group_name
