"""
Helper for testing distributed PyTorch operations in pytest.

Launches a function across N worker processes using the gloo backend,
collects per-rank return values back to the main process, and re-raises worker
exceptions with their original tracebacks so pytest reports them cleanly.

The gloo backend is used exclusively because it supports both CPU and CUDA
tensors and — unlike nccl — allows multiple ranks to share a single GPU,
which is the common case in development and CI environments.
"""

from __future__ import annotations

import math
import os
import socket
import sys
import traceback
from contextlib import closing
from dataclasses import dataclass
from typing import Any, Callable, Sequence, TypeGuard, cast

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class WorkerError:
    rank: int
    exc_type: str
    message: str
    tb: str

    def __str__(self) -> str:
        return (
            f"\n[rank {self.rank}] {self.exc_type}: {self.message}\n"
            f"--- remote traceback ---\n{self.tb}"
        )


class DistributedTestError(RuntimeError):
    """Error raised in the parent when one or more workers fail.

    Args:
        errors: Worker errors collected from child processes.
    """

    def __init__(self, errors: list[WorkerError]):
        self.errors = errors
        joined = "\n".join(str(e) for e in errors)
        super().__init__(f"{len(errors)} worker(s) failed:{joined}")


def _worker_entry(
    rank: int,
    world_size: int,
    master_port: int,
    result_queue: mp.Queue,
    done_event: Any,
    done_timeout: float,
    fn: Callable[..., Any],
    args: tuple,
    kwargs: dict,
) -> None:
    """Run one distributed test worker.

    Args:
        rank: Rank assigned to this worker.
        world_size: Number of worker processes in the group.
        master_port: Localhost port used for process-group initialization.
        result_queue: Queue used to return success or failure payloads.
        done_event: Event set by the parent after result deserialization.
        done_timeout: Maximum seconds to keep worker tensor FD servers alive.
        fn: Per-rank function to execute.
        args: Positional arguments forwarded to ``fn``.
        kwargs: Keyword arguments forwarded to ``fn``.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    try:
        dist.init_process_group(
            backend="gloo",
            rank=rank,
            world_size=world_size,
            init_method=f"tcp://127.0.0.1:{master_port}",
        )

        try:
            result = fn(rank, world_size, *args, **kwargs)
            # Move any tensors to CPU before sending across the queue
            result = _to_cpu(result)
            result_queue.put((rank, "ok", result))
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

    except BaseException as e:  # noqa: BLE001 - we really do want everything
        err = WorkerError(
            rank=rank,
            exc_type=type(e).__name__,
            message=str(e),
            tb=traceback.format_exc(),
        )
        result_queue.put((rank, "err", err))

    # Keep this process alive until the parent has finished deserializing all
    # results from the queue.  torch.multiprocessing shares tensor storage via
    # a per-process FD server; if we exit before the parent reads our tensors,
    # the server socket vanishes and deserialization fails with EOFError.
    done_event.wait(timeout=done_timeout)


def _to_cpu(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = type(obj)
        return t(_to_cpu(v) for v in obj)
    return obj


def run_distributed(
    fn: Callable[..., Any],
    world_size: int,
    *args,
    timeout: float = 60.0,
    **kwargs,
) -> list[Any]:
    """Launch a function across distributed worker processes.

    ``fn`` must have the signature ``fn(rank, world_size, *args, **kwargs)``
    and is responsible only for its distributed logic — process group
    initialization and teardown are handled here.

    ``fn`` also needs to be importable by qualified name (i.e. defined at
    module scope, not nested inside a test function) because the 'spawn'
    start method re-imports it in each child process.

    The gloo backend is always used because it supports both CPU and CUDA
    tensors and allows multiple ranks to share a single GPU (nccl does not).
    Test functions that want GPU tensors can simply create them on ``"cuda"``
    — gloo will handle the collectives.

    Args:
        fn: Per-rank function to run.
        world_size: Number of worker processes to spawn.
        *args: Positional arguments forwarded to ``fn``.
        timeout: Seconds to wait for all workers to finish before giving up.
        **kwargs: Keyword arguments forwarded to ``fn``.

    Returns:
        Results in rank order. Tensors are returned on CPU.

    Raises:
        TimeoutError: If one or more workers do not report before ``timeout``.
        DistributedTestError: If one or more workers fail.
    """
    master_port = _find_free_port()

    # 'spawn' is required for CUDA and safest for gloo too.
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    # signalled after the parent has finished deserializing all results,
    # so workers keep their FD servers alive until then.
    done_event = ctx.Event()

    procs: list[Any] = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_worker_entry,
            args=(
                rank,
                world_size,
                master_port,
                result_queue,
                done_event,
                timeout,
                fn,
                args,
                kwargs,
            ),
            daemon=False,
        )
        p.start()
        procs.append(p)

    results: dict[int, Any] = {}
    errors: list[WorkerError] = []
    timed_out = False

    from queue import Empty

    for _ in range(world_size):
        try:
            rank, status, payload = result_queue.get(timeout=timeout)
        except Empty:
            timed_out = True
            break
        if status == "ok":
            results[rank] = payload
        else:
            errors.append(payload)

    # all results deserialized — workers can now exit safely
    done_event.set()
    if timed_out:
        for p in procs:
            if p.is_alive():
                p.terminate()
    for p in procs:
        p.join(timeout=5.0)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5.0)

    # surface any non-zero exits that didn't report through the queue
    # (e.g. segfault, OOM-kill).
    for rank, p in enumerate(procs):
        if (
            p.exitcode not in (0, None)
            and rank not in results
            and not any(e.rank == rank for e in errors)
        ):
            errors.append(
                WorkerError(
                    rank=rank,
                    exc_type="ProcessExit",
                    message=f"exit code {p.exitcode}",
                    tb="(no traceback — process died before reporting)",
                )
            )

    if timed_out:
        missing = [
            r
            for r in range(world_size)
            if r not in results and not any(e.rank == r for e in errors)
        ]
        raise TimeoutError(
            f"Distributed test timed out after {timeout}s; ranks that never reported: {missing}"
        )

    if errors:
        raise DistributedTestError(sorted(errors, key=lambda e: e.rank))

    return [results[r] for r in range(world_size)]


def distributed(
    world_size: int | Sequence[int] | None = None,
    *,
    mesh_shape: tuple[int, ...] | Sequence[tuple[int, ...]] | None = None,
    device: str | Sequence[str] | None = None,
    timeout: float = 60.0,
) -> Callable:
    """Pytest decorator that runs the decorated function on each rank of a spawned group.

    The decorated test becomes a per-rank worker with signature
    ``(rank, world_size[, device])``. Assertion failures inside any rank are
    surfaced as a ``DistributedTestError`` with the rank's traceback.

    Args:
        world_size: Single int or a sequence. A sequence parametrizes the test
            across world sizes. Pass exactly one of ``world_size`` or
            ``mesh_shape``.
        mesh_shape: Single mesh shape or a sequence of mesh shapes. When
            provided, ``world_size`` is derived from ``math.prod(mesh_shape)``
            and ``mesh_shape`` is forwarded to the worker as a keyword argument.
        device: Optional device name or sequence of device names. ``"cuda"``
            entries auto-skip when CUDA is unavailable. When provided, the
            worker receives ``device`` as a keyword argument.
        timeout: Per-test wall-clock timeout forwarded to ``run_distributed``.

    Returns:
        Decorator that wraps a per-rank test function.

    Raises:
        ValueError: If both or neither of ``world_size`` and ``mesh_shape`` are
            provided, or if any provided size is invalid.
    """
    if (world_size is None) == (mesh_shape is None):
        raise ValueError("pass exactly one of world_size or mesh_shape")

    if mesh_shape is None:
        assert world_size is not None
        if isinstance(world_size, int):
            parametrize_ws = False
            ws_list = [world_size]
        else:
            parametrize_ws = True
            ws_list = list(world_size)

        for ws in ws_list:
            if not isinstance(ws, int) or ws <= 0:
                raise ValueError(f"world_size values must be positive ints, got {ws!r}")
        mesh_cases = None
    else:
        if _is_mesh_shape(mesh_shape):
            mesh_shapes = [mesh_shape]
            parametrize_ws = False
        else:
            mesh_shapes = list(cast(Sequence[tuple[int, ...]], mesh_shape))
            parametrize_ws = True

        mesh_cases = []
        for shape in mesh_shapes:
            if not _is_mesh_shape(shape):
                raise ValueError(
                    f"mesh_shape values must be non-empty tuples of ints, got {shape!r}"
                )
            if any(dim <= 0 for dim in shape):
                raise ValueError(f"mesh_shape dimensions must be positive, got {shape!r}")
            mesh_cases.append((math.prod(shape), shape))
        ws_list = [ws for ws, _shape in mesh_cases]

    if device is None:
        dev_params = None
    else:
        dev_list = list(device) if isinstance(device, (list, tuple)) else [device]
        dev_params = [
            pytest.param(
                d,
                marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
            )
            if d == "cuda"
            else d
            for d in dev_list
        ]

    def deco(fn: Callable) -> Callable:
        # Spawn pickles fn by (module, qualname). After decoration, the module
        # name binds to the wrapper, so stash the original under a mangled
        # attribute so pickle can still find it in the child.
        original_name = fn.__name__
        mod = sys.modules.get(fn.__module__)
        if mod is not None:
            stash_name = f"_distributed_inner__{fn.__qualname__.replace('.', '_')}"
            setattr(mod, stash_name, fn)
            fn.__qualname__ = stash_name
            fn.__name__ = stash_name

        if dev_params is None:
            if mesh_cases is not None and parametrize_ws:

                def wrapper(world_size, mesh_shape):
                    run_distributed(fn, world_size, timeout=timeout, mesh_shape=mesh_shape)

                wrapper = pytest.mark.parametrize(("world_size", "mesh_shape"), mesh_cases)(wrapper)
            elif mesh_cases is not None:
                ws, shape = mesh_cases[0]

                def wrapper():
                    run_distributed(fn, ws, timeout=timeout, mesh_shape=shape)
            elif parametrize_ws:

                def wrapper(world_size):
                    run_distributed(fn, world_size, timeout=timeout)

                wrapper = pytest.mark.parametrize("world_size", ws_list)(wrapper)
            else:
                ws = ws_list[0]

                def wrapper():
                    run_distributed(fn, ws, timeout=timeout)
        else:
            if mesh_cases is not None and parametrize_ws:

                def wrapper(world_size, mesh_shape, device):
                    run_distributed(
                        fn,
                        world_size,
                        timeout=timeout,
                        mesh_shape=mesh_shape,
                        device=device,
                    )

                wrapper = pytest.mark.parametrize("device", dev_params)(wrapper)
                wrapper = pytest.mark.parametrize(("world_size", "mesh_shape"), mesh_cases)(wrapper)
            elif mesh_cases is not None:
                ws, shape = mesh_cases[0]

                def wrapper(device):
                    run_distributed(fn, ws, timeout=timeout, mesh_shape=shape, device=device)
            elif parametrize_ws:

                def wrapper(world_size, device):
                    run_distributed(fn, world_size, timeout=timeout, device=device)

                wrapper = pytest.mark.parametrize("device", dev_params)(wrapper)
                wrapper = pytest.mark.parametrize("world_size", ws_list)(wrapper)
            else:
                ws = ws_list[0]

                def wrapper(device):
                    run_distributed(fn, ws, timeout=timeout, device=device)

                wrapper = pytest.mark.parametrize("device", dev_params)(wrapper)

        wrapper.__name__ = original_name
        wrapper.__qualname__ = original_name
        wrapper.__doc__ = fn.__doc__
        return wrapper

    return deco


def _is_mesh_shape(value: object) -> TypeGuard[tuple[int, ...]]:
    return isinstance(value, tuple) and len(value) > 0 and all(isinstance(v, int) for v in value)


def gather_per_rank(
    results: list[dict[str, Any]],
    key: str,
    dim: int = 0,
) -> torch.Tensor:
    """Concatenate per-rank tensors.

    Args:
        results: Per-rank result dictionaries.
        key: Key to read from each rank's result dictionary.
        dim: Dimension along which tensors are concatenated.

    Returns:
        Concatenated tensor.
    """
    return torch.cat([r[key] for r in results], dim=dim)


def assert_close_per_rank(
    results: list[Any],
    expected: list[Any],
    key: str | None = None,
    **kwargs: Any,
) -> None:
    """Assert each rank's result matches the corresponding entry of ``expected``.

    Args:
        results: Per-rank result objects.
        expected: Per-rank expected objects.
        key: Optional key to read from each rank's result before comparison.
        **kwargs: Extra keyword arguments forwarded to
            ``torch.testing.assert_close``.
    """
    for r, got in enumerate(results):
        if key is not None:
            got = got[key]
        torch.testing.assert_close(
            got,
            expected[r],
            msg=lambda m, r=r: f"rank {r}: {m}",
            **kwargs,
        )
