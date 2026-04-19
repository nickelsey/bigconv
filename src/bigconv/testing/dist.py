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

import os
import socket
import traceback
from contextlib import closing
from dataclasses import dataclass
from typing import Any, Callable

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
    """Raised in the parent when one or more workers failed."""

    def __init__(self, errors: list[WorkerError]):
        self.errors = errors
        joined = "\n".join(str(e) for e in errors)
        super().__init__(f"{len(errors)} worker(s) failed:{joined}")


def _worker_entry(
    rank: int,
    world_size: int,
    master_port: int,
    result_queue: mp.Queue,
    done_event: mp.Event,
    fn: Callable[..., Any],
    args: tuple,
    kwargs: dict,
) -> None:
    """Runs inside each spawned process."""
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
    done_event.wait(timeout=60)


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
    """
    Launch ``fn`` on ``world_size`` worker processes and return a list of
    results indexed by rank.

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

    Parameters
    ----------
    fn : callable
        The per-rank function to run.
    world_size : int
        Number of worker processes to spawn.
    timeout : float
        Seconds to wait for all workers to finish before giving up.
    *args, **kwargs
        Forwarded to ``fn``.

    Returns
    -------
    list
        Results in rank order. Tensors are returned on CPU.
    """
    master_port = _find_free_port()

    # 'spawn' is required for CUDA and safest for gloo too.
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    # Signalled after the parent has finished deserializing all results,
    # so workers keep their FD servers alive until then.
    done_event = ctx.Event()

    procs: list[mp.Process] = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_worker_entry,
            args=(
                rank,
                world_size,
                master_port,
                result_queue,
                done_event,
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

    # All results deserialized — workers can now exit safely.
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

    # Surface any non-zero exits that didn't report through the queue
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
            f"Distributed test timed out after {timeout}s; "
            f"ranks that never reported: {missing}"
        )

    if errors:
        raise DistributedTestError(sorted(errors, key=lambda e: e.rank))

    return [results[r] for r in range(world_size)]
