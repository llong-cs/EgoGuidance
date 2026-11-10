"""Small distributed-runtime helpers shared by training and data setup."""

import os
import threading


def distributed_barrier() -> None:
    """Synchronize ranks while making the NCCL rank-to-device mapping explicit."""
    import torch
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return

    backend = str(dist.get_backend()).rsplit(".", 1)[-1].lower()
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL barrier requires an available CUDA device")
        dist.barrier(device_ids=[torch.cuda.current_device()])
        return

    dist.barrier()


def release_distributed(timeout_s: float = 120.0) -> None:
    """Tear the process group down and guarantee the process actually exits.

    A rank that raises mid-collective leaves its peers blocked, so the NCCL abort
    can never complete: ``destroy_process_group`` blocks and the job keeps its GPUs
    while doing nothing. Run the teardown on a daemon thread and, if it has not
    returned within ``timeout_s``, leave via ``os._exit`` so the GPUs are released.

    Set ``STARVLA_NO_HARD_EXIT=1`` to keep the interpreter alive after a stuck
    teardown, e.g. to attach a debugger to the hung rank.
    """
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return

    def _destroy() -> None:
        try:
            dist.destroy_process_group()
        except Exception:  # teardown must never mask the failure that got us here
            pass

    worker = threading.Thread(target=_destroy, name="destroy-process-group", daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive() and os.environ.get("STARVLA_NO_HARD_EXIT", "0") != "1":
        os._exit(1)
