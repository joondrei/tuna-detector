"""Shared, dependency-light helpers used by the pipeline modules.

Everything here was previously defined in the notebook cells:
`fixed_rngs`, `seed_worker`, `make_metrics`, `_resolve_device`, `_sanitize`,
`_atomic_torch_save`, `_flat_params`, `_jsonable`.

The leading underscores are kept so the call sites in `training.py` stay
identical to the notebook version.
"""
import os
import random
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone


import numpy as np
import torch
from torcheval.metrics import (BinaryAccuracy, BinaryAUROC, BinaryF1Score,
                               BinaryRecall)
from torch.utils.data import DataLoader

from loguru import logger
from collections.abc import Callable
import time
from typing import ParamSpec, TypeVar
from functools import wraps
P = ParamSpec("P")
R = TypeVar("R")

__all__ = ["fixed_rngs", "seed_worker", "make_metrics", "rng_generator",
           "prune_checkpoints", "_resolve_device", "_sanitize", "_atomic_torch_save",
           "_summary", "_flat_params", "_jsonable"]


# --------------------------------------------------------------------- rngs
def fixed_rngs(seed=42, deterministic=True, warn_only=True):
    """Seed every RNG used by the preprocess -> train -> eval pipeline."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(False)
    return rng_generator(seed)


def rng_generator(seed=42):
    """Seeded torch generator for DataLoader shuffling / workers."""
    return torch.Generator().manual_seed(int(seed))


def seed_worker(worker_id):
    """Reseed numpy/random inside DataLoader worker processes."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------- device/io
def _resolve_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _atomic_torch_save(payload, path):
    """Write to a temp file then rename: never leaves a truncated checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def prune_checkpoints(directory, keep=5, prefix="epoch_"):
    """Delete all but the `keep` most recent <<prefix>><number>.pt files.

    Only files whose name is exactly the prefix plus an integer are considered,
    so `best.pt`, `last.pt` and in-flight `*.tmp` files are never touched. The
    newest file wins by the parsed number, not by name or mtime, so it stays
    correct past 999 epochs (`epoch_1000.pt` sorts before `epoch_999.pt`
    lexicographically but after it numerically).

    `keep <= 0` (or None) disables pruning. Returns the deleted paths, oldest
    first, so callers can log what went away.
    """
    directory = Path(directory)
    if not keep or int(keep) <= 0 or not directory.is_dir():
        return []
    keep = int(keep)

    numbered = []
    for path in directory.iterdir():
        if not path.is_file() or path.suffix != ".pt" or not path.name.startswith(prefix):
            continue
        try:
            number = int(path.name[len(prefix):-len(".pt")])
        except ValueError:
            continue                      # e.g. epoch_backup.pt -> not ours
        numbered.append((number, path))

    if len(numbered) <= keep:
        return []

    numbered.sort(key=lambda item: item[0])     # ascending number == oldest first
    removed = []
    for _, path in numbered[:-keep]:
        try:
            path.unlink()
            removed.append(path)
        except OSError:
            continue                      # already gone or locked; never break training
    return removed


# ------------------------------------------------------------------ metrics
def make_metrics(device, threshold=0.5):
    """torcheval binary metrics take PROBABILITIES and threshold internally."""
    return {
        "accuracy": BinaryAccuracy(threshold=threshold, device=device),
        "recall": BinaryRecall(threshold=threshold, device=device),
        "auroc": BinaryAUROC(device=device),
        "f1": BinaryF1Score(threshold=threshold, device=device),
    }


# ------------------------------------------------------- config serialising
def _sanitize(obj):
    """Make configs safe for torch.load(..., weights_only=True)."""
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return repr(obj)                   # callables (scaling fns) -> string


def _summary(value):
    """Short, stable, comparable text for values a param cannot hold.

    Never uses repr() of tensors/functions: those carry memory addresses or
    tensor dumps that make runs look different when they are identical.
    """
    if isinstance(value, torch.Tensor):
        return f"tensor{tuple(value.shape)}"
    if isinstance(value, (list, tuple)):
        if all(isinstance(v, (str, int, float, bool, type(None))) for v in value):
            return ",".join(str(v) for v in value)
        return f"<{len(value)} items>"
    if callable(value):
        return getattr(value, "__name__", type(value).__name__)
    return str(value)


def _flat_params(config, prefix=""):
    """Flatten a nested config into {str: primitive} so params stay filterable."""
    flat = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flat_params(value, prefix=f"{name}."))
        elif isinstance(value, (str, int, float, bool, type(None))):
            flat[name] = "None" if value is None else value
        else:
            flat[name] = _summary(value)
    return flat


def _jsonable(obj):
    """Same job as _flat_params but keeps nesting, for mlflow.log_dict."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return _summary(obj)

def timeit(func: Callable[P, R]) -> Callable[P, R]:
    """Measure and log the execution time of a function.

    :param func: The function whose execution time is to be measured.
    :return: A wrapped function that logs its execution duration.
    """

    @wraps(func)
    def timeit_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.info(f"Function {func.__name__} Took {total_time:.4f} seconds")
        return result

    return timeit_wrapper


def build_cache(cache_dir, processing_params, processor, dtype=torch.float32,
                overwrite=False, log_every=250):
    """Write a fixed-size-record binary cache from a DataLoader over processor(**processing_params).

    `dtype` must be torch.float32 or torch.float16.
    """
    if dtype is torch.uint8:
        raise ValueError(
            "dtype=torch.uint8 is not supported: casting normalised features in [0, 1] "

    def _iter_samples(loader):
        """Yield ({name: tensor}, label) one sample at a time from collated batches."""
        for batch in loader:
            if isinstance(batch, dict):
                missing = {"features", "label"} - batch.keys()
                if missing:
                    raise TypeError(f"expected a batch dict with 'features' and 'label', "
                                    f"missing {sorted(missing)}")
                features, labels = batch["features"], batch["label"]
            elif isinstance(batch, (tuple, list)) and len(batch) == 2:
                features, labels = batch
            else:
                raise TypeError("expected each batch to be (features, label) or "
                                f"{{'features', 'label'}}, got {type(batch).__name__}")
            if not isinstance(features, dict):
                raise TypeError(f"expected features to be a dict, got {type(features).__name__}")
            flat = torch.as_tensor(labels).reshape(-1)
            for i in range(flat.shape[0]):
                yield {name: value[i] for name, value in features.items()}, flat[i]


    def _fingerprint(payload):
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    cache_dir = Path(cache_dir)
    if cache_dir.exists() and any(cache_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{cache_dir} already has files; pass overwrite=True")
    cache_dir.mkdir(parents=True, exist_ok=True)

    samples_tmp = cache_dir / "samples.bin.tmp"
    labels_tmp = cache_dir / "labels.npy.tmp"
    meta_tmp = cache_dir / "meta.json.tmp"
    for stale in (samples_tmp, labels_tmp, meta_tmp):
        stale.unlink(missing_ok=True)
    if overwrite:
        (cache_dir / "meta.json").unlink(missing_ok=True)   # invalidate the previous cache

    dataset = processor(**processing_params)
    loader = DataLoader(dataset, batch_size=64, num_workers=0)

    order, layout = [], {}
    record_bytes = 0
    labels = []
    written = 0
    started = time.perf_counter()

    try:
        with open(samples_tmp, "wb") as sink:
            for features, label in _iter_samples(loader):
                if not layout:
                    if not features:
                        raise ValueError("dataset yielded a sample with no features")
                    order = list(features)
                    offset = 0
                    for name in order:
                        tensor = torch.as_tensor(features[name])
                        feature_dtype = dtype
                        nbytes = tensor.numel() * torch.empty((), dtype=feature_dtype).element_size()
                        layout[name] = {"shape": tuple(tensor.shape), "dtype": feature_dtype,
                                        "offset": offset, "nbytes": nbytes}
                        offset += nbytes
                    record_bytes = offset

                blobs = []
                for name in order:
                    spec = layout[name]
                    tensor = torch.as_tensor(features[name]).detach().cpu()
                    shape = tuple(tensor.shape)
                    if shape != spec["shape"]:
                        raise ValueError(
                            f"feature {name!r} has shape {shape}, expected {spec['shape']} "
                            f"(from the first sample). A cache needs fixed-size records: resample "
                            f"every sample to one size, or drop the varying feature.")
                    target_dtype = spec["dtype"]
                    if tensor.dtype != target_dtype:
                        tensor = tensor.to(target_dtype)

                    blobs.append(tensor.contiguous().numpy().tobytes())
                record = b"".join(blobs)
                if len(record) != record_bytes:
                    raise RuntimeError(f"sample {written} serialized to {len(record)} bytes, "
                                       f"expected {record_bytes}")
                sink.write(record)

                labels.append(int(label))
                written += 1
                if log_every and written % int(log_every) == 0:
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    print(f"[cache] {written} samples | {written / elapsed:.1f}/s | "
                          f"{written * record_bytes / 2 ** 30:.2f} GiB")

        if written == 0:
            raise ValueError("dataset produced no samples; nothing to cache")
        expected = written * record_bytes
        actual = samples_tmp.stat().st_size
        if actual != expected:
            raise RuntimeError(f"{samples_tmp} is {actual} bytes, "
                               f"expected {expected} ({written} x {record_bytes})")

        with open(labels_tmp, "wb") as sink:
            np.save(sink, np.asarray(labels, dtype=np.int64))
    except BaseException:
        for stale in (samples_tmp, labels_tmp, meta_tmp):
            stale.unlink(missing_ok=True)
        raise

    elapsed = time.perf_counter() - started
    feature_layout = {name: {"shape": list(spec["shape"]),
                             "dtype": str(spec["dtype"]).replace("torch.", ""),
                             "offset": spec["offset"],
                             "nbytes": spec["nbytes"]}
                      for name, spec in layout.items()}
    meta = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "fingerprint": _fingerprint({"processing_params": processing_params,
                                         "feature_order": order,
                                         "features": feature_layout}),
            "n_samples": written,
            "record_bytes": record_bytes,
            "label_dtype": "int64",
            "feature_order": order,
            "features": feature_layout,
            "config": processing_params,
            "source": {"n_samples": len(dataset) if hasattr(dataset, "__len__") else written,
                       "processing_params": processing_params}}

    with open(meta_tmp, "w") as sink:
        json.dump(meta, sink, indent=2, default=str)
    os.replace(labels_tmp, cache_dir / "labels.npy")
    os.replace(samples_tmp, cache_dir / "samples.bin")
    os.replace(meta_tmp, cache_dir / "meta.json")      # written last = completeness marker

    print(f"[cache] {written} samples -> {cache_dir} | "
          f"{record_bytes / 2 ** 10:.0f} KiB/sample | "
          f"{record_bytes * written / 2 ** 30:.2f} GiB | {elapsed:.1f}s")
    return {"cache_dir": str(cache_dir), "n_samples": written,
            "record_bytes": record_bytes, "bytes": record_bytes * written,
            "features": order, "workers": 0,
            "dtypes": {name: str(spec["dtype"]).replace("torch.", "") for name, spec in layout.items()},
            "fingerprint": meta["fingerprint"], "elapsed": elapsed,
            "samples": str(cache_dir / "samples.bin"),
            "labels": str(cache_dir / "labels.npy"),
            "meta": str(cache_dir / "meta.json")}
