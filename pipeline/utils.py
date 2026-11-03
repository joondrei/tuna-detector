import os
import random
from pathlib import Path
import hashlib
import json


import numpy as np
import torch
from torcheval.metrics import (BinaryAccuracy, BinaryAUROC, BinaryF1Score,
                               BinaryRecall)

from loguru import logger
from collections.abc import Callable
import time
from typing import ParamSpec, TypeVar
from functools import wraps
P = ParamSpec("P")
R = TypeVar("R")

__all__ = ["fixed_rngs", "seed_worker", "make_metrics", "rng_generator",
           "prune_checkpoints", "_resolve_device", "_sanitize", "_atomic_torch_save",
           "_summary", "_flat_params", "_jsonable", "resolve_experiment",
           "DEFAULT_MLFLOW_EXPERIMENT_ROOT"]


# mlflow experiment naming
DEFAULT_MLFLOW_EXPERIMENT_ROOT = "/Shared/experiments/ai-detection"


def resolve_experiment(name=None, root=None):
    """Turn a short experiment name into a full MLflow experiment path.

    resolve_experiment(None)                -> "/Shared/experiments/ai-detection"
    resolve_experiment("resnet18")          -> "/Shared/experiments/ai-detection/resnet18"
    resolve_experiment("/Users/me/my_exp")  -> "/Users/me/my_exp"  (full path wins)
    resolve_experiment("resnet18", root="/tmp/x") -> "/tmp/x/resnet18"

    The root, in order of precedence: the `root` argument, the
    MLFLOW_EXPERIMENT_ROOT environment variable, MLFLOW_DATABRICKS_ROOT, then
    DEFAULT_MLFLOW_EXPERIMENT_ROOT. Read at call time, so a .env file loaded by
    the caller is honoured.
    """
    root = (root or os.environ.get("MLFLOW_EXPERIMENT_ROOT")
            or os.environ.get("MLFLOW_DATABRICKS_ROOT")
            or DEFAULT_MLFLOW_EXPERIMENT_ROOT).strip().rstrip("/")
    if name is None or str(name).strip() == "":
        return root
    name = str(name).strip()
    if name.startswith("/"):            # already a full path
        return name.rstrip("/")
    return f"{root}/{name.strip('/')}"


# rngs
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


# device/io
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


# metrics
def make_metrics(device, threshold=0.5):
    """torcheval binary metrics take PROBABILITIES and threshold internally."""
    return {
        "accuracy": BinaryAccuracy(threshold=threshold, device=device),
        "recall": BinaryRecall(threshold=threshold, device=device),
        "auroc": BinaryAUROC(device=device),
        "f1": BinaryF1Score(threshold=threshold, device=device),
    }


# config serialising
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

def _fingerprint(payload):
    """Short, stable hash of a JSON-able payload (used to label cached data)."""
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]

