"""
    python scripts/build_dataset_cache.py                
    python scripts/build_dataset_cache.py --overwrite    # delete and rebuild, no questions
    python scripts/build_dataset_cache.py --cache-dir /mnt/fast/cache
    python scripts/build_dataset_cache.py --dtype 32     # full precision, twice the size
    python scripts/build_dataset_cache.py --splits test  # only the test cache

A finished cache is a directory holding `samples.bin`, `labels.npy` and `meta.json`;
`pipeline.preprocessing.CachedFeatureDataset` reads it back.

"""
import argparse
import glob
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import pyarrow.parquet as pq
import torch
from loguru import logger
from torch.utils.data import DataLoader

from configs.pipeline_config import (DATA_DIR, TEST_PATTERN, TRAIN_PATTERN,
                                     VAL_PATTERN, processing_parameters)
from pipeline.preprocessing import (CachedFeatureDataset, FeatureBuilder,
                                    FeatureDataset, ProcessingConfig)
from pipeline.utils import _fingerprint

PATTERNS = {"train": TRAIN_PATTERN, "val": VAL_PATTERN, "test": TEST_PATTERN}
SPLIT_ORDER = ("train", "val", "test") 
DTYPES = {16: torch.float16, 32: torch.float32}


def iter_samples(loader):
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


def build_cache(cache_dir, processing_parameters, dataset_factory, batch_size=64,
                    dataset_workers=4, dtype=torch.float32,
                overwrite=False, log_every=250):
    """Write a fixed-size-record binary cache from a DataLoader over dataset(**processing_parameters).

    `dtype` must be torch.float32 or torch.float16.
    torch.uint8 is not supported because features need to be normalised to [0, 1]
    and casting uint8 -> float32 is lossy.

    """
    if dtype is torch.uint8:
        raise ValueError(
            "dtype=torch.uint8 is not supported: lossy data conversion [0, 1] ")

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

    dataset = dataset_factory(**processing_parameters)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=dataset_workers)

    order, layout = [], {}
    record_bytes = 0
    labels = []
    written = 0
    started = time.perf_counter()

    try:
        with open(samples_tmp, "wb") as sink:
            for features, label in iter_samples(loader):
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
            "fingerprint": _fingerprint({"processing_parameters": processing_parameters,
                                         "feature_order": order,
                                         "features": feature_layout}),
            "n_samples": written,
            "record_bytes": record_bytes,
            "label_dtype": "int64",
            "feature_order": order,
            "features": feature_layout,
            "config": processing_parameters,
            "source": {"n_samples": len(dataset) if hasattr(dataset, "__len__") else written,
                       "processing_parameters": processing_parameters}}

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
            "features": order, "workers": dataset_workers,
            "dtypes": {name: str(spec["dtype"]).replace("torch.", "") for name, spec in layout.items()},
            "fingerprint": meta["fingerprint"], "elapsed": elapsed,
            "samples": str(cache_dir / "samples.bin"),
            "labels": str(cache_dir / "labels.npy"),
            "meta": str(cache_dir / "meta.json")}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build the feature caches for the train, val and test splits.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DATA_DIR / "cache",
        help="root directory holding one sub-directory per split (default: %(default)s)")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="delete the contents of existing cache directories and skip the "
             "confirmation questions")
    parser.add_argument(
        "--dtype",
        type=int,
        choices=(16, 32),
        default=32,
        help="bits per stored number: 16 is half the size of 32 but keeps fewer "
             "decimals (default: %(default)s)")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLIT_ORDER,
        default=["train", "val"],
        metavar="SPLIT",
        help="which caches to build, one or more of: train, val, test "
             "(default: train val); each becomes <cache-dir>/<split>")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="samples handed over per batch while reading the dataset "
             "(default: %(default)s)")
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        metavar="N",
        help="reader processes: 0 keeps everything in one process; more decode faster "
             "but each holds a table chunk in memory (default: %(default)s)")
    return parser.parse_args(argv)


def dir_bytes(path):
    """Bytes taken by the files directly inside `path` (0 when it does not exist)."""
    if not path.is_dir():
        return 0
    return sum(item.stat().st_size for item in path.iterdir() if item.is_file())


def describe(path):
    """Return (has_content, text): what sits in `path`, and what deleting it would lose."""
    if not path.exists():
        return False, "does not exist yet"
    files = [item for item in path.iterdir() if item.is_file()]
    if not files:
        return False, "empty"
    size = dir_bytes(path)
    state = ("complete cache" if any(item.name == "meta.json" for item in files)
             else "partial cache (no meta.json)")
    return True, f"{state}, {size / 2 ** 30:.2f} GiB in {len(files)} files"


def confirm(question):
    """True or False from the terminal; None when there is no terminal to ask on."""
    if not sys.stdin.isatty():
        return None
    try:
        answer = input(f"{question} [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def guard(cache_dir):
    """Refuse a cache root whose deletion could take the dataset with it."""
    root = cache_dir.expanduser().resolve()
    if root == Path(root.anchor):
        raise ValueError(f"refusing to use the filesystem root {root} as a cache directory")
    if any(root.glob("*.parquet")):
        raise ValueError(f"{root} contains .parquet files; pass a sub-directory such as "
                         f"{root / 'cache'} so the dataset cannot be deleted")
    return root


def record_bytes(processing_params, dtype):
    """ # of Bytes one cached sample needs, measured on a made-up image of the right size.

    """
    config = ProcessingConfig(**processing_params)
    builder = FeatureBuilder.from_config(config)
    side = int(getattr(config, "img_size", 0) or 512)
    features = builder(torch.rand(3, side, side))
    per_value = torch.empty((), dtype=dtype).element_size()
    return sum(int(tensor.numel()) * per_value for tensor in features.values()), list(features)


def count_rows(pattern):
    """Number of files and rows behind a pattern, read from the parquet footers only."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no parquet files match {pattern!r}")
    return len(files), sum(pq.ParquetFile(name).metadata.num_rows for name in files)


def free_bytes(path):
    """Free space on the disk that will hold `path`, even before it is created."""
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def verify(cache_dir, stats, dtype):
    """Reopen a finished cache, read it back, and describe what is inside.

    Raises RuntimeError when the files on disk disagree with what was just written,
    so a bad build is reported here rather than in the middle of training.
    """
    dataset = CachedFeatureDataset(cache_dir)
    wanted = str(dtype).replace("torch.", "")

    if len(dataset) != stats["n_samples"]:
        raise RuntimeError(f"{cache_dir}: {len(dataset)} samples readable, "
                           f"{stats['n_samples']} were written")
    if len(dataset) != len(dataset.labels):
        raise RuntimeError(f"{cache_dir}: {len(dataset.labels)} labels for "
                           f"{len(dataset)} samples")

    stored = {name: spec["dtype"] for name, spec in dataset.features.items()}
    wrong = {name: got for name, got in stored.items() if got != wanted}
    if wrong:
        raise RuntimeError(f"{cache_dir}: stored as {wrong}, but {wanted} was requested")

    low, high = 0.0, 1.0
    for index in sorted({0, len(dataset) // 2, len(dataset) - 1}):
        sample = dataset[index]
        for name, tensor in sample["features"].items():
            expected_shape = tuple(dataset.features[name]["shape"])
            if tuple(tensor.shape) != expected_shape:
                raise RuntimeError(f"{cache_dir}: {name} reads back as "
                                   f"{tuple(tensor.shape)}, written as {expected_shape}")
            if not bool(torch.isfinite(tensor).all()):
                raise RuntimeError(f"{cache_dir}: {name} holds NaN or infinity")
            low = min(low, float(tensor.min()))
            high = max(high, float(tensor.max()))

    if low < -1e-3 or high > 1 + 1e-3:
        logger.warning(f"{cache_dir}: values reach {low:.3f}..{high:.3f}, "
                       f"expected everything scaled into [0, 1]")

    positive = float(dataset.labels.mean()) if len(dataset) else 0.0
    return (f"{len(dataset)} samples, features {dataset.feature_names}, stored as "
            f"{wanted}, {positive:.1%} positive, values {low:.3f}..{high:.3f}")


def main(argv=None):
    args = parse_args(argv)
    if args.batch_size < 1 or args.workers < 0:
        logger.error(f"batch size must be at least 1 and workers at least 0, "
                     f"got batch_size={args.batch_size} workers={args.workers}")
        return 2
    try:
        cache_root = guard(args.cache_dir)
    except ValueError as error:
        logger.error(str(error))
        return 2

    if args.workers > 0:
        free_ram = psutil.virtual_memory().available
        logger.warning(
            f"{args.workers} worker process(es): every worker is a separate process with "
            f"its own copy of the reader and {args.batch_size} decoded images in flight, "
            f"so memory grows with workers x batch size and not just with the dataset "
            f"({free_ram / 2 ** 30:.1f} GiB of RAM free right now); if the machine starts "
            f"swapping or the build stops with 'Killed', lower --workers or --batch-size")

    splits = [name for name in SPLIT_ORDER if name in set(args.splits)]
    logger.info(f"building {len(splits)} cache(s): {', '.join(splits)}")

    present = {}
    for split in splits:
        has_content, text = describe(cache_root / split)
        if has_content:
            present[split] = text

    dtype = DTYPES[args.dtype]
    dtype_name = f"float{args.dtype}"

    logger.info(f"reader: batch size {args.batch_size}, {args.workers} worker process(es)")
    per_record, feature_names = record_bytes(processing_parameters, dtype)
    rows = {}
    for split in splits:
        n_files, n_rows = count_rows(str(DATA_DIR / PATTERNS[split]))
        rows[split] = n_rows
        logger.info(f"plan {split:>5s}: {n_files} files, {n_rows:,} samples x "
                    f"{per_record / 2 ** 10:.0f} KiB = "
                    f"{n_rows * per_record / 2 ** 30:.2f} GiB")

    required = sum(n_rows * per_record for n_rows in rows.values())
    reclaim = sum(dir_bytes(cache_root / split) for split in splits)
    needed = max(0, required - reclaim)
    available = free_bytes(cache_root)
    logger.info(f"plan total: {required / 2 ** 30:.2f} GiB as {dtype_name} "
                f"({', '.join(feature_names)})")
    if reclaim:
        logger.info(f"plan total: {reclaim / 2 ** 30:.2f} GiB is freed by deleting "
                    f"the old caches first")
    logger.info(f"plan total: {needed / 2 ** 30:.2f} GiB to write, "
                f"{available / 2 ** 30:.2f} GiB free at {cache_root}")
    if needed > available:
        logger.error(f"not enough space: {needed / 2 ** 30:.2f} GiB needed, "
                     f"{available / 2 ** 30:.2f} GiB free at {cache_root}")
        return 2

    for split, text in present.items():
        logger.warning(f"existing {split} cache at {cache_root / split}: {text}")

    if not args.overwrite:
        action = "Delete these caches and rebuild" if present else "Build"
        answer = confirm(f"{action} {', '.join(splits)} "
                         f"({required / 2 ** 30:.2f} GiB)?")
        if answer is None:
            logger.error("no terminal to ask on, so nothing was changed; "
                         "pass --overwrite to skip the question")
            return 2
        if not answer:
            logger.info("nothing was deleted and nothing was built")
            return 0

    for split in present:
        target = cache_root / split
        logger.info(f"deleting contents of {target}")
        shutil.rmtree(target)

    results = []

    for split in splits:
        params = dict(processing_parameters)
        params["filepattern"] = str(DATA_DIR / PATTERNS[split])
        params["shuffle"] = False        
        target = cache_root / split
        logger.info(f"building {split} cache -> {target} ({dtype_name})")
        stats = build_cache(
            cache_dir=target,
            processing_parameters=params,
            dataset_factory=FeatureDataset,
            batch_size=args.batch_size,
            dataset_workers=args.workers,
            dtype=dtype,
            overwrite=True,
        )
        logger.info(f"built {split} cache: {stats['n_samples']} samples, "
                    f"{stats['bytes'] / 2 ** 30:.2f} GiB in {stats['elapsed']:.1f}s")
        logger.info(f"checked {split} cache: {verify(target, stats, dtype)}")
        results.append((split, stats))

    logger.info("summary")
    for split, stats in results:
        logger.info(f"  {split:5s} {stats['n_samples']:>8,} samples  "
                    f"{stats['bytes'] / 2 ** 30:7.2f} GiB  {stats['elapsed']:8.1f}s")
    logger.info(f"caches in {cache_root}, stored as {dtype_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())