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
import shutil
import sys
from pathlib import Path

import psutil
import pyarrow.parquet as pq
import torch
from loguru import logger

from configs.pipeline_config import (DATA_DIR, TEST_PATTERN, TRAIN_PATTERN,
                                     VAL_PATTERN, processing_parameters)
from pipeline.preprocessing import (CachedFeatureDataset, FeatureBuilder,
                                    FeatureDataset, ProcessingConfig)
from pipeline.utils import build_cache

PATTERNS = {"train": TRAIN_PATTERN, "val": VAL_PATTERN, "test": TEST_PATTERN}
SPLIT_ORDER = ("train", "val", "test") 
DTYPES = {16: torch.float16, 32: torch.float32}


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
        default=16,
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