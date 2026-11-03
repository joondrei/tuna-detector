"""Does the cached feature data still match the current processing parameters?

    python scripts/verify_cache.py

Exits 0 when both caches are usable and 1 when they are not, naming the reason.
Nothing is written or deleted.
"""
import sys
from pathlib import Path

from configs.pipeline_config import DATA_DIR, processing_parameters

CACHE_DIR = DATA_DIR / "cache"
SPLITS = ("train", "val")


def cache_problems(cache_dir, processing_params, splits=SPLITS):
    """Why each split's cache cannot be used (empty list means it can).

    The whole test lives in `pipeline.preprocessing.CachedFeatureDataset`
    """
    from pipeline.preprocessing import CachedFeatureDataset   # noqa: PLC0415 (lazy)

    problems = []
    for split in splits:
        path = Path(cache_dir) / split
        if not (path / "meta.json").is_file():
            problems.append(f"{split}: no cache at {path}")
            continue
        try:
            CachedFeatureDataset(path, **processing_params)
        except Exception as exc:                          # disagreement, truncation, ...
            problems.append(f"{split}: {exc}")
    return problems


def check_cache(cache_dir, processing_params, splits=SPLITS):
    """True when every split's cache matches `processing_params`."""
    return not cache_problems(cache_dir, processing_params, splits)


def main():
    problems = cache_problems(CACHE_DIR, processing_parameters, SPLITS)
    for problem in problems:
        print("problem:", problem)
    print("compatible:", not problems)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())

