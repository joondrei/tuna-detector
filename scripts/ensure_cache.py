"""Make sure the cached features match the current processing parameters.

    python scripts/ensure_cache.py

The decision itself lives in `pipeline.utils.ensure_cache`; this file is only
the command-line entry point so the DAG (and a human) can run it unchanged.

Exits 0 when the caches are usable - rebuilding them first if they are stale -
and 1 when they are still unusable afterwards.
"""
import sys

from configs.pipeline_config import DATA_DIR, processing_parameters
from pipeline.utils import ensure_cache

CACHE_DIR = DATA_DIR / "cache"
SPLITS = ("train", "val")
WORKERS = 4          # reader processes while rebuilding; each holds a parquet chunk
REBUILD = True       # False: report only, never write


def main():
    result = ensure_cache(CACHE_DIR, processing_parameters, splits=SPLITS,
                          rebuild=REBUILD, workers=WORKERS)
    for problem in result["problems"]:
        print("problem:", problem)
    print("rebuilt:", result["rebuilt"])
    print("usable:", result["ok"])
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

# make check_cache use the meta.json finger print,
#  rebuild the cache to save space,
#  dag makes the permutations,
#  use TPE sampler but limit to same features but degraded