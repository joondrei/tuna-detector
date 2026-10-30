"""
  DEFAULT_TEXTURE_FILTER  - 3x3 Laplacian used for texture residuals
  BASE_KERNELS            - the 5x5 directional kernels (a, ad, b, bd, c, cd, d, e, f, g)
  build_spatial_filters() - the 30-kernel residual bank built from them
  SPATIAL_FILTERS         - that bank, ready to use
  default_spatial_filters() - a fresh copy of the bank
"""
import torch

__all__ = ["DEFAULT_TEXTURE_FILTER", "BASE_KERNELS", "build_spatial_filters",
           "SPATIAL_FILTERS", "default_spatial_filters"]

# 3x3 Laplacian: the texture residual filter
DEFAULT_TEXTURE_FILTER = torch.tensor([[0.0, 1.0, 0.0],
                                       [1.0, -4.0, 1.0],
                                       [0.0, 1.0, 0.0]], dtype=torch.float32)

# 5x5 base kernels for residual noise extraction
_BASE_KERNEL_VALUES = (
    ("a",  [[0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]]),
    ("ad", [[0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]]),
    ("b",  [[0, 0, -1, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, -3, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]]),
    ("bd", [[0, 0, 0, 0, -1],
            [0, 0, 0, 3, 0],
            [0, 0, -3, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0]]),
    ("c",  [[0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, -2, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]]),
    ("cd", [[0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, -2, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0]]),
    ("d",  [[0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]]),
    ("e",  [[-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]]),
    ("f",  [[0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0]]),
    ("g",  [[-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1]]),
)

BASE_KERNELS = {name: torch.tensor(values, dtype=torch.float32)
                for name, values in _BASE_KERNEL_VALUES}


def build_spatial_filters():
    """The 30-filter bank built from the base kernels (index order preserved)."""
    bank = []
    for name in ("a", "b", "d", "e"):
        m = BASE_KERNELS[name]
        bank.extend([m, m.flip(0), m.t(), m.t().flip(1)])
    for name in ("ad", "bd"):
        m = BASE_KERNELS[name]
        bank.extend([m, m.flip(0), m.t(), m.t().flip(0)])

    c, cd, f, g = BASE_KERNELS["c"], BASE_KERNELS["cd"], BASE_KERNELS["f"], BASE_KERNELS["g"]
    bank.extend([c, cd, c.t(), cd.flip(0), f, g])
    return bank


SPATIAL_FILTERS = tuple(build_spatial_filters())


def default_spatial_filters():
    """A fresh list of the filter bank, so callers cannot mutate the default."""
    return list(SPATIAL_FILTERS)
