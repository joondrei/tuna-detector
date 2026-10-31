"""
`processing_parameters` / `training_parameters` keep the exact key names used by
the notebook, so `Trainer(processing_parameters, build_custom_resnet_model, training_parameters)`
works unchanged. Two notebook-only values are replaced by portable ones:

  texture_filter        torch tensor           -> DEFAULT_TEXTURE_FILTER
  *_scaling_func        function object        -> "minmax_patches" (Scalers key)
  spatial_filters       notebook global list   -> build_spatial_filters()

Filters live in `configs.filters`; the model in `models.model`.
"""
import os
from pathlib import Path

from models.model import CustomResNet18, build_custom_resnet_model
from .filters import DEFAULT_TEXTURE_FILTER, build_spatial_filters

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data"

SEED = 42

TRAIN_PATTERN = "train*.parquet"
VAL_PATTERN = "val*.parquet"
TEST_PATTERN = "test*.parquet"

processing_parameters = {
    "filepattern": str(DATA_DIR / TRAIN_PATTERN),
    "features": ["fft_poor", "fft_rich"],
    "colorspace": "Y",
    "labeltype": "binary",
    "augment": False,
    "resize_img": True,
    "img_size": 256,
    "n_patches": 64,
    "patch_size": 32,
    "patch_tc_dim": 16,
    "pad_with": "min_rank",            # min_rank | max_rank: patch repeated in a short mosaic
    "ignore_patch_overlap": False,     # True: poor/rich may pick the same patches
    "texture_filter": DEFAULT_TEXTURE_FILTER,
    "fft_filter_size": 30,
    "freq_patch_scaling_func": "minmax_patches",
    "spatial_patch_scaling_func": "minmax_patches",
    "spatial_filters": build_spatial_filters(),
    "shuffle": True,
}

training_parameters = {
    "num_epochs": 1,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "pos_weight": 0.2,                 
    "dataset_workers": 0,
    "max_samples_per_epoch": 300,      # None -> full pass
    "log_every": 5,                    # 0 -> silent
    "grad_clip": None,
    "threshold": 0.5,

    # ---- checkpointing / early stopping --------------------------------
    "checkpoint_dir": "checkpoints",   
    "checkpoint_every": 1,             # 0 -> no per-epoch checkpoints
    "keep_checkpoints": 5,             # no. before chkpoint pruning 
    "monitor": "auroc",                # key of eval_fn's dict; "loss" -> train loss
    "monitor_mode": None,              # None -> "min" for loss, "max" otherwise
    "early_stopping_patience": 3,      # None/0 -> disabled
    "min_delta": 1e-4,
    "restore_best": True,              # load best.pt into the model at the end

    # ---- resume --------------------------------------------------------
    "resume_from": None,               # None | "auto" (= <checkpoint_dir>/last.pt) | path
}
