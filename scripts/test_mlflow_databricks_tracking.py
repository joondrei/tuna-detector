import os
import sys
from pathlib import Path
from configs.filters import default_spatial_filters, DEFAULT_TEXTURE_FILTER
from configs.pipeline_config import DATA_DIR, TRAIN_PATTERN, VAL_PATTERN
from pipeline.preprocessing import FeatureDataset
from pipeline.training import Trainer, MlflowTracker, Evaluator
from models.model import build_custom_resnet_model
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ENV = ("DATABRICKS_HOST", "DATABRICKS_TOKEN")


def require(package: str, extra: str):
    """Import an optional dependency or exit with the exact fix."""
    try:
        return __import__(package)
    except ImportError:
        logger.critical("missing '{}' — run: uv sync --extra {}", package, extra)
        raise SystemExit(1)


def load_env() -> None:
    from dotenv import find_dotenv, load_dotenv          # declared dep: python-dotenv
    path = Path(find_dotenv(usecwd=True) or (REPO_ROOT / ".env"))
    if not path.is_file():
        logger.critical("no .env found (searched from {} upward)", Path.cwd())
        raise SystemExit(1)
    load_dotenv(path)
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        logger.critical(".env at {} is missing keys: {}", path, missing)
        raise SystemExit(1)
    logger.info(".env loaded from {}", path)


def main() -> int:
    load_env()
    require("mlflow", "all")
    require("databricks.sdk", "all")

    SEED = 42

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
        "texture_filter": DEFAULT_TEXTURE_FILTER,
        "fft_filter_size": 30,
        "freq_patch_scaling_func": "minmax_patches",
        "spatial_patch_scaling_func": "minmax_patches",
        "spatial_filters": default_spatial_filters(),
        "shuffle": True,
    }

    training_parameters = {
        "num_epochs": 20,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "pos_weight": 0.2,                 
        "dataset_workers": 0,
        "max_samples_per_epoch": 1000,      
        "log_every": 5,                    
        "grad_clip": None,
        "threshold": 0.5,

        "checkpoint_dir": "checkpoints",   
        "checkpoint_every": 1,             
        "keep_checkpoints": 5,
        "monitor": "auroc",                
        "monitor_mode": None,              
        "early_stopping_patience": 10,      
        "min_delta": 1e-4,
        "restore_best": True,              
        "resume_from": None,              
    }

    validator = Evaluator(
        max_samples=300,
        dataset_factory=FeatureDataset
    )

    tracker = MlflowTracker(
        tracking_uri="databricks",
        experiment="/Shared/experiments/ai-detection/test",
        run_name="fishyrun",
        log_params=True,
        log_state_dict=True,
        log_checkpoints=True,
        log_history=True,
        log_system=True,
        async_logging=True,
    )

    trainer = Trainer(
        processing_parameters,
        build_custom_resnet_model,
        training_parameters,
        seed=SEED,
        dataset_factory=FeatureDataset,
        validation=validator,
        tracker=tracker
    )

    result = trainer.fit()

    return 0


if __name__ == "__main__":
    sys.exit(main())