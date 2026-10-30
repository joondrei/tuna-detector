from configs.filters import default_spatial_filters, DEFAULT_TEXTURE_FILTER
from configs.pipeline_config import DATA_DIR, TRAIN_PATTERN, VAL_PATTERN
from pipeline.preprocessing import FeatureDataset
from pipeline.training import Trainer, Evaluator
from models.model import build_custom_resnet_model

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

validator = Evaluator(batch_size=64,
          workers=0,
          max_samples=300,
          threshold=0.5,
          dataset_factory=FeatureDataset
)

trainer = Trainer(processing_parameters, build_custom_resnet_model, training_parameters,
                  seed=SEED,
                  dataset_factory=FeatureDataset,
                  validation=validator)
result = trainer.fit()
