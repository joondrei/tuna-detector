"""Defactify image-detection pipeline.

    from pipeline import (Trainer, Evaluator, NullTracker, FeatureDataset,
                          processing_parameters, training_parameters, build_model)
"""
from configs.filters import (BASE_KERNELS, DEFAULT_TEXTURE_FILTER,
                             SPATIAL_FILTERS, build_spatial_filters,
                             default_spatial_filters)
from configs.pipeline_config import (DATA_DIR, SEED, TEST_PATTERN, TRAIN_PATTERN,
                                     VAL_PATTERN, CustomResNet18, build_model,
                                     processing_parameters, training_parameters)
from models.model import (AvgPoolPadding, ConvProjection, IdentityPadding,
                          ResBlock, ResNet, feature_extractor)
from .preprocessing import (ColorSpaceConverter, FeatureBuilder, FeatureDataset,
                            FilterBank, ImageAugmenter, MosaicAssembler,
                            ParquetImageSource, PatchExtractor, PatchRanker,
                            ProcessingConfig, Scalers, TextureComplexityScorer)
from .training import (CallableValidator, CheckpointStore, EarlyStopping,
                       Evaluator, MlflowTracker, NullTracker, NullValidator,
                       TrainConfig, Trainer, move_batch, train_step,
                       update_metrics)
from .utils import fixed_rngs, make_metrics, rng_generator, seed_worker

__all__ = [
    # config
    "DATA_DIR", "SEED", "TRAIN_PATTERN", "VAL_PATTERN", "TEST_PATTERN",
    "processing_parameters", "training_parameters", "build_model",
    "CustomResNet18",
    # model
    "feature_extractor", "ResNet", "ResBlock", "IdentityPadding",
    "ConvProjection", "AvgPoolPadding",
    # preprocessing
    "ProcessingConfig", "ColorSpaceConverter", "PatchExtractor", "FilterBank",
    "TextureComplexityScorer", "Scalers", "PatchRanker", "MosaicAssembler",
    "ImageAugmenter", "FeatureBuilder", "ParquetImageSource", "FeatureDataset",
    "DEFAULT_TEXTURE_FILTER", "BASE_KERNELS", "SPATIAL_FILTERS",
    "build_spatial_filters", "default_spatial_filters",
    # training
    "TrainConfig", "EarlyStopping", "CheckpointStore", "NullTracker",
    "MlflowTracker", "NullValidator", "CallableValidator", "Evaluator",
    "Trainer", "move_batch", "train_step", "update_metrics",
    # utils
    "fixed_rngs", "seed_worker", "rng_generator", "make_metrics",
]
