# =============================================================================
# DatasetPreprocessing, refactored into composable pieces
#
#   ProcessingConfig        - validated processing_parameters (no io, no tensors built)
#   ColorSpaceConverter     - RGB tensor -> Y / Cb / Cr / YCbCr
#   PatchExtractor          - center crop + non-overlapping unfold
#   FilterBank              - residual filters (channel-wise) + FFT high-pass
#   TextureComplexityScorer - per-patch log-energy score
#   Scalers                 - registry of patch normalisers (name or callable)
#   PatchRanker             - pick the N least / most textured patches
#   MosaicAssembler         - N patches -> one grid image
#   ImageAugmenter          - flip / random crop, RNG injectable
#   FeatureBuilder          - tensor -> {'feature': tensor} (no io, no workers)
#   ParquetImageSource      - file discovery, worker sharding, epoch shuffle, decoding
#   FeatureDataset          - IterableDataset gluing Source + Builder
#
# Filters (Laplacian + the 30-kernel residual bank) live in configs/filters.py
#
# `FeatureDataset(**processing_parameters, seed=SEED)` is a drop-in replacement
# for `DatasetPreprocessing` (same kwargs, same yielded dict), and every part is
# injectable, so it also works as `Trainer(..., dataset_factory=FeatureDataset)`.
# =============================================================================
import glob
import io
import math

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
import pyarrow.parquet as pq
from PIL import Image

from torchvision.transforms.functional import center_crop, to_tensor, resize
import torchvision.transforms as transforms

from configs.filters import (DEFAULT_TEXTURE_FILTER, SPATIAL_FILTERS,
                             build_spatial_filters, default_spatial_filters)



# ------------------------------------------------------------------ config
class ProcessingConfig:
    """Validates `processing_parameters` for the preprocessing pipeline."""

    LABELTYPES = ("binary", "multiclass")
    FEATURE_NAMES = ("residuals", "tc", "prepatch", "poor", "rich", "fft_poor", "fft_rich")

    def __init__(self, filepattern, features=None, colorspace="RGB", labeltype="binary",
                 augment=False, resize_img=True, img_size=256, n_patches=64,
                 patch_size=32, patch_tc_dim=16, texture_filter=None, fft_filter_size=30,
                 freq_patch_scaling_func="minmax_patches",
                 spatial_patch_scaling_func="minmax_patches", spatial_filters=None,
                 shuffle=False):
        self.filepattern = filepattern
        self.features = list(features or ["prepatch"])
        self.colorspace = colorspace
        self.labeltype = labeltype
        self.augment = bool(augment)
        self.resize_img = bool(resize_img)
        self.img_size = int(img_size)
        self.n_patches = int(n_patches)
        self.patch_size = int(patch_size)
        self.patch_tc_dim = int(patch_tc_dim)
        self.texture_filter = DEFAULT_TEXTURE_FILTER if texture_filter is None else texture_filter
        self.fft_filter_size = int(fft_filter_size)
        self.freq_patch_scaling_func = freq_patch_scaling_func
        self.spatial_patch_scaling_func = spatial_patch_scaling_func
        self.spatial_filters = default_spatial_filters() if spatial_filters is None else spatial_filters
        self.shuffle = bool(shuffle)
        self.validate()

    # -- validation --------------------------------------------------------
    def validate(self):
        unknown = set(self.features) - set(FEATURE_NAMES)
        if not self.features or unknown:
            raise ValueError(f"features must be a non-empty subset of {list(FEATURE_NAMES)}, "
                             f"got {self.features}")
        if self.labeltype not in self.LABELTYPES:
            raise ValueError(f"labeltype must be one of {list(self.LABELTYPES)}")
        if self.n_patches < 1 or math.isqrt(self.n_patches) ** 2 != self.n_patches:
            raise ValueError(f"n_patches must be a perfect square, got {self.n_patches}")
        if self.patch_size < 1 or self.img_size < 1:
            raise ValueError("img_size and patch_size must be positive")
        return self

    # -- derived -----------------------------------------------------------
    @property
    def needs_residuals(self):
        return "residuals" in self.features or "tc" in self.features

    @property
    def needs_spatial_patches(self):
        return "poor" in self.features or "rich" in self.features

    @property
    def needs_fft_patches(self):
        return "fft_poor" in self.features or "fft_rich" in self.features

    @property
    def needs_prepatch(self):
        return (self.needs_spatial_patches or self.needs_fft_patches
                or "prepatch" in self.features)

    def to_dict(self):
        """The processing_parameters this config came from (seed excluded)."""
        return {"filepattern": self.filepattern,
                "features": list(self.features),
                "colorspace": self.colorspace,
                "labeltype": self.labeltype,
                "augment": self.augment,
                "resize_img": self.resize_img,
                "img_size": self.img_size,
                "n_patches": self.n_patches,
                "patch_size": self.patch_size,
                "patch_tc_dim": self.patch_tc_dim,
                "texture_filter": self.texture_filter,
                "fft_filter_size": self.fft_filter_size,
                "freq_patch_scaling_func": self.freq_patch_scaling_func,
                "spatial_patch_scaling_func": self.spatial_patch_scaling_func,
                "spatial_filters": self.spatial_filters,
                "shuffle": self.shuffle}


# ------------------------------------------------------------- colour space
class ColorSpaceConverter:
    """RGB tensor (*, 3, H, W) -> one luma/chroma plane or a full YCbCr stack."""

    DELTA = 0.5

    def __init__(self, channel="RGB"):
        self.channel = (channel or "rgb").lower()

    def __call__(self, image):
        if not torch.is_tensor(image):
            raise TypeError(f"input must be a torch.Tensor, got {type(image)}")
        if image.shape[-3] != 3:
            raise ValueError(f"input must have shape (*, 3, H, W), got {tuple(image.shape)}")
        if self.channel == "rgb":
            return image

        r, g, b = image[..., 0, :, :], image[..., 1, :, :], image[..., 2, :, :]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = (b - y) * 0.564 + self.DELTA
        cr = (r - y) * 0.713 + self.DELTA

        if self.channel in ("y", "cb", "cr"):
            return {"y": y, "cb": cb, "cr": cr}[self.channel].unsqueeze(0)
        if self.channel == "ycbcr":
            return torch.stack((y, cb, cr), -3)
        raise ValueError(f"unknown channel {self.channel!r}")


# ----------------------------------------------------------------- patches
class PatchExtractor:
    """Crops to a multiple of `patch_size` then unfolds into non-overlapping patches."""

    def __init__(self, patch_size=32):
        self.patch_size = int(patch_size)

    def center_crop(self, image):
        p = self.patch_size
        h, w = image.shape[-2:]
        if p > h or p > w:
            raise ValueError(f"patch_size {p} must be smaller than image {h}x{w}")
        if h % p or w % p:
            return center_crop(image, [(h // p) * p, (w // p) * p])
        return image

    def unfold(self, image):
        p = self.patch_size
        patches = image.unfold(1, p, p).unfold(2, p, p)
        patches = patches.permute(1, 2, 0, 3, 4).contiguous()
        return patches.view(-1, image.shape[0], p, p)


# ----------------------------------------------------------------- filters
class FilterBank:
    """Residual convolution (channel-wise) and the FFT high-pass residual."""

    def __init__(self, filters=None, fft_filter_size=30):
        if filters is None:
            self.bank = None
        elif torch.is_tensor(filters):
            self.bank = filters
        else:
            self.bank = torch.stack(list(filters))
        self.fft_filter_size = int(fft_filter_size)

    @staticmethod
    def apply(image, filter_map):
        """Convolve `image` channel-wise with `filter_map` (2-D or a stack)."""
        depth = image.shape[1] if image.dim() == 4 else image.shape[0]
        if filter_map.dim() == 2:
            filter_map = filter_map.unsqueeze(0)
        kernel = filter_map.unsqueeze(1).repeat(depth, 1, 1, 1)
        k_h, k_w = kernel.shape[-2:]
        return F.conv2d(image, kernel, padding=(k_h // 2, k_w // 2), groups=depth, bias=None)

    def high_pass(self, image, filter_size=None):
        size = self.fft_filter_size if filter_size is None else int(filter_size)
        if image.dim() == 2:
            image = image.unsqueeze(0)

        _, rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2

        mask = torch.ones((rows, cols), device=image.device)
        mask[crow - size:crow + size, ccol - size:ccol + size] = 0

        spectrum = torch.fft.fftshift(torch.fft.fft2(image, dim=(-2, -1)), dim=(-2, -1))
        filtered = torch.fft.ifft2(
            torch.fft.ifftshift(spectrum * mask, dim=(-2, -1)), dim=(-2, -1))
        return torch.abs(filtered)


# --------------------------------------------------------- texture scoring
class TextureComplexityScorer:
    """sum of 2^-|r| * |r| over each patch: low for smooth, high for textured."""

    def __call__(self, residuals, patch_dim=16):
        depth = residuals.shape[1] if residuals.dim() == 4 else residuals.shape[0]
        abs_res = residuals.abs()

        weights = torch.full((residuals.shape[-2], residuals.shape[-1]), 2)
        weights = weights.repeat(depth, 1, 1).unsqueeze(0)
        weights = torch.pow(weights, -abs_res) * abs_res / patch_dim

        kernel = torch.full((patch_dim, patch_dim), -1, dtype=torch.float32)
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(depth, 1, 1, 1)

        out = F.conv2d(weights, kernel, stride=patch_dim, padding=0, groups=depth, bias=None)
        return (out + 1).squeeze(0)


class Scalers:
    """Registry of patch normalisers: resolve('scaler_name') or
     pass a custom callable scaler function with resolve(function_name)."""

    _registry = {}

    @classmethod
    def resolve(cls, scaler):
        if scaler is None or callable(scaler):
            return scaler
        try:
            return cls._registry[scaler]
        except KeyError:
            raise ValueError(f"unknown scaler {scaler!r}; known: {sorted(cls._registry)}") from None

    #Scalers
    @staticmethod
    def minmax_patches(patches, eps=1e-8):
        pmin = patches.amin(dim=(2, 3), keepdim=True)
        pmax = patches.amax(dim=(2, 3), keepdim=True)
        return ((patches - pmin) / (pmax - pmin).clamp_min(eps)).clamp(0.0, 1.0)

    @staticmethod
    def minmax_global(patches, eps=1e-8):
        gmin, gmax = patches.min(), patches.max()
        return ((patches - gmin) / (gmax - gmin).clamp_min(eps)).clamp(0.0, 1.0)

    @staticmethod
    def log(patches):
        return torch.log1p(patches)

    @staticmethod
    def power(patches, power=0.5, eps=1e-8):
        return torch.sign(patches) * torch.pow(patches.abs() + eps, power)


Scalers._registry.update({"minmax_patches": Scalers.minmax_patches,
                          "minmax_global": Scalers.minmax_global,
                          "log": Scalers.log,
                          "power": Scalers.power})


class PatchRanker:
    """Selects the N least and N most textured patches of a patch stack."""

    def __init__(self, n_patches=64, tc_dim=16, scorer=None):
        self.n_patches = int(n_patches)
        self.tc_dim = int(tc_dim)
        self.scorer = scorer or TextureComplexityScorer()

    def __call__(self, patches, image_filter=None):
        if patches.shape[0] < self.n_patches:
            raise ValueError(f"insufficient patches: need {self.n_patches}, "
                             f"have {patches.shape[0]}")
        residuals = patches if image_filter is None else FilterBank.apply(patches, image_filter)
        scores = self.scorer(residuals, self.tc_dim).flatten(1).sum(dim=1)

        _, order = scores.sort()
        return patches[order[:self.n_patches]], patches[order[-self.n_patches:]]


class MosaicAssembler:
    """Stacks N patches back into one square grid image."""

    def __init__(self, n_patches=64):
        self.side = math.isqrt(int(n_patches))

    def __call__(self, patches):
        side, channels, dims = self.side, patches.shape[1], patches.shape[-1]
        return (patches.view(side, side, channels, dims, dims)
                       .permute(2, 0, 3, 1, 4)
                       .reshape(channels, side * dims, side * dims))


# --------------------------------------------------------------- augmenting
class ImageAugmenter:
    """Flip + random crop >= 256 px. Pass `generator=` for an isolated RNG stream."""

    def __init__(self, prob=0.5, generator=None, min_crop=256):
        self.prob = float(prob)
        self.generator = generator
        self.min_crop = int(min_crop)

    def _uniform(self):
        return torch.rand(1, generator=self.generator).item()

    def __call__(self, image):
        if self._uniform() < self.prob:
            image = torch.flip(image, dims=[-1])          # horizontal flip
        if self._uniform() < self.prob:
            image = torch.flip(image, dims=[-2])          # vertical flip

        img_h, img_w = image.shape[-2:]
        if img_h < self.min_crop or img_w < self.min_crop:
            return image

        if self._uniform() < self.prob:
            crop_h = torch.randint(self.min_crop, img_h + 1, (1,), generator=self.generator).item()
            crop_w = torch.randint(self.min_crop, img_w + 1, (1,), generator=self.generator).item()
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(crop_h, crop_w))
            image = transforms.functional.crop(image, i, j, h, w)

        return image


# ----------------------------------------------------------- feature builder
class FeatureBuilder:
    """image tensor -> {'feature': tensor}."""

    def __init__(self, config, converter=None, filters=None, scorer=None,
                 ranker=None, assembler=None, patches=None, augmenter=None):
        self.cfg = config
        self.converter = converter or ColorSpaceConverter(config.colorspace)
        self.filters = filters or FilterBank(config.spatial_filters, config.fft_filter_size)
        self.scorer = scorer or TextureComplexityScorer()
        self.ranker = ranker or PatchRanker(config.n_patches, config.patch_tc_dim, self.scorer)
        self.assembler = assembler or MosaicAssembler(config.n_patches)
        self.patches = patches or PatchExtractor(config.patch_size)
        self.augmenter = augmenter

    @classmethod
    def from_config(cls, config):
        augmenter = ImageAugmenter() if config.augment else None
        return cls(config, augmenter=augmenter)

    # -- parts ------------------------------------------------------------
    def _mosaic(self, selected, spatial_filter, scaling_fnc):
        """spatial filter -> scaling -> grid, in that order."""
        patches = selected
        if spatial_filter is not None:
            patches = FilterBank.apply(patches, spatial_filter)
        scaler = Scalers.resolve(scaling_fnc)
        if scaler is not None:
            patches = scaler(patches)
        return self.assembler(patches)

    def _residual_features(self, image):
        cfg = self.cfg
        base = resize(image, size=(cfg.img_size, cfg.img_size)) if cfg.resize_img else image
        residual = self.filters.apply(base, cfg.texture_filter)
        out = {}
        if "residuals" in cfg.features:
            out["residuals"] = residual
        if "tc" in cfg.features:
            out["tc"] = self.scorer(residual, cfg.img_size)
        return out

    def _spatial_patch_features(self, patch_stack):
        cfg = self.cfg
        poor, rich = self.ranker(patch_stack, cfg.texture_filter)
        bank = self.filters.bank
        # both mosaics are produced together, exactly like the original pipeline
        return {"poor": self._mosaic(poor, bank, cfg.spatial_patch_scaling_func),
                "rich": self._mosaic(rich, bank, cfg.spatial_patch_scaling_func)}

    def _fft_patch_features(self, patch_stack):
        cfg = self.cfg
        poor, rich = self.ranker(patch_stack, cfg.texture_filter)
        return {"fft_poor": self._mosaic(poor, None, cfg.freq_patch_scaling_func),
                "fft_rich": self._mosaic(rich, None, cfg.freq_patch_scaling_func)}

    # -- entry point -------------------------------------------------------
    def __call__(self, image_tensor):
        cfg = self.cfg
        image = self.converter(image_tensor)
        if self.augmenter is not None:
            image = self.augmenter(image)

        features = {}
        if cfg.needs_residuals:
            features.update(self._residual_features(image))

        prepatch = self.patches.center_crop(image) if cfg.needs_prepatch else None

        if cfg.needs_spatial_patches:
            features.update(self._spatial_patch_features(self.patches.unfold(prepatch)))

        if cfg.needs_fft_patches:
            filtered = self.filters.high_pass(prepatch, cfg.fft_filter_size)
            features.update(self._fft_patch_features(self.patches.unfold(filtered)))

        if "prepatch" in cfg.features:
            if cfg.resize_img:
                prepatch = resize(prepatch, size=(cfg.img_size, cfg.img_size))
            features["prepatch"] = prepatch

        return features


# -------------------------------------------------------------------- source
class ParquetImageSource:
    """Sharded, epoch-shuffled parquet stream yielding (image tensor, label)."""

    IMAGE_COLUMN = "Image"
    LABEL_COLUMNS = {"binary": "Label_A", "multiclass": "Label_B"}

    def __init__(self, filepattern, labeltype="binary", shuffle=False, seed=42):
        if labeltype not in self.LABEL_COLUMNS:
            raise ValueError(f"labeltype must be one of {sorted(self.LABEL_COLUMNS)}")
        self.filepaths = sorted(glob.glob(filepattern))
        if not self.filepaths:
            raise FileNotFoundError(f"no parquet files match {filepattern!r}")
        self.label_column = self.LABEL_COLUMNS[labeltype]
        self.shuffle = bool(shuffle)
        self.seed = int(seed)

    def shard(self, epoch=0, worker_id=0, num_workers=1):
        """This worker's files for `epoch`; shuffled from a local generator only."""
        files = self.filepaths[worker_id::num_workers]
        if not self.shuffle:
            return files
        generator = torch.Generator().manual_seed(self.seed + 1_000_003 * int(epoch) + worker_id)
        order = torch.randperm(len(files), generator=generator).tolist()
        return [files[i] for i in order]

    def __call__(self, files):
        for file in files:
            parquet_file = pq.ParquetFile(file)
            for group in range(parquet_file.num_row_groups):
                table = parquet_file.read_row_group(
                    group, columns=[self.IMAGE_COLUMN, self.label_column])
                frame = table.to_pandas()
                blobs = frame[self.IMAGE_COLUMN].str["bytes"]
                for blob, label in zip(blobs, frame[self.label_column]):
                    image = Image.open(io.BytesIO(blob)).convert("RGB")
                    yield to_tensor(image), torch.tensor(label, dtype=torch.long)


# ------------------------------------------------------------------ dataset
class FeatureDataset(IterableDataset):
    """`DatasetPreprocessing`, split into a source and a feature builder.

    FeatureDataset(**processing_parameters, seed=SEED)          # same kwargs
    yields {'features': {...}, 'label': tensor} like the original.
    """

    def __init__(self, filepattern, features=None, colorspace="RGB", labeltype="binary",
                 augment=False, resize_img=True, img_size=256, n_patches=64,
                 patch_size=32, patch_tc_dim=16, texture_filter=None, fft_filter_size=30,
                 freq_patch_scaling_func="minmax_patches",
                 spatial_patch_scaling_func="minmax_patches", spatial_filters=None,
                 seed=42, shuffle=False, source=None, builder=None):

        self.cfg = ProcessingConfig(filepattern, features, colorspace, labeltype, augment,
                                    resize_img, img_size, n_patches, patch_size, patch_tc_dim,
                                    texture_filter, fft_filter_size, freq_patch_scaling_func,
                                    spatial_patch_scaling_func, spatial_filters, shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.source = source or ParquetImageSource(filepattern, labeltype=labeltype,
                                                   shuffle=shuffle, seed=seed)
        self.builder = builder or FeatureBuilder.from_config(self.cfg)

    def set_epoch(self, epoch):
        """Advance the shuffle permutation. Call once per epoch."""
        self.epoch = int(epoch)

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        num_workers, worker_id = (1, 0) if info is None else (info.num_workers, info.id)
        return self.stream(self.source.shard(self.epoch, worker_id, num_workers))

    def stream(self, files):
        for image, label in self.source(files):
            yield {"features": self.builder(image), "label": label}


# -------------------------------------------------------------------- usage
# from configs.pipeline_config import processing_parameters, SEED
# from pipeline.preprocessing import FeatureDataset
#
# dataset = FeatureDataset(**processing_parameters, seed=SEED)      # drop-in
#
# pieces are usable on their own (no parquet, no workers):
# from pipeline.preprocessing import FeatureBuilder, ProcessingConfig
# builder = FeatureBuilder.from_config(ProcessingConfig(
#     "unused", features=["fft_poor", "fft_rich"], colorspace="Y"))