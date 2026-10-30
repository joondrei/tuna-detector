# =============================================================================
# Class-based trainer, built from small decoupled pieces
#
#   TrainConfig     - validated view of training_parameters (no torch, no io)
#   EarlyStopping   - score bookkeeping only (no model, no disk, no mlflow)
#   CheckpointStore - atomic .pt io only (no training state machine)
#   NullTracker     - no-op tracker, so Trainer never depends on mlflow
#   MlflowTracker   - adapter mapping trainer events onto mlflow
#   NullValidator   - no-op validation, so validation is always optional
#   CallableValidator - adapter for a legacy (eval_fn, eval_loader) pair
#   Evaluator       - owns the val loader, the metrics and the reporting
#   Trainer         - orchestration: build -> resume -> loop -> finalize
#
# Every collaborator is injectable (model factory, dataset factory, metric
# factory, validation, tracker), so each piece can be tested alone.
#
# Depends on .utils (rngs, metrics, config serialising, atomic io),
# .preprocessing (FeatureDataset), loguru (all logging) and, only for
# MlflowTracker, mlflow.
# =============================================================================
import dataclasses
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader

try:                                   # optional: NullTracker needs no mlflow
    import mlflow
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.errors.platform import ResourceDoesNotExist
except ImportError:                    # pragma: no cover
    mlflow = None

from configs.pipeline_config import SEED
from .preprocessing import FeatureDataset
from .utils import (_atomic_torch_save, _flat_params, _jsonable, _resolve_device,
                    _sanitize, fixed_rngs, make_metrics, prune_checkpoints,
                    seed_worker, timeit)

__all__ = ["TrainConfig", "EarlyStopping", "CheckpointStore", "NullTracker",
           "MlflowTracker", "NullValidator", "CallableValidator", "Evaluator",
           "Trainer", "move_batch", "train_step", "update_metrics"]


# ------------------------------------------------------------------ config
@dataclass
class TrainConfig:
    """Validated training parameters. `TrainConfig.from_dict(training_parameters)`."""

    num_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float = 0.0
    pos_weight: Optional[float] = None
    dataset_workers: int = 0
    max_samples_per_epoch: Optional[int] = None
    log_every: int = 0
    grad_clip: Optional[float] = None
    threshold: float = 0.5

    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 1
    keep_checkpoints: int = 5          # how many epoch_*.pt to keep; 0 disables pruning
    monitor: str = "loss"
    monitor_mode: Optional[str] = None
    early_stopping_patience: Optional[int] = None
    min_delta: float = 0.0
    restore_best: bool = True

    resume_from: Optional[str] = None

    def __post_init__(self):
        if self.monitor_mode is None:
            self.monitor_mode = "min" if self.monitor == "loss" else "max"
        if self.monitor_mode not in ("min", "max"):
            raise ValueError(f"monitor_mode must be 'min' or 'max', got {self.monitor_mode!r}")
        if self.num_epochs < 0:
            raise ValueError(f"num_epochs must be >= 0, got {self.num_epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.checkpoint_every is not None and self.checkpoint_every < 0:
            raise ValueError(f"checkpoint_every must be >= 0, got {self.checkpoint_every}")
        if self.keep_checkpoints is not None and self.keep_checkpoints < 0:
            raise ValueError(f"keep_checkpoints must be >= 0, got {self.keep_checkpoints}")

    @classmethod
    def from_dict(cls, params):
        if isinstance(params, cls):
            return params
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(params) - known
        if unknown:
            raise TypeError(f"unknown training parameter(s): {sorted(unknown)}")
        return cls(**params)

    def to_dict(self):
        return dataclasses.asdict(self)

    @property
    def steps_per_epoch(self):
        if self.max_samples_per_epoch is None:
            return None
        return math.ceil(self.max_samples_per_epoch / self.batch_size)


# ---------------------------------------------------------- early stopping
@dataclass
class EarlyStopping:
    """Tracks the monitored score for early stopping."""

    monitor: str = "loss"
    mode: str = "min"
    patience: Optional[int] = None
    min_delta: float = 0.0

    best_score: Optional[float] = None
    best_epoch: int = -1
    no_improve: int = 0

    def is_improvement(self, score, mode=None):
        mode = mode or self.mode
        if self.best_score is None:
            return True
        if mode == "min":
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta

    def update(self, score, epoch, mode=None):
        """Record `score`; returns True when it is the new best."""
        if self.is_improvement(score, mode):
            self.best_score, self.best_epoch, self.no_improve = score, epoch, 0
            return True
        self.no_improve += 1
        return False

    @property
    def should_stop(self):
        return bool(self.patience) and self.no_improve >= self.patience

    def state_dict(self):
        return {"best_score": self.best_score,
                "best_epoch": self.best_epoch,
                "no_improve": self.no_improve}

    def load_state_dict(self, state):
        self.best_score = state.get("best_score", self.best_score)
        self.best_epoch = int(state.get("best_epoch", self.best_epoch))
        self.no_improve = int(state.get("no_improve", self.no_improve))


# ------------------------------------------------------------- checkpoints
class CheckpointStore:
    """_atomic_torch_save provides safe recovery; epoch files are pruned to `keep`."""

    def __init__(self, directory, every=1, keep=5):
        self.dir = Path(directory or "checkpoints")
        self.every = int(every or 0)
        self.keep = keep
        self.dir.mkdir(parents=True, exist_ok=True)

    def prune(self):
        """Drop all but the `keep` newest epoch_*.pt files; best/last always survive."""
        return prune_checkpoints(self.dir, self.keep)

    def path(self, name):
        return self.dir / f"{name}.pt"

    @property
    def best_path(self):
        return self.path("best")

    def due(self, epoch):
        return bool(self.every) and (epoch + 1) % self.every == 0

    def save(self, name, payload):
        path = self.path(name)
        _atomic_torch_save(dict(payload), path)
        return path

    def resolve(self, source):
        path = self.path("last") if str(source) == "auto" else Path(source)
        if not path.exists():
            raise FileNotFoundError(f"resume_from='{source}' -> {path} does not exist")
        return path

    def load(self, source, map_location="cpu"):
        path = self.resolve(source)
        return path, torch.load(path, map_location=map_location, weights_only=True)

    def load_best(self, map_location="cpu"):
        return torch.load(self.best_path, map_location=map_location, weights_only=True)


# ---------------------------------------------------------------- trackers
class NullTracker:
    """No-op tracker: `Trainer` never imports or needs mlflow."""

    run_id = None

    def start(self, run_name=None, tags=None):
        return None

    def end(self, status="FINISHED"):
        return None

    def log_params(self, params, config=None):
        return None

    def log_step(self, step, loss):
        return None

    def log_epoch(self, epoch, metrics, val=None, extras=None):
        return None

    def log_history(self, history, upto=None):
        return None

    def log_artifacts(self, path):
        return None

    def log_state_dict(self, model, name):
        return None

    def log_summary(self, tags=None, metrics=None):
        return None


class MlflowTracker(NullTracker):
    """Adapter: maps trainer events onto mlflow. """

    def __init__(self, tracking_uri=None, experiment=None, run_name=None,
                 log_params=True, log_state_dict=True, log_checkpoints=True,
                 log_history=True, log_system=True, async_logging=True):
        self.tracking_uri = tracking_uri
        self.experiment = experiment
        self.default_run_name = run_name
        self._log_params = log_params
        self._log_state_dict = log_state_dict
        self._log_checkpoints = log_checkpoints
        self._log_history = log_history
        self._log_system = log_system
        self._async_logging = async_logging
        self._run = None
        self.run_id = None

    # -- lifecycle ---------------------------------------------------------
    def start(self, run_name=None, tags=None):
        if mlflow is None:
            raise RuntimeError("MlflowTracker requires mlflow (pip install mlflow); "
                               "use NullTracker to train without tracking")
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
            logger.debug(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")
        if self._async_logging:
            mlflow.config.enable_async_logging(True)
            logger.debug(f"MLflow is set to asynchronously log.")
        if self._log_system:
            mlflow.enable_system_metrics_logging()
            logger.debug(f"MLflow is tracking system metrics.")
        if self.experiment:
            if mlflow.get_tracking_uri() == "databricks":
                w = WorkspaceClient()
                logger.debug("Connected to Databricks")
                exp_dir = "/".join(self.experiment.split("/")[:-1])
                try:
                    w.workspace.get_status(exp_dir)
                    logger.debug(f"Directory already exists: {exp_dir}")
                except ResourceDoesNotExist:
                    w.workspace.mkdirs(exp_dir)
                    logger.success(f"Directory created: {exp_dir}")
                except Exception as e:
                    logger.error(f"Error creating experiment directory")
                    raise e
                finally:
                    mlflow.set_experiment(self.experiment)
                    logger.success(f"MLflow experiment ready: {self.experiment}")
            else:
                mlflow.set_experiment(self.experiment)
        self._run = mlflow.start_run(run_name=run_name or self.default_run_name)
        self.run_id = self._run.info.run_id
        if tags:
            mlflow.set_tags(tags)
        logger.info("mlflow run {} | {}", self.run_id, run_name or self.default_run_name)
        return self.run_id

    def end(self, status="FINISHED"):
        if self._run is not None:
            mlflow.end_run(status=status)
            self._run = None

    # -- logging -----------------------------------------------------------
    def log_params(self, params, config=None):
        if not self._log_params:
            return
        mlflow.log_params(params)                    # scalars only, by contract
        if config is not None:
            mlflow.log_dict(_jsonable(config), "config.json")

    def log_step(self, step, loss):
        mlflow.log_metric("step_loss", float(loss), step=step)

    def log_epoch(self, epoch, metrics, val=None, extras=None):
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=epoch)
        if val:
            mlflow.log_metrics({f"val_{k}": float(v) for k, v in val.items()}, step=epoch)
        for key, value in (extras or {}).items():
            mlflow.log_metric(key, float(value), step=epoch)

    def log_history(self, history, upto=None):
        """Re-log restored epochs so a resumed run has complete curves."""
        if not self._log_history:
            return
        losses = history.get("loss") or []
        upto = len(losses) if upto is None else min(int(upto), len(losses))
        val_hist = history.get("val") or []
        for e in range(upto):
            mlflow.log_metric("loss", float(losses[e]), step=e + 1)
            if e < len(history.get("metrics") or []):
                mlflow.log_metrics({k: float(v) for k, v in history["metrics"][e].items()},
                                   step=e + 1)
            if e < len(val_hist) and isinstance(val_hist[e], dict):
                mlflow.log_metrics({f"val_{k}": float(v) for k, v in val_hist[e].items()},
                                   step=e + 1)
            if e < len(history.get("epoch_time") or []):
                mlflow.log_metric("epoch_time", float(history["epoch_time"][e]), step=e + 1)
            if e < len(history.get("epoch_prep_time") or []):
                mlflow.log_metric("epoch_prep_time", float(history["epoch_prep_time"][e]), step=e + 1)
            if e < len(history.get("epoch_train_time") or []):
                mlflow.log_metric("epoch_train_time", float(history["epoch_train_time"][e]), step=e + 1)
            if e < len(history.get("samples") or []):
                mlflow.log_metric("samples", float(history["samples"][e]), step=e + 1)
        if upto:
            logger.info("replayed {} historical epoch(s) into run {}", upto, self.run_id)

    def log_artifacts(self, path):
        if not self._log_checkpoints:
            return
        try:
            mlflow.log_artifacts(str(path), artifact_path="checkpoints")
        except Exception as exc:
            logger.warning("log_artifacts('{}') failed: {}", path, exc)

    def log_state_dict(self, model, name):
        if not self._log_state_dict:
            return
        try:
            mlflow.pytorch.log_state_dict(model.state_dict(), name)
        except Exception as exc:
            logger.warning("log_state_dict('{}') failed: {}", name, exc)

    def log_summary(self, tags=None, metrics=None):
        if tags:
            mlflow.set_tags(tags)
        if metrics:
            mlflow.log_metrics(metrics)


# -------------------------------------------------------------- validation
class NullValidator:
    """No-op validation:

    `Trainer(..., validation=None)` gets this, so `fit()` trains without
    performing any evaluation on a validation dataset.
    """

    name = "none"

    def build_loader(self, proc_p, seed=42):
        return None

    def __call__(self, model):
        return None


class CallableValidator(NullValidator):
    """ legacy adaptor for `(eval_fn, eval_loader)`. 
        eval_fn is a function that evaluates the model on
        dataset generated from eval_loader and returns a dict
        of metric results.
    """

    name = "callable"

    def __init__(self, eval_fn, eval_loader, name="callable"):
        self.eval_fn = eval_fn
        self.eval_loader = eval_loader
        self.loader = eval_loader
        self.name = name

    def build_loader(self, proc_p, seed=42):
        return self.eval_loader

    def __call__(self, model):
        return self.eval_fn(model, self.eval_loader)


class Evaluator(NullValidator):
    """Validation component: combines loader and evaluation and returns metrics.

    Evaluator()                                  # dataloader configured by proc_p
    Evaluator(max_samples=300)                   # cheap sanity checks
    Evaluator(dataset_factory=FeatureDataset)    # Preprocessing pipeline used by dataloader
    Evaluator(loader=my_loader)                  # custom dataloader
    """

    name = "val"

    def __init__(self, loader=None, filepattern=None, batch_size=64, workers=0,
                 max_samples=None, threshold=0.5, criterion=None,
                 dataset_factory=FeatureDataset, metric_factory=None,
                 pin_memory=None, device=None):
        self.loader = loader
        self.filepattern = filepattern
        self.batch_size = int(batch_size)
        self.workers = int(workers)
        self.max_samples = max_samples
        self.threshold = float(threshold)
        self.criterion = criterion
        self.dataset_factory = dataset_factory
        self.metric_factory = metric_factory
        self.pin_memory = pin_memory
        self.device = device
        self.dataset = None

    # -- loader ------------------------------------------------------------
    def resolve_filepattern(self, proc_p):
        """Explicit `filepattern`, else the training pattern with train -> val."""
        if self.filepattern:
            return self.filepattern
        pattern = str(proc_p.get("filepattern", ""))
        derived = pattern.replace("train", "val")
        if derived == pattern:
            raise ValueError("Evaluator: cannot derive a validation pattern from "
                             f"{pattern!r}; pass filepattern=... explicitly")
        return derived

    def build_loader(self, proc_p, seed=42):
        """Build (once) a val loader from the training processing config."""
        if self.loader is not None:
            return self.loader

        eval_p = dict(proc_p)
        eval_p["filepattern"] = self.resolve_filepattern(proc_p)
        eval_p["augment"] = False          
        eval_p["shuffle"] = False
        self.dataset = self.dataset_factory(**eval_p, seed=seed)
        self.loader = DataLoader(
            self.dataset, batch_size=self.batch_size, num_workers=self.workers,
            pin_memory=(torch.cuda.is_available() if self.pin_memory is None
                        else bool(self.pin_memory)))
        return self.loader

    # -- evaluation --------------------------------------------------------
    def metrics(self, device):
        if self.metric_factory is not None:
            return self.metric_factory(device, self.threshold)
        return make_metrics(device, self.threshold)

    def __call__(self, model):
        if self.loader is None:
            raise RuntimeError("Evaluator: no loader; call build_loader(proc_p) or pass "
                               "loader=... to the constructor")

        device = torch.device(self.device) if self.device else next(model.parameters()).device
        criterion = self.criterion or nn.BCEWithLogitsLoss()
        metrics = self.metrics(device)
        batch_size = int(getattr(self.loader, "batch_size", self.batch_size))

        was_training = model.training
        model.eval()

        total_loss, total = 0.0, 0
        steps = None if self.max_samples is None else math.ceil(self.max_samples / batch_size)

        with torch.no_grad():
            for step, batch in enumerate(self.loader):
                inputs, labels = move_batch(batch, device)
                logits = model(*inputs).reshape(-1)

                total_loss += criterion(logits, labels).item() * labels.numel()
                total += labels.numel()
                update_metrics(metrics, torch.sigmoid(logits), labels)

                if steps is not None and (step + 1) >= steps:
                    break

        if was_training:
            model.train()
        if total == 0:
            raise RuntimeError("Evaluator: loader yielded no samples")

        results = {"loss": total_loss / total}
        results.update({k: m.compute().item() for k, m in metrics.items()})
        logger.info("val | {} | {} samples",
                    " | ".join(f"{k} {v:.4f}" for k, v in results.items()), total)
        return results


# ------------------------------------------------------- decoupled helpers
def move_batch(batch, device):
    """DataLoader batch -> (inputs, labels). Caches to device memory"""
    inputs = [feat.to(device, non_blocking=True) for feat in batch["features"].values()]
    labels = batch["label"].to(device).float()
    return inputs, labels


def train_step(model, criterion, optimizer, inputs, labels, grad_clip=None):
    """One optimisation step: returns (loss, probabilities). """
    optimizer.zero_grad(set_to_none=True)
    logits = model(*inputs).reshape(-1)
    loss = criterion(logits, labels)
    loss.backward()
    if grad_clip is not None:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return loss.detach(), torch.sigmoid(logits.detach())


def update_metrics(metrics, probs, labels):
    for metric in metrics.values():
        metric.update(probs.float(), labels.int())


# ----------------------------------------------------------------- trainer
class Trainer:
    """Orchestrates training from injectable, independently testable parts.

    trainer = Trainer(processing_parameters, build_model, training_parameters,
                      seed=SEED, validation=Evaluator(),
                      tracker=MlflowTracker(MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT))
    result = trainer.fit()

    validation:
        Evaluator(...)   built-in validation (loader derived from processing_parameters)
        NullValidator()  or validation=None -> no validation at all (no-op)
        CallableValidator(eval_fn, eval_loader) -> wrap an existing pair
        (legacy `eval_fn=` / `eval_loader=` kwargs are wrapped automatically)

    Swapping `tracker=NullTracker()` removes every mlflow call; swapping
    `dataset_factory` / `metric_factory` removes the preprocessing / metric
    dependency. `fit()` is the only stateful entry point.
    """

    def __init__(self, processing, model, training, seed=SEED,
                 eval_fn=None, eval_loader=None, validation=None, tracker=None,
                 dataset_factory=FeatureDataset,
                 metric_factory=make_metrics,
                 device=None):
        self.proc = dict(processing)
        self.cfg = TrainConfig.from_dict(training)
        self.model_factory = model
        self.seed = int(seed)
        self.validation = self._resolve_validation(validation, eval_fn, eval_loader)
        self.tracker = tracker if tracker is not None else NullTracker()
        self.dataset_factory = dataset_factory
        self.metric_factory = metric_factory
        self.device = torch.device(device) if device else _resolve_device()

        # populated by build() / restore() / fit()
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.dataset = None
        self.dataloader = None
        self.store = None
        self.stopper = None
        self.gen = None

        self.history = {}
        self.start_epoch = 0
        self.global_step = 0
        self.resumed_from = None
        self.restored_best = False
        self.effective_monitor = self.cfg.monitor
        self.last = None

    @staticmethod
    def _resolve_validation(validation, eval_fn, eval_loader):
        """Explicit validator wins; legacy pair is wrapped; otherwise no-op."""
        if validation is not None:
            return validation
        if eval_fn is not None and eval_loader is not None:
            return CallableValidator(eval_fn, eval_loader)
        return NullValidator()

    # ------------------------------------------------------------- describe
    def run_name(self):
        cfg = self.cfg
        return "{}|{}|bs{}|lr{:g}|seed{}".format(
            "+".join(str(f) for f in self.proc.get("features", [])),
            self.proc.get("colorspace"), cfg.batch_size, cfg.learning_rate, self.seed)

    def tags(self):
        return {"features": ",".join(str(f) for f in self.proc.get("features", [])),
                "colorspace": str(self.proc.get("colorspace")),
                "labeltype": str(self.proc.get("labeltype")),
                "augment": str(self.proc.get("augment")),
                "monitor": str(self.cfg.monitor),
                "monitor_mode": str(self.cfg.monitor_mode),
                "validation": str(getattr(self.validation, "name", "custom")),
                "resumed_from": str(self.cfg.resume_from or "none")}

    def params(self):
        params = {"seed": self.seed}
        params.update(_flat_params(self.proc, prefix="proc."))
        params.update(_flat_params(self.cfg.to_dict(), prefix="train."))
        return params

    def config_snapshot(self):
        return {"processing": _sanitize(self.proc),
                "training": _sanitize(self.cfg.to_dict()),
                "seed": self.seed}

    # ---------------------------------------------------------------- build
    @timeit
    def build(self):
        cfg = self.cfg
        self.gen = fixed_rngs(self.seed)

        self.model = self.model_factory if isinstance(self.model_factory, nn.Module) \
            else self.model_factory()
        self.model.to(self.device)

        pos_weight = cfg.pos_weight
        if pos_weight is not None:
            pos_weight = torch.as_tensor(pos_weight, dtype=torch.float32, device=self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate,
                                    weight_decay=cfg.weight_decay)

        self.dataset = self.dataset_factory(**self.proc, seed=self.seed)
        self.dataloader = self._make_loader(self.dataset)
        self.validation.build_loader(self.proc, self.seed)   # no-op for NullValidator

        self.store = CheckpointStore(cfg.checkpoint_dir, cfg.checkpoint_every,
                                     cfg.keep_checkpoints)
        self.stopper = EarlyStopping(cfg.monitor, cfg.monitor_mode,
                                     cfg.early_stopping_patience, cfg.min_delta)
        self.history = {"loss": [], "metrics": [], "epoch_time": [], "epoch_train_time": [], "epoch_prep_time": [],
                         "samples": [], "best": {}, "stopped_early": False}
        return self

    @timeit
    def _make_loader(self, dataset):
        cfg = self.cfg
        kwargs = dict(batch_size=cfg.batch_size, num_workers=cfg.dataset_workers,
                      pin_memory=(self.device.type == "cuda"))
        if cfg.dataset_workers > 0:            # reproducible worker RNG / augmentation
            kwargs.update(generator=self.gen, worker_init_fn=seed_worker)
        return DataLoader(dataset, **kwargs)

    # --------------------------------------------------------------- resume
    @timeit
    def restore(self):
        """Load model/optimizer/RNG/history; returns True when a checkpoint was used."""
        cfg = self.cfg
        if not cfg.resume_from:
            return False

        path, state = self.store.load(cfg.resume_from, map_location=self.device)

        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])

        self.start_epoch = int(state["epoch"]) + 1
        self.global_step = int(state.get("global_step", 0))
        self.stopper.load_state_dict(state)
        if isinstance(state.get("history"), dict):
            self.history = state["history"]
        self.history["stopped_early"] = False       # training continues

        rng = state.get("rng") or {}
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"].cpu())
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])

        self.resumed_from = str(path)
        self._warn_config_drift(state)
        logger.info("resumed from {} | last epoch {} | best {}={} | continuing at epoch {}"
                    " | no_improve={}", path, state["epoch"] + 1, cfg.monitor,
                    self.stopper.best_score, self.start_epoch + 1, self.stopper.no_improve)
        return True

    def _warn_config_drift(self, state):
        saved = state.get("config") or {}
        old_feats = (saved.get("processing") or {}).get("features")
        if old_feats is not None and list(old_feats) != list(self.proc.get("features", [])):
            logger.warning("features changed: checkpoint={} now={}",
                           old_feats, self.proc.get("features"))
        if saved.get("seed") not in (None, self.seed):
            logger.warning("checkpoint seed {} != seed {} "
                           "(dataset order/augmentation will differ)",
                           saved.get("seed"), self.seed)
        if state.get("monitor") not in (None, self.cfg.monitor):
            logger.warning("monitor changed: checkpoint={} now={}",
                           state.get("monitor"), self.cfg.monitor)
        prev_run = state.get("mlflow_run_id")
        if prev_run and self.tracker.run_id and prev_run != self.tracker.run_id:
            logger.warning("checkpoint was written by run {}; this run ({}) replays "
                           "its history", prev_run, self.tracker.run_id)

    # ---------------------------------------------------------------- epoch
    def train_one_epoch(self, epoch):
        cfg = self.cfg
        self.dataset.set_epoch(epoch)          # reshuffle data sources per worker
        self.model.train()
        metrics = self.metric_factory(self.device, cfg.threshold)

        running_loss, seen = 0.0, 0
        started = time.perf_counter()
        epoch_prep_time = 0.0
        epoch_train_time = 0.0

        prep_start_time = time.perf_counter()
        for step, batch in enumerate(self.dataloader):
            prep_time = time.perf_counter() - prep_start_time
            epoch_prep_time += prep_time
            train_start_time = time.perf_counter()
            inputs, labels = move_batch(batch, self.device)

            loss, probs = train_step(self.model, self.criterion, self.optimizer,
                                     inputs, labels, cfg.grad_clip)
            self.global_step += 1
            update_metrics(metrics, probs, labels)

            n = labels.numel()
            running_loss += loss.item() * n
            seen += n

            train_time = time.perf_counter() - train_start_time
            epoch_train_time += train_time

            if cfg.log_every and step % cfg.log_every == 0:
                self.tracker.log_step(self.global_step, loss.item())
                logger.info("step {:4d} | loss {:.4f} | acc {:.4f} | auroc {:.4f} | {:.2f}s prep time | {:.2f}s train time",
                            step, loss.item(), metrics["accuracy"].compute().item(),
                            metrics["auroc"].compute().item(), prep_time, train_time)

            prep_start_time = time.perf_counter()

            if cfg.steps_per_epoch is not None and (step + 1) >= cfg.steps_per_epoch:
                break

        if seen == 0:
            raise RuntimeError("Trainer.train_one_epoch(): dataloader yielded no samples")

        return {"loss": running_loss / seen,
                "metrics": {k: m.compute().item() for k, m in metrics.items()},
                "samples": seen,
                "epoch_time": time.perf_counter() - started,
                "epoch_prep_time": epoch_prep_time,
                "epoch_train_time": epoch_train_time
                }

    @timeit
    def validate(self):
        """Delegate to the validator; NullValidator returns None (no-op)."""
        return self.validation(self.model)

    def score(self, train_loss, val):
        """(score, used_val) for the monitored key."""
        used_val = val is not None and self.cfg.monitor in val
        return (float(val[self.cfg.monitor]) if used_val else float(train_loss)), used_val

    def _payload(self, epoch, epoch_metrics, val, score):
        return {
            "epoch": epoch,
            "global_step": self.global_step,
            "score": score,
            "monitor": self.cfg.monitor,
            "best_score": self.stopper.best_score,
            "best_epoch": self.stopper.best_epoch,
            "no_improve": self.stopper.no_improve,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epoch_metrics": epoch_metrics,
            "val": val,
            "history": self.history,
            "config": self.config_snapshot(),
            "mlflow_run_id": self.tracker.run_id,
            "rng": {"torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all()
                            if torch.cuda.is_available() else None},
        }

    @timeit
    def _record(self, epoch, train, val):
        self.history["loss"].append(train["loss"])
        self.history["metrics"].append(train["metrics"])
        self.history["epoch_time"].append(train["epoch_time"])
        self.history["epoch_prep_time"].append(train["epoch_prep_time"])
        self.history["epoch_train_time"].append(train["epoch_train_time"])
        self.history["samples"].append(train["samples"])


        if val is not None:                   # keep the val series aligned with epochs
            self.history.setdefault("val", []).append(val)

        line = ("epoch {} | loss {:.4f} | {} | {} samples | {:.1f}s".format(
            epoch + 1, train["loss"],
            " | ".join(f"{k} {v:.4f}" for k, v in train["metrics"].items()),
            train["samples"], train["epoch_time"], train["epoch_prep_time"],
            train["epoch_train_time"]))
        if val is not None:
            line += " || val " + " | ".join(f"{k} {v:.4f}" for k, v in val.items())
        logger.info(line)

        self.tracker.log_epoch(epoch + 1, train["metrics"], val, {
            "loss": train["loss"],
            "epoch_time": train["epoch_time"],
            "epoch_prep_time": train["epoch_prep_time"],
            "epoch_train_time": train["epoch_train_time"],
            "samples": float(train["samples"]),
            "lr": self.optimizer.param_groups[0]["lr"],
            "global_step": float(self.global_step),
        })

    @timeit
    def _checkpoint(self, epoch, train, val, score):
        """Checkpoint + best model + early stopping. Returns True to stop."""
        cfg = self.cfg
        payload = self._payload(epoch, train["metrics"], val, score)

        if self.store.due(epoch):
            # filename carries the same 1-based label as the logs and mlflow steps
            path = self.store.save(f"epoch_{epoch + 1:03d}", payload)
            logger.info("checkpoint -> {}", path)
            pruned = self.store.prune()
            if pruned:
                logger.info("pruned {} old checkpoint(s): {}", len(pruned),
                            ", ".join(p.name for p in pruned))

        if self.stopper.update(score, epoch, mode=cfg.monitor_mode):
            self.history["best"] = {"epoch": epoch, "score": score, "monitor": cfg.monitor}
            path = self.store.save("best", payload)
            logger.info("best model -> {}  ({}={:.4f})", path, cfg.monitor, score)
        elif self.stopper.patience:
            logger.info("no improvement for {}/{} epoch(s)",
                        self.stopper.no_improve, self.stopper.patience)

        if self.stopper.should_stop:
            self.history["stopped_early"] = True
            logger.info("early stopping at epoch {} | best epoch {} ({}={:.4f})",
                        epoch + 1, self.stopper.best_epoch + 1, cfg.monitor,
                        self.stopper.best_score)
            return True
        return False

    # ------------------------------------------------------------------ fit
    @timeit
    def fit(self):
        if self.model is None:
            self.build()

        cfg = self.cfg
        self.tracker.start(self.run_name(), self.tags())
        try:
            self.tracker.log_params(self.params(), self.config_snapshot())

            if self.restore():
                self.tracker.log_history(self.history, upto=self.start_epoch)

            if self.start_epoch >= cfg.num_epochs:
                logger.info("nothing to train: next epoch {} > num_epochs {}",
                            self.start_epoch + 1, cfg.num_epochs)

            # `epoch` is the internal 0-based index: it is the key into
            # `history`, the checkpoint payload and the epoch_NNN.pt filename.
            # Everything a human or the MLflow UI sees is rendered as epoch + 1.
            for epoch in range(self.start_epoch, cfg.num_epochs):
                train = self.train_one_epoch(epoch)
                val = self.validate()

                self._record(epoch, train, val)

                score, used_val = self.score(train["loss"], val)
                self.effective_monitor = cfg.monitor if used_val else "loss"
                if val is not None and not used_val:
                    logger.warning("monitor '{}' not in val={}; falling back to train loss",
                                   cfg.monitor, sorted(val))

                self.last = dict(epoch=epoch, metrics=train["metrics"],
                                 val=val, score=score)

                if self._checkpoint(epoch, train, val, score):
                    break

            self._finalize()
            self.tracker.log_summary(self.summary_tags(), self.summary_metrics())
        except BaseException:
            self.tracker.end("FAILED")
            raise
        self.tracker.end("FINISHED")
        return self.result()

    @timeit
    def _finalize(self):
        cfg = self.cfg

        if self.last is not None:
            self.store.save("last", self._payload(self.last["epoch"], self.last["metrics"],
                                                  self.last["val"], self.last["score"]))

        self.tracker.log_artifacts(self.store.dir)

        self.model.eval()
        self.tracker.log_state_dict(self.model, "model_last")

        if cfg.restore_best and self.store.best_path.exists():
            state = self.store.load_best(self.device)
            self.model.load_state_dict(state["model_state"])
            self.model.eval()
            self.restored_best = True
            logger.info("restored best weights from {} (epoch {}, {}={:.4f})",
                        self.store.best_path, state["epoch"] + 1, cfg.monitor, state["score"])
            self.tracker.log_state_dict(self.model, "model_best")

    def summary_tags(self):
        return {"effective_monitor": self.effective_monitor,
                "validation": str(getattr(self.validation, "name", "custom")),
                "best_epoch": str(self.stopper.best_epoch + 1
                                  if self.stopper.best_epoch >= 0 else -1),
                "stopped_early": str(self.history.get("stopped_early", False)),
                "restored_best": str(self.restored_best),
                "start_epoch": str(self.start_epoch),
                "resumed_from": str(self.resumed_from or "none"),
                "checkpoint_dir": str(self.store.dir)}

    def summary_metrics(self):
        if self.stopper.best_score is None:
            return {}
        return {"best_score": float(self.stopper.best_score)}

    def result(self):
        return {"model": self.model,
                "optimizer": self.optimizer,
                "history": self.history,
                "best_epoch": self.stopper.best_epoch,
                "best_score": self.stopper.best_score,
                "stopped_early": self.history.get("stopped_early", False),
                "restored_best": self.restored_best,
                "start_epoch": self.start_epoch,
                "resumed_from": self.resumed_from,
                "checkpoint_dir": str(self.store.dir),
                "effective_monitor": self.effective_monitor,
                "validation": getattr(self.validation, "name", "custom"),
                "mlflow_run_id": self.tracker.run_id}


# ------------------------------------------------------------------- usage
# validation with the built-in evaluator (val loader derived from processing_parameters)
# trainer = Trainer(processing_parameters, build_model, training_parameters, seed=SEED,
#                   dataset_factory=FeatureDataset,
#                   validation=Evaluator(dataset_factory=FeatureDataset))
# result = trainer.fit()

# legacy pair still accepted (wrapped in CallableValidator):
# Trainer(..., eval_fn=evaluate_model, eval_loader=val_loader).fit()

# resume (model, optimizer, RNG, history and early-stopping state all come
# from the checkpoint, so a fresh Trainer simply continues):
# training_parameters["num_epochs"] += 5
# trainer = Trainer(processing_parameters, build_model,
#                   dict(training_parameters, resume_from="auto"), seed=SEED,
#                   validation=Evaluator(), tracker=MlflowTracker(
#                       MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT))
# result = trainer.fit()