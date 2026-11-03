"""Run Optuna trials for the ai-detection model until the study is complete.

Everything about a run is configured in the SETTINGS block below:

    python scripts/tune_trial.py

One process keeps running trials, one after another, until the study holds
TARGET_TRIALS trials. Any number of processes may run at the same time: they
all read the same counter in the database, so N of them divide the remaining
trials between themselves and all stop at the same point.
"""
import os

import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from dotenv import load_dotenv
from loguru import logger

from configs.pipeline_config import DATA_DIR, processing_parameters, training_parameters
from models.model import CustomResNet18
from pipeline.preprocessing import CachedFeatureDataset
from pipeline.training import Evaluator, MlflowTracker, NullTracker, Trainer
from pipeline.utils import resolve_experiment

load_dotenv()

SEED = 42
METRIC = "auroc"                 # Metric being optimized

# TUNER SETTINGS
STUDY_NAME = "resnet"            
TARGET_TRIALS = 40               # stop when the study holds this many trials
                                 # if already reached, the process does nothing and exits
EPOCHS = 10                      # epochs per trial
SAMPLES_PER_EPOCH = None         # None = a full pass over the train cache
VAL_SAMPLES = None               # None = the whole validation cache
WORKERS = 2                      # data-loading processes
PRUNE_WARMUP = 3                 # epochs a trial may run before it can be stopped
STARTUP_TRIALS = 5               # trials that always run fully before pruning starts
USE_MLFLOW = True                # False keeps test runs out of Databricks
HEARTBEAT_INTERVAL = 60          # seconds between "still alive" writes per trial
HEARTBEAT_GRACE = 180            # silence (seconds) before a dead trial is FAIL

# STATES which count towards TARGET_TRIALS:
STATES_TRACKED = [TrialState.COMPLETE, TrialState.PRUNED]


class PruningTracker(NullTracker):
    """Reports the epoch score to Optuna, then forwards everything
     to an `inner` tracker.   
    """

    def __init__(self, trial, inner=None, metric=METRIC):
        self.trial = trial
        self.inner = inner if inner is not None else NullTracker()
        self.metric = metric
        self.pruning = False

    @property
    def run_id(self):
        return self.inner.run_id

    def log_epoch(self, epoch, metrics, val=None, extras=None):
        self.inner.log_epoch(epoch, metrics, val, extras)
        score = (val or {}).get(self.metric)
        if score is None:                       # no validation -> fall back to loss
            score = (extras or {}).get("loss")
        if score is None:
            return
        self.trial.report(float(score), step=epoch)
        if self.trial.should_prune():
            self.pruning = True
            logger.info("trial {} pruned at epoch {} ({}={:.4f})",
                        self.trial.number, epoch, self.metric, score)
            raise optuna.TrialPruned()

    def start(self, run_name=None, tags=None):
        try:
            return self.inner.start(run_name, tags)
        except Exception as exc:                # tracker down / bad credentials
            logger.warning("experiment tracking disabled for this trial: {}", exc)
            self.inner = NullTracker()
            return None

    def end(self, status="FINISHED"):
        self.inner.end("KILLED" if self.pruning else status)

    def log_params(self, params, config=None):
        self.inner.log_params(params, config)

    def log_step(self, step, loss):
        self.inner.log_step(step, loss)

    def log_history(self, history, upto=None):
        self.inner.log_history(history, upto)

    def log_artifacts(self, path):
        self.inner.log_artifacts(path)

    def log_state_dict(self, model, name):
        self.inner.log_state_dict(model, name)

    def log_summary(self, tags=None, metrics=None):
        self.inner.log_summary(tags, metrics)


def make_tracker(trial):
    """MLflow (if available) underneath the pruning hook."""
    if not USE_MLFLOW:
        return PruningTracker(trial, NullTracker())
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri is None and os.environ.get("DATABRICKS_HOST"):
        uri = "databricks"
    inner = MlflowTracker(tracking_uri=uri,
                          experiment=STUDY_NAME,   # root directory is added by the tracker
                          run_name=f"{STUDY_NAME}|trial{trial.number}")
    return PruningTracker(trial, inner)


def objective(trial):
    """Train one configuration and return its score for Optuna.

    `tuned` holds the values Optuna chose; everything else comes from
    `configs.pipeline_config` plus the SETTINGS block above.
    """
    # processing / cache configuration
    proc_p = dict(processing_parameters)

    # tuned training hyperparameters examples
    train_tuned = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_int("batch_size", 16, 64, step=16),
    }

    # tuned model hyperparameters
    model_tuned = {
        "model": "resnet_sample",
        "n" :trial.suggest_categorical("n", [4, 7, 9]),
        "use_dropout" : trial.suggest_categorical("use_dropout", [False, True])
    }

    # apply
    train_p = dict(training_parameters)
    train_p.update(train_tuned)
    train_p.update(
        num_epochs=EPOCHS,
        max_samples_per_epoch=SAMPLES_PER_EPOCH,
        dataset_workers=WORKERS,
        checkpoint_dir=f"checkpoints/tuning/{STUDY_NAME}/trial_{trial.number:03d}",
        checkpoint_every=0,
        keep_checkpoints=0,
    )

    # Delays model creation so the seed set in Trainer.build() takes effect
    def tune_custom_resnet():
        return CustomResNet18(in_channels=1, n=model_tuned["n"],
                              res_option="A", use_dropout=model_tuned["use_dropout"])

    trainer = Trainer(
        proc_p, tune_custom_resnet, train_p,
        seed=SEED + trial.number,
        dataset_factory=CachedFeatureDataset.factory(DATA_DIR / "cache/train"),
        validation=Evaluator(
            dataset_factory=CachedFeatureDataset.factory(DATA_DIR / "cache/val"),
            max_samples=VAL_SAMPLES,
        ),

        tracker=make_tracker(trial),
        extra_params=model_tuned,
    )
    return trainer.fit()["best_score"]


def make_storage():
    """Postgresql store, or None when no database can be reached."""
    try:
        dsn = (f"postgresql+psycopg2://{os.environ['OPTUNA_USER']}:"
               f"{os.environ['OPTUNA_PASSWORD']}@"
               f"{os.environ.get('OPTUNA_HOST', 'localhost')}:"
               f"{os.environ.get('OPTUNA_PORT', '5434')}/"
               f"{os.environ['OPTUNA_DB']}")
        storage = optuna.storages.RDBStorage(url=dsn,
                                             heartbeat_interval=HEARTBEAT_INTERVAL,
                                             grace_period=HEARTBEAT_GRACE)
        storage.get_all_studies()        # cheap read: also fails on a broken schema
    except Exception as exc:             # missing keys, no server, wrong password, ...
        logger.warning("no Optuna database ({}: {}); tuning in memory, "
                       "trials are lost on exit and cannot be shared",
                       type(exc).__name__, exc)
        return None
    logger.info("study storage: postgres at {}:{}/{}",
                os.environ.get("OPTUNA_HOST", "localhost"),
                os.environ.get("OPTUNA_PORT", "5434"),
                os.environ["OPTUNA_DB"])
    return storage


def build_study():
    storage = make_storage()
    pruner = optuna.pruners.MedianPruner(n_startup_trials=STARTUP_TRIALS,
                                         n_warmup_steps=PRUNE_WARMUP)
    study = optuna.create_study(study_name=STUDY_NAME, storage=storage,
                                direction="maximize", load_if_exists=True,
                                pruner=pruner)    

    # a study's direction is fixed when it is created: refuse to tune backwards
    if study.direction != optuna.study.StudyDirection.MAXIMIZE:
        raise SystemExit(f"study {STUDY_NAME!r} is stored as {study.direction.name}, "
                         f"but {METRIC} is better when larger - change STUDY_NAME")
    return study


def counted_trials(study):
    """How many trials in the study already count towards TARGET_TRIALS."""
    return len(study.get_trials(deepcopy=False, states=STATES_TRACKED))


def print_best(study):
    """Show the best result so far (a study of only pruned trials has none)."""
    try:
        best = study.best_trial
    except ValueError:
        logger.info("study {} has no completed trial yet", study.study_name)
        return
    print(f"best trial {best.number}: {METRIC}={best.value:.4f} {best.params}")


def main():
    if USE_MLFLOW:
        logger.info("mlflow experiment -> {}", resolve_experiment(STUDY_NAME))

    study = build_study()
    done = counted_trials(study)
    logger.info("study {} | direction {} | {}/{} trials counted ({})",
                study.study_name, study.direction.name, done, TARGET_TRIALS,
                "/".join(state.name.lower() for state in STATES_TRACKED))

    if done >= TARGET_TRIALS:
        logger.info("target reached - this process has nothing to do")
        print_best(study)
        return

    logger.info("running trials, one after another, until the study reaches {}",
                TARGET_TRIALS)
    study.optimize(objective, n_jobs=1,
                   callbacks=[MaxTrialsCallback(TARGET_TRIALS, states=STATES_TRACKED)])
    print_best(study)


if __name__ == "__main__":
    main()
