
"""Make sure the feature cache is current, then run the sweeps in parallel.

1. ensure_cache - compare data/cache/*/meta.json with the live processing
                  parameters; rebuild the caches only if they disagree
2. trial        - one mapped DockerOperator per copy, each running
                  scripts/tune_trial.py unchanged

PARALLEL_RUNS is the only quantity this file decides: how many containers run
at once. Each container keeps running trials until the study reaches its
TARGET_TRIALS, so the work is shared, not one trial per container. Everything
about what a trial does lives in scripts/tune_trial.py.

Deliberately left out:
  - killing a trial that outgrows its memory: mem_limit="3g" already does
    that, and the task simply fails
  - cleaning up a trial left RUNNING by a killed container: scripts/tune_trial.py
    turns Optuna's heartbeat on, so the trial is marked FAIL once
    HEARTBEAT_GRACE seconds pass without a heartbeat and another container
    starts its next trial, so no extra code is needed
"""

import os
from datetime import timedelta

from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sdk import DAG
from docker.types import DeviceRequest, Mount

REPO = os.environ.get("TUNING_REPO") or "/home/Fish/coding/aigen/ai_detection"
IMAGE = "ai-detection-tuner"
NETWORK = "ai_detection_default"        

DATABASE = {"OPTUNA_HOST": "postgres", "OPTUNA_PORT": "5432"}

# How many trials run at the same time. Each container is capped at 3 GiB, so
# keep PARALLEL_RUNS x 3 GiB inside the memory you can spare.
PARALLEL_RUNS = 3

# The check-and-rebuild decision lives in pipeline.utils.ensure_cache and is
# driven by scripts/ensure_cache.py, so the DAG only names a command.

COMMON = dict(
    image=IMAGE,
    network_mode=NETWORK,
    mounts=[Mount(source=REPO, target="/app", type="bind")],
    working_dir="/app",
    device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
    shm_size="2g",
    auto_remove="force",
    mount_tmp_dir=False,                  # do not require a shared /tmp with Airflow
    docker_url="unix://var/run/docker.sock",
)

with DAG(
    dag_id="parallel_tuning",
    description="verify the cache, then run N tuning sweeps in parallel",
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "ml", "retries": 0},
    tags=["tuning"],
) as dag:

    ensure_cache = DockerOperator(
        task_id="ensure_cache",
        command=["python", "scripts/ensure_cache.py"],
        execution_timeout=timedelta(hours=6),
        # rebuilding holds a parquet chunk per worker, so allow more than a trial
        **{**COMMON, "mem_limit": "8g"},
    )

    trials = DockerOperator.partial(
        task_id="trial",
        command=["python", "scripts/tune_trial.py"],
        execution_timeout=timedelta(hours=12),
        # trials run one at a time inside a container, so the 3 GB ceiling is
        # the measured peak of a single trial + margin
        **{**COMMON, "mem_limit": "3g"},
    ).expand(environment=[dict(DATABASE, COPY=str(i)) for i in range(PARALLEL_RUNS)])

    ensure_cache >> trials
