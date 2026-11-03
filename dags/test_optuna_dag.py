from datetime import timedelta
import os

from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sdk import DAG
from docker.types import DeviceRequest, Mount
from dotenv import load_dotenv

# dotenv exposes `load_dotenv` (there is no `load_env`). The repo is mounted at
# /opt/project inside the airflow containers, not at its host path, so the file
# has to be named: a bare load_dotenv() searches from the DAG and finds nothing.
load_dotenv("/opt/project/.env")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE = "ai-detection-tuner"
NETWORK = "ai_detection_default"        

CHECK = (
    "import torch, optuna;"
    "avail = torch.cuda.is_available();"
    "print('torch', torch.__version__);"
    "print('optuna', optuna.__version__);"
    "print('cuda available:', avail);"
    "print('device:', torch.cuda.get_device_name(0) if avail else 'none')"
)

SMOKE = """\
import os
import uuid

import optuna
from dotenv import load_dotenv

load_dotenv("/app/.env")

dsn = (f"postgresql+psycopg2://{os.environ['OPTUNA_USER']}:"
        f"{os.environ['OPTUNA_PASSWORD']}@"
        f"{os.environ.get('OPTUNA_HOST', 'localhost')}:"
        f"{os.environ.get('OPTUNA_PORT', '5434')}/"
        f"{os.environ['OPTUNA_DB']}")


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


storage = optuna.storages.RDBStorage(url=dsn, heartbeat_interval=60, grace_period=180)
storage.get_all_studies()                      # fail here if the db is unreachable

name = "smoke_" + uuid.uuid4().hex[:8]         # throwaway, removed again below
study = optuna.create_study(study_name=name, storage=storage, direction="minimize")
study.optimize(objective, n_trials=5)

# read it back through a second connection: proves the rows really reached postgres
reloaded = optuna.load_study(study_name=name, storage=optuna.storages.RDBStorage(url=dsn))
print("study", name, "->", len(reloaded.trials), "trials, best value", reloaded.best_value)
print("database:", os.environ["OPTUNA_HOST"], os.environ["OPTUNA_PORT"], os.environ["OPTUNA_DB"])

optuna.delete_study(study_name=name, storage=storage)
print("deleted", name)
"""

COMMON = dict(
    image=IMAGE,
    network_mode=NETWORK,
    mounts=[Mount(source=REPO, target="/app", type="bind")],
    working_dir="/app",
    device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
    shm_size="2g",
    mem_limit="3g",
    auto_remove="force",
    mount_tmp_dir=False,                  # do not require a shared /tmp with Airflow
    docker_url="unix://var/run/docker.sock",
)

with DAG(
    dag_id="test_optuna_container",
    description="check the tuning image, then run a tiny study against the database",
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "ml", "retries": 0},
    tags=["tuning", "test"],
) as dag:
    check = DockerOperator(
        task_id="check_environment",
        command=["python", "-c", CHECK],
        **COMMON,
    )

    trial = DockerOperator(
        task_id="one_trial",
        command=["python", "-c", SMOKE],
        environment={"OPTUNA_HOST": "postgres", "OPTUNA_PORT": "5432"},
        execution_timeout=timedelta(minutes=10),
        **COMMON,
    )

    check >> trial
