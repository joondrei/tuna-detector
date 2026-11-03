from datetime import timedelta
import os

from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sdk import DAG
from docker.types import DeviceRequest, Mount

REPO = os.environ.get("TUNING_REPO") or "/home/Fish/coding/aigen/ai_detection"
IMAGE = "ai-detection-tuner"
NETWORK = "ai_detection_default"        

DATABASE = {"OPTUNA_HOST": "postgres", "OPTUNA_PORT": "5432"}

CHECK = (
    "import torch, optuna;"
    "avail = torch.cuda.is_available();"
    "print('torch', torch.__version__);"
    "print('optuna', optuna.__version__);"
    "print('cuda available:', avail);"
    "print('device:', torch.cuda.get_device_name(0) if avail else 'none')"
)

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
    description="check the tuning image, then run one trial",
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
        command=["python", "scripts/tune_trial.py"],
        environment=DATABASE,
        execution_timeout=timedelta(hours=2),
        **COMMON,
    )

    check >> trial
