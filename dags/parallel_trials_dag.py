import os
from datetime import timedelta

from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.sdk import TriggerRule
from airflow.sdk import DAG
from docker.types import DeviceRequest, Mount

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE = "ai-detection-tuner"
NETWORK = "ai_detection_default"        

DATABASE = {"OPTUNA_HOST": "postgres", "OPTUNA_PORT": "5432"}

# Balance with ram usage
PARALLEL_RUNS = 2
DATALOADER_WORKERS = 2

COMMON = dict(
    image=IMAGE,
    network_mode=NETWORK,
    mounts=[Mount(source=REPO, target="/app", type="bind")],
    working_dir="/app",
    device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
    shm_size="2g",
    auto_remove="force",
    mount_tmp_dir=False,                 
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

    verify_cache = DockerOperator(
        task_id="verify_cache",
        command=["python", "scripts/verify_cache.py"],
        do_xcom_push=True,
        **{**COMMON, "mem_limit": "2g"},
    )

    def parse_cache_boolean(**kwargs):
        ti = kwargs['ti']
        raw_output = ti.xcom_pull(task_ids='verify_cache')
        
        last = (raw_output or "").strip().splitlines()[-1] if raw_output else ""
        compatible = last.rsplit(":", 1)[-1].strip() == "True"
        return "skip_rebuild" if compatible else "build_cache"

    branch = BranchPythonOperator(
        task_id="branch",
        python_callable=parse_cache_boolean,
    )

    build_cache = DockerOperator(
        task_id="build_cache",
        command=["python", "scripts/build_dataset_cache.py", "--overwrite",
                    "--workers", DATALOADER_WORKERS ],
        execution_timeout=timedelta(hours=6),
        **{**COMMON, "mem_limit": "8g"},
    )

    skip_rebuild = EmptyOperator(task_id="skip_rebuild")

    trials = DockerOperator.partial(
        task_id="trial",
        command=["python", "scripts/tune_trial.py"],
        execution_timeout=timedelta(hours=12),
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
        **{**COMMON, "mem_limit": "3g"},
    ).expand(environment=[dict(DATABASE) for i in range(PARALLEL_RUNS)])

    verify_cache >> branch >> [build_cache, skip_rebuild]
    build_cache >> trials
    skip_rebuild >> trials