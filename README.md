repo for my masters thesis, under construction

Done:
full training pipeline featuring patch based noise extraction with options for live preprocessing or cache generation, and training with MLFlow experiment tracking

To do:
Optuna hyperparameter search with parallelization orchestrated by airflow

To setup, run:
uv python install 3.12
uv sync --all-extras          # or: uv sync --extra all

To verify, run scripts

Note: mlflow need .env file in root dir with secrets DATABRICKS_HOST and DATABRICKS_TOKEN
