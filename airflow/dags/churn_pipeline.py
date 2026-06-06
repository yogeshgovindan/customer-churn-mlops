from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    "owner": "mlops",
    "start_date": datetime(2026, 1, 1)
}

with DAG(
    dag_id="customer_churn_pipeline",
    default_args=default_args,
    schedule=None,
    catchup=False
) as dag:

    preprocess = BashOperator(
        task_id="preprocess",
        bash_command="""
cd E:/customer-churn-mlops &&
venv/Scripts/python src/data/preprocess.py
"""
    )

    train = BashOperator(
        task_id="train",
        bash_command="""
cd E:/customer-churn-mlops &&
venv/Scripts/python src/training/train.py
"""
    )

    evaluate = BashOperator(
        task_id="evaluate",
        bash_command="""
cd E:/customer-churn-mlops &&
venv/Scripts/python src/training/evaluate.py
"""
    )

    preprocess >> train >> evaluate
