from functools import partial
from kfp.components import InputPath, create_component_from_func

@partial(
    create_component_from_func,
    packages_to_install=["dill==0.3.4", "pandas==1.3.5", "numpy==1.21.6", "mne==1.1.0", "scikit-learn==1.0.2", "mlflow==1.28.0", "boto3"],
)
def upload_sklearn_model_to_mlflow(
    secret: dict,
    model_name: str,
    model_path: InputPath("dill"),
    signature_path: InputPath("dill"),
    conda_env_path: InputPath("dill"),
):
    import os
    import dill
    from mlflow.pyfunc import save_model
    from mlflow.tracking.client import MlflowClient
    
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://mlflow-minio.mlflow-system.svc:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = secret["AWS_ACCESS_KEY_ID"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = secret["AWS_SECRET_ACCESS_KEY"]

    client = MlflowClient("http://mlflow-service.mlflow-system.svc:5000")

    with open(model_path, mode="rb") as file_reader:
        model = dill.load(file_reader)
    with open(signature_path, "rb") as file_reader:
        signature = dill.load(file_reader)
    with open(conda_env_path, "rb") as file_reader:
        conda_env = dill.load(file_reader)

    save_model(
        python_model=model,
        path=model_name,
        signature=signature,
        conda_env=conda_env,
    )
    run = client.create_run(experiment_id="0")
    client.log_artifact(run.info.run_id, model_name)
        