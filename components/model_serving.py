
from functools import partial
from kfp.components import InputPath, create_component_from_func

@partial(
    create_component_from_func,
    packages_to_install=["dill==0.3.4", "pandas==1.3.5", "numpy==1.21.6", "mne==1.1.0", "scikit-learn==1.0.2", "mlflow==1.28.0", "boto3"],
)
def serving(
    secret: dict,
    model_name: str,
    model_path: InputPath("dill"),
    signature_path: InputPath("dill"),
    conda_env_path: InputPath("dill"),
):
    import os
    import mlflow
    from mlflow.pyfunc import load_model
    from mlflow.tracking import MlflowClient

    """ Load Model """
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://mlflow-minio.mlflow-system.svc:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = secret["AWS_ACCESS_KEY_ID"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = secret["AWS_SECRET_ACCESS_KEY"]
    client = MlflowClient("http://mlflow-service.mlflow-system.svc:5000")

    filter_string = f"name='{model_name}'"
    results = client.search_model_versions(filter_string)

    latest_version = 0
    for res in results: 
        if res.version > latest_version:
            model_uri = res.source

    model = load_model(model_uri)
    
