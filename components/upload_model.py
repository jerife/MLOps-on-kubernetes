from functools import partial
from kfp.components import InputPath, create_component_from_func

@partial(
    create_component_from_func,
    packages_to_install=["dill", "pandas", "scikit-learn", "mlflow", "boto3"],
)
def upload_sklearn_model_to_mlflow(
    secret: dict,
    model_name: str,
    model_path: InputPath("dill"),
    feature_extractor_path: InputPath("dill"),
    input_example_path: InputPath("dill"),
    signature_path: InputPath("dill"),
    conda_env_path: InputPath("dill"),
):
    import os
    import dill
    from mlflow.sklearn import save_model
    from mlflow.tracking.client import MlflowClient

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://mlflow-minio.mlflow-system.svc:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = secret["AWS_ACCESS_KEY_ID"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = secret["AWS_SECRET_ACCESS_KEY"]

    client = MlflowClient("http://mlflow-service.mlflow-system.svc:5000")

    with open(model_path, mode="rb") as file_reader:
        clf = dill.load(file_reader)
    with open(feature_extractor_path, mode="rb") as file_reader:
        clf = dill.load(file_reader)
    with open(input_example_path, "rb") as file_reader:
        input_example = dill.load(file_reader)
    with open(signature_path, "rb") as file_reader:
        signature = dill.load(file_reader)
    with open(conda_env_path, "rb") as file_reader:
        conda_env = dill.load(file_reader)

    save_model(
        sk_model=clf,
        path=model_name,
        serialization_format="cloudpickle",
        conda_env=conda_env,
        signature=signature,
        input_example=input_example,
    )
    run = client.create_run(experiment_id="0")
    client.log_artifact(run.info.run_id, model_name)