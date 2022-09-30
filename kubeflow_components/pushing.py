from functools import partial
from kfp.components import OutputPath, create_component_from_func    

@partial(
    create_component_from_func,
    base_image="jerife/bentoml_serve:v0.9"
)
def push_to_bentoml(
    secret: dict,
):
    import os
    from mlflow.pyfunc import load_model
    from mlflow.tracking import MlflowClient
    import boto3

    """Load model from mlflow"""
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://mlflow-minio.mlflow-system.svc:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = secret["AWS_ACCESS_KEY_ID"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = secret["AWS_SECRET_ACCESS_KEY"]
    client = MlflowClient("http://mlflow-service.mlflow-system.svc:5000")

    filter_string = "name='BCI-Model'"
    results = client.search_model_versions(filter_string)

    latest_version = 0
    for res in results: 
        if int(res.version) > int(latest_version):
            latest_version=res.version
            model_uri = res.source
    
    mlflow_model = load_model(model_uri)
    
    
    """Push model to bentoml"""
    import bentoml
    bentoml.picklable_model.save_model(
        name="bci_clf",
        model=mlflow_model,
    )
    
    api_token = secret["BENTOML"]["API_TOKEN"]
    ip = secret["BENTOML"]["IP"]
    os.system(f"bentoml yatai login --api-token {api_token} --endpoint {ip}")
    os.system(f"bentoml build")
    os.system(f"bentoml push bci_classifier:latest")