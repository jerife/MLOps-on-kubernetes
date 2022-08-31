import os

from mlflow.pyfunc import load_model
from mlflow.tracking import MlflowClient
import boto3
import bentoml

from utils import yaml_parser
SECRET = yaml_parser("./secret.yaml")

def load_mlflow():
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://mlflow-minio.mlflow-system.svc:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = SECRET["AWS_ACCESS_KEY_ID"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = SECRET["AWS_SECRET_ACCESS_KEY"]
    client = MlflowClient("http://mlflow-service.mlflow-system.svc:5000")

    filter_string = "name='BCI-Model'"
    results = client.search_model_versions(filter_string)

    latest_version = 0
    for res in results: 
        if int(res.version) > int(latest_version):
            latest_version=res.version
            model_uri = res.source
    
    model = load_model(model_uri)
    return model
    
def bento_serve():
    bentoml.picklable_model.save_model(
        name="bci_clf",
        model=load_mlflow(),
    )


if __name__ == '__main__':
    bento_serve()
    
    api_token = SECRET["BENTOML"]["API_TOKEN"]
    ip = SECRET["BENTOML"]["IP"]
    os.system(f"bentoml yatai login --api-token {api_token} --endpoint {ip}")
    os.system(f"bentoml build")
    os.system(f"bentoml push bci_classifier:latest")