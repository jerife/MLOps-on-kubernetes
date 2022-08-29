# mlflow의 최신버전과 train에서 학습된 모델끼리 성적 비교 후 업로드 결정
from functools import partial
from pickle import FALSE
from kfp.components import InputPath, OutputPath, create_component_from_func

@partial(
    create_component_from_func,
    packages_to_install=["tqdm==4.64.0", "numpy==1.21.6", "mne==1.1.0", "dill==0.3.4", "scikit_learn==1.0.2", "mlflow==1.28.0", "boto3"],
)
def evaluate(
    secret: dict,
    model_name: str,
    test_x_path: InputPath("dill"),
    test_y_path: InputPath("dill"),
    model_path: InputPath("dill"),
)->bool:
    
    import os
    import dill
    from sklearn.metrics import accuracy_score
    
    """ Load Test data to validate pipeline model """    
    with open(test_x_path, mode="rb") as file_reader:
        test_x = dill.load(file_reader)
    with open(test_y_path, mode="rb") as file_reader:
        test_y = dill.load(file_reader)
    
    
    """ Load pipeline model """
    with open(model_path, mode="rb") as file_reader:
        pipeline_model = dill.load(file_reader)
    
    pipeline_pred = pipeline_model.predict(context="", input=test_x)
    pipeline_acc = accuracy_score(pipeline_pred, test_y)
    
    
    """ Load latest version model from Mlflow server """
    from mlflow.pyfunc import load_model
    from mlflow.tracking import MlflowClient
    import boto3
    
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://mlflow-minio.mlflow-system.svc:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = secret["AWS_ACCESS_KEY_ID"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = secret["AWS_SECRET_ACCESS_KEY"]
    client = MlflowClient("http://mlflow-service.mlflow-system.svc:5000")

    filter_string = "name='BCI-Model'"
    results = client.search_model_versions(filter_string)

    latest_version = 0
    for res in results: 
        if int(res.version) > latest_version:
            latest_version=res.version
            model_uri = res.source

    latest_model = load_model(model_uri)
    latest_pred = latest_model.predict(test_x)
    latest_acc = accuracy_score(latest_pred, test_y)

    
    """ If pipeline_acc is larger than latest_acc, Upload model """
    if pipeline_acc >= latest_acc:
        return True
    else:
        return False