import kfp
from kfp import dsl
from kfp.dsl import pipeline

from utils import yaml_parser
from components.processing import load_data_and_preprocess
from components.train import train
from components.train_automl import train_with_wandb
from components.upload_model import upload_sklearn_model_to_mlflow

@pipeline(name="bci_pipeline")
def bci_pipeline(
    hyperparameter_tuning: bool,
    model_name: str
): 
    """ Load yaml info """
    SECRET = yaml_parser("./secret.yaml")
    CFG = yaml_parser("./config.yaml")
    
    
    """ MAIN PIPELINE """
    data = load_data_and_preprocess(
        secret=SECRET,
        cfg=CFG,
    ).set_display_name("Load dataset from GCS and Preprocess it")
    
    with dsl.Condition(hyperparameter_tuning==True):
        model = train_with_wandb(
            cfg=CFG,
            train_x=data.outputs["train_x"],
            train_y=data.outputs["train_y"],
            test_x=data.outputs["test_x"],
            test_y=data.outputs["test_y"],
        ).set_display_name("Train data with hyperparameter tuning")
        
        _ = upload_sklearn_model_to_mlflow(
            secret=SECRET,
            model_name=model_name,
            model=model.outputs["model"],
            signature=model.outputs["signature"],
            conda_env=model.outputs["conda_env"],
        ).set_display_name("Upload custom model to mlflow")
        
    with dsl.Condition(hyperparameter_tuning==False):
        model = train(
            cfg=CFG,
            train_x=data.outputs["train_x"],
            train_y=data.outputs["train_y"],
            test_x=data.outputs["test_x"],
            test_y=data.outputs["test_y"],
        ).set_display_name("Train data with default parameter")
        
        _ = upload_sklearn_model_to_mlflow(
            secret=SECRET,
            model_name=model_name,
            model=model.outputs["model"],
            signature=model.outputs["signature"],
            conda_env=model.outputs["conda_env"],
        ).set_display_name("Upload custom model to mlflow")
    
if __name__ == "__main__":
    kfp.compiler.Compiler().compile(bci_pipeline, "./yamls/bci_pipeline.yaml")
 