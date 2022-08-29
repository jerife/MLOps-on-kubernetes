from unittest import result
import kfp
from kfp import dsl
from kfp.dsl import pipeline

from utils import yaml_parser
from components.processing import load_data_and_preprocess
from components.train import train
from components.train_automl import train_with_wandb
from components.upload_model import upload_model_to_mlflow
from components.evaluation import evaluate

@pipeline(
    name='BCI-PIPELINE',
    description='Classify Brain waves to left/right class through the MI task of BCI.'
)
def bci_pipeline(
    hyperparameter_tuning: bool,
    model_name: str
): 
    """ Load yaml info """
    SECRET = yaml_parser("./secret.yaml")
    CFG = yaml_parser("./config.yaml")
    
    
    """ MAIN PIPELINE """
    data_result = load_data_and_preprocess(
        secret=SECRET,
        cfg=CFG,
    ).set_display_name("Load dataset from GCS and Preprocess it")
    
    with dsl.Condition(hyperparameter_tuning==True):
        model_result = train(
            cfg=CFG,
            train_x=data_result.outputs["train_x"],
            train_y=data_result.outputs["train_y"],
        ).set_display_name("Train model with hyperparameter tuning") 
    with dsl.Condition(hyperparameter_tuning==False):
        model_result = train(
            cfg=CFG,
            train_x=data_result.outputs["train_x"],
            train_y=data_result.outputs["train_y"],
        ).set_display_name("Train model with default parameter")
    
    
    eval_result = evaluate(
        secret=SECRET,
        model_name=model_name,
        test_x=data_result.outputs["test_x"],
        test_y=data_result.outputs["test_y"],
        model=model_result.outputs["model"]
    ).set_display_name("Evaluate model")
    
    with dsl.Condition(eval_result.output==True):
        _ = upload_model_to_mlflow(
            secret=SECRET,
            model_name=model_name,
            model=model_result.outputs["model"],
            conda_env=model_result.outputs["conda_env"],
        ).set_display_name("Upload custom model to mlflow")
    

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(bci_pipeline, "./yamls/BCI-PIPELINE.yaml")