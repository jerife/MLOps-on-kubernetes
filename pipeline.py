import kfp
from kfp import dsl
from kfp.dsl import pipeline

from utils import yaml_parser
from kubeflow_components.processing import load_data_and_preprocess
from kubeflow_components.train import train, hyperparameter_tunig
from kubeflow_components.registration import upload_model_to_mlflow
from kubeflow_components.evaluation import evaluate
from kubeflow_components.pushing import push_to_bentoml

@pipeline(
    name='BCI-PIPELINE',
    description='Classify Brain waves to left/right class through the MI task of BCI.'
)
def bci_pipeline(
    hyperparameter_tunig_count=0,
    wandb_project_name="BCI-sweeps",
    model_name="test_model",
    param_band_order=2,
    param_fs=250,
    param_window_size=750,
    param_n_components=20,
    param_log=True,
    param_norm_trace=False,
    param_c=1.0,
    param_kernel="rbf",
    param_gamma="scale",
    random_state=2022,
): 
    """ Load yaml info """
    SECRET = yaml_parser("./secret.yaml")
    
    
    """ MAIN PIPELINE """
    data_result = load_data_and_preprocess(
        secret=SECRET,
        random_state=random_state,
        band_order=param_band_order,
        fs=param_fs,
        window_size=param_window_size,
    ).set_display_name("Load dataset from GCS and Preprocess it")
    
    with dsl.Condition(hyperparameter_tunig_count!=0):
        model_result = hyperparameter_tunig(
            secret=SECRET,
            hyperparameter_tunig_count=hyperparameter_tunig_count,
            wandb_project_name=wandb_project_name,
            train_x=data_result.outputs["train_x"],
            train_y=data_result.outputs["train_y"],
            test_x=data_result.outputs["test_x"],
            test_y=data_result.outputs["test_y"],   
        ).set_display_name("Hyperparameter tuning with W&B") 
    with dsl.Condition(hyperparameter_tunig_count==0):
        model_result = train(
            param_n_components=param_n_components,
            param_log=param_log,
            param_norm_trace=param_norm_trace,
            param_c=param_c,
            param_kernel=param_kernel,
            param_gamma=param_gamma,
            train_x=data_result.outputs["train_x"],
            train_y=data_result.outputs["train_y"],
        ).set_display_name("Train model with User/Default parameters")
    
        eval_result = evaluate(
            secret=SECRET,
            model_name=model_name,
            test_x=data_result.outputs["test_x"],
            test_y=data_result.outputs["test_y"],
            model=model_result.outputs["model"]
        ).set_display_name("Evaluate model and Determines whether to upload the model")
        
        with dsl.Condition(eval_result.output==True):
            _ = upload_model_to_mlflow(
                secret=SECRET,
                model_name=model_name,
                model=model_result.outputs["model"],
                conda_env=model_result.outputs["conda_env"],
            ).set_display_name("Upload custom model to MLflow")

            _ = push_to_bentoml(
                secret=SECRET,
            ).set_display_name("Push model that was registered in MLflow to BentoML").after(_)
    """ MAIN PIPELINE """


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(bci_pipeline, "./BCI-PIPELINE.yaml")