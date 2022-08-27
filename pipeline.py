import yaml
import kfp
from kfp import dsl
from kfp.dsl import pipeline

from components.load_data import component_load_data
from components.preprocessing import component_preprocessing

@pipeline(name="bci_pipeline")
def bci_pipeline(hyperparameter_tuning: bool, 
                 kernel: str,
                 model_name: str
                 ):
    
    with open('secret.yaml') as file_reader:
        yaml_loader = yaml.load(file_reader, Loader=yaml.FullLoader)
    
    
    
    GCS_BUCKET_NAME = yaml_loader["GCS_BUCKET_NAME"]
    data_result = component_load_data(gcs_bucket_name=GCS_BUCKET_NAME)
    
    preprocessing_result = component_preprocessing(
        data_path=data_result.outputs["data"],
        target_path=data_result.outputs["target"],
        bandorder=2
        )
    
    # with dsl.Condition(hyperparameter_tuning==True):
    #     print("automl_train()")
        
    # with dsl.Condition(hyperparameter_tuning==False):
    #     print("train()")
    
if __name__ == "__main__":
    kfp.compiler.Compiler().compile(bci_pipeline, "yamls/bci_pipeline.yaml")
 