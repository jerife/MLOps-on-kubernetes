from functools import partial
from kfp.components import InputPath, OutputPath, create_component_from_func

@partial(
    create_component_from_func,
    packages_to_install=["tqdm==4.64.0", "numpy==1.21.6", "mne==1.1.0", "dill==0.3.4", "scikit_learn==1.0.2", "mlflow==1.28.0"],
)
def train(
    cfg: dict,
    train_x_path: InputPath("dill"),
    train_y_path: InputPath("dill"),
    model_path: OutputPath("dill"),
    conda_env_path: OutputPath("dill"),
):
    """ Load train/test data """
    import dill
    with open(train_x_path, mode="rb") as file_reader:
        train_x = dill.load(file_reader)
    with open(train_y_path, mode="rb") as file_reader:
        train_y = dill.load(file_reader)
    
    
    """ Define the model class and Train """
    import mlflow

    class CSP_SVM(mlflow.pyfunc.PythonModel):  
        def __init__(self, n_components, reg, log, norm_trace, kernel):
            from sklearn.svm import SVC
            from mne.decoding import CSP

            self.csp = CSP(
                n_components=n_components,
                reg=reg,
                log=log, 
                norm_trace=norm_trace
            )
            self.svm = SVC(kernel=kernel)

        def fit(self, train_x, train_y):
            transform_train_x = self.csp.fit_transform(train_x, train_y)
            self.svm.fit(transform_train_x, train_y)

        def predict(self, context, input):
            transform_test_x = self.csp.transform(input)
            pred = self.svm.predict(transform_test_x)
            return pred

    model = CSP_SVM(
        n_components=cfg["csp"]["n_components"], 
        reg=None,
        log=cfg["csp"]["log"], 
        norm_trace=cfg["csp"]["norm_trace"],       
        kernel=cfg["svm"]["kernel"],
    )
    model.fit(train_x, train_y)

    """ Convert data to mlflow format data """
    from mlflow.utils.environment import _mlflow_conda_env
    
    conda_env = _mlflow_conda_env(
        additional_pip_deps=["dill", "pandas", "scikit-learn", "mne"]
    )
    
    with open(model_path, mode="wb") as file_writer:
        dill.dump(model, file_writer)
    with open(conda_env_path, "wb") as file_writer:
        dill.dump(conda_env, file_writer)