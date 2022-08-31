from functools import partial
from kfp.components import InputPath, OutputPath, create_component_from_func

        
@partial(
    create_component_from_func,
    packages_to_install=["tqdm==4.64.0", "numpy==1.21.6", "mne==1.1.0", "dill==0.3.4", "scikit_learn==1.0.2", "mlflow==1.28.0"],
)
def train(
    param_n_components: int,
    param_log: bool,
    param_norm_trace: bool,
    param_c: float,
    param_kernel: str,
    param_gamma: str,
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
    
    
    """ Define the model class """
    import mlflow
    class CSP_SVM(mlflow.pyfunc.PythonModel):  
        def __init__(self, n_components=10, reg=None, log=True, norm_trace=False,
                     C=1.0, kernel='rbf', gamma='scale'):
            from sklearn.svm import SVC
            from mne.decoding import CSP

            self.csp = CSP(
                n_components=n_components,
                reg=reg,
                log=log, 
                norm_trace=norm_trace,
            )
            self.svm = SVC(
                C=C,
                kernel=kernel,
                gamma=gamma,
            )

        def fit(self, train_x, train_y):
            transform_train_x = self.csp.fit_transform(train_x, train_y)
            self.svm.fit(transform_train_x, train_y)

        def predict(self, context, input):
            transform_test_x = self.csp.transform(input)
            pred = self.svm.predict(transform_test_x)
            return pred

    model = CSP_SVM(
        n_components=param_n_components,
        log=param_log,
        norm_trace=param_norm_trace,
        C=param_c,
        kernel=param_kernel,
        gamma=param_gamma,
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
        
        
@partial(
    create_component_from_func,
    packages_to_install=["tqdm==4.64.0", "numpy==1.21.6", "mne==1.1.0", "dill==0.3.4", "scikit_learn==1.0.2", "mlflow==1.28.0", "wandb"],
)
def hyperparameter_tunig(
    secret: dict,
    wandb_project_name: str,
    hyperparameter_tunig_count: int,
    train_x_path: InputPath("dill"),
    train_y_path: InputPath("dill"),
    test_x_path: InputPath("dill"),
    test_y_path: InputPath("dill"),
):
    """ Load train/test data """
    import dill
    with open(train_x_path, mode="rb") as file_reader:
        train_x = dill.load(file_reader)
    with open(train_y_path, mode="rb") as file_reader:
        train_y = dill.load(file_reader)
    with open(test_x_path, mode="rb") as file_reader:
        test_x = dill.load(file_reader)
    with open(test_y_path, mode="rb") as file_reader:
        test_y = dill.load(file_reader)
    
    
    """ Define the model class and Train """
    """ Define the model class """
    import mlflow
    class CSP_SVM(mlflow.pyfunc.PythonModel):  
        def __init__(self, n_components=10, reg=None, log=True, norm_trace=False,
                     C=1.0, kernel='rbf', gamma='scale'):
            from sklearn.svm import SVC
            from mne.decoding import CSP

            self.csp = CSP(
                n_components=n_components,
                reg=reg,
                log=log, 
                norm_trace=norm_trace,
            )
            self.svm = SVC(
                C=C,
                kernel=kernel,
                gamma=gamma,
            )

        def fit(self, train_x, train_y):
            transform_train_x = self.csp.fit_transform(train_x, train_y)
            self.svm.fit(transform_train_x, train_y)

        def predict(self, context, input):
            transform_test_x = self.csp.transform(input)
            pred = self.svm.predict(transform_test_x)
            return pred
    
    
    """ Define the model class """
    import wandb
    from sklearn.metrics import accuracy_score
    def sweep_train(config=None):
        wandb.init(
            config = wandb.config,
        )
        cfg = wandb.config

        model = CSP_SVM(
            n_components=cfg["csp_n_components"], 
            reg=None,
            log=cfg["csp_log"], 
            norm_trace=cfg["csp_norm_trace"],       
            C=cfg["svm_C"],
            kernel=cfg["svm_kernel"],
            gamma=cfg["svm_gamma"],
        )
        model.fit(train_x, train_y)
        pred = model.predict(context="", input=test_x)
        acc = accuracy_score(pred, test_y)
        wandb.log({"acc": acc})  
        
        
    """ Main Code """
    wandb.login(key=secret["WANDB"])

    sweep_config = {
        'method': 'random' # grid/random/bayes
    }
    metric = {
        'name': 'acc',
        'goal': 'maximize' # minimize/maximize
    }
    parameters_dict = {
        'csp_n_components': {
            'values': [5, 10, 20, 40, 60]
        },
        'csp_log': {
            'values': [True, False]
        },
        'csp_norm_trace': {
            'values': [True, False]
        },
        'svm_C': {
            'values': [0.8, 1.0, 1.2, 1.5]
        },
        'svm_kernel': {
            'values': ['linear', 'poly', 'rbf', 'sigmoid']
        },
        'svm_gamma': {
            'values': ['scale', 'auto']
        }
    }
    sweep_config['metric'] = metric
    sweep_config['parameters'] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project=wandb_project_name)
    print("sweep_id:", sweep_id)
    wandb.agent(sweep_id, sweep_train, count=hyperparameter_tunig_count)