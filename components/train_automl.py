from functools import partial
from kfp.components import InputPath, OutputPath, create_component_from_func

@partial(
    create_component_from_func,
    packages_to_install=["tqdm==4.64.0", "numpy==1.21.6", "mne==1.1.0", "dill==0.3.4", "scikit_learn==1.0.2", "mlflow==1.28.0", "wandb"],
)
def train_with_wandb(
    cfg: dict,
    train_x_path: InputPath("dill"),
    train_y_path: InputPath("dill"),
    test_x_path: InputPath("dill"),
    test_y_path: InputPath("dill"),
    model_path: OutputPath("dill"),
    feature_extractor_path: OutputPath("dill"),
    signature_path: OutputPath("dill"),
    conda_env_path: OutputPath("dill"),
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


    """ Train and Validate """
    from sklearn.metrics import accuracy_score
    from sklearn.svm import SVC
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from mne.decoding import CSP
    
    # Train
    csp = CSP(
        n_components=cfg["csp"]["n_components"], 
        reg=None,
        log=cfg["csp"]["log"], 
        norm_trace=cfg["csp"]["norm_trace"],
    )
    transform_train_x = csp.fit_transform(train_x, train_y)
    clf = SVC()
    clf.fit(transform_train_x, train_y)
    
    # Validate  
    transform_test_x = csp.transform(test_x)
    pred = clf.predict(transform_test_x)
    acc = accuracy_score(pred, test_y)


    """ Convert data to mlflow format data """
    from mlflow.models.signature import infer_signature
    from mlflow.utils.environment import _mlflow_conda_env
    
    signature = infer_signature(train_x, clf.predict(csp.transform(train_x)))
    conda_env = _mlflow_conda_env(
        additional_pip_deps=["dill", "pandas", "scikit-learn", "mne"]
    )
    
    with open(model_path, mode="wb") as file_writer:
        dill.dump(clf, file_writer)
    with open(feature_extractor_path, mode="wb") as file_writer:
        dill.dump(csp, file_writer)    
    with open(signature_path, "wb") as file_writer:
        dill.dump(signature, file_writer)    
    with open(conda_env_path, "wb") as file_writer:
        dill.dump(conda_env, file_writer)