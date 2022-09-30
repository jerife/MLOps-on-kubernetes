# MLOps-on-kubernetes
<div align="center">
    <img src="https://user-images.githubusercontent.com/68190553/187675054-52a8e90e-24c1-46fc-b7d0-af49d2478fb9.png"
    width="100%"/>
</div>

### Why Kubernetes in MLOps?
Kubernetes provides flexible control over containers through orchestrations such as Scheduling, Load balancing, and Scaling.<br/>
Therefore, it is suitable to systematically build and operate all configurations of ML projects, including Data collection, Preprocessing, Feature extraction, Data validation, Monitoring, and Deploying.<br/>
In particular, containers packaged with train code can be run on nodes with GPUs, and containers packaged with data preprocessing code can be run on nodes with plenty of memory.<br/>
Also, because it is managed as a Kubernette container, Dockerfile ensure the same environment as engineers.


## Architecture
<div align="center">
    <img src="https://user-images.githubusercontent.com/68190553/187680507-b59dc6fe-4b1d-4113-a043-fd5adeb31761.png"
    width="100%"/>
</div><br/>

- Kubernetes: Deploy a Kubernetes cluster with the Google Kubernetes Engine.
- ML PIPELINE: Build an ML PIPELINE that learns and deploys the model only by entering parameters with Kubeflow.
- Data Storage: Manage your data with Google Cloud Storage.
- Experiment Tracking & AutoML: Use Weight & Biases to track the experiment and find the optimal Hyperparameter.
- Model Versioning: Manage and save models by version with Mlflow.
- Model Serving: API communication with user through BentoML.
- Monitoring: Monitor the cluster's resources with Prometheus & Grafana.


## ML PIPELINE
<div align="center">
    <img src="https://user-images.githubusercontent.com/68190553/187819501-7d6a3bbb-6e81-4e53-9c91-60307f6d1d95.png"
    width="90%"/>
</div><br/>

Pipeline has two conditions depending on the user's parameter input.
```
- Condition 1: Hyperparameter tunning
- Condition 2: Train
```


<details open>
<summary>Condition 1. Hyperparameter tunning 🔍 </summary>

### A. Hyperparameter tunning with Weight&Biases
<div align="center">
    <br/>
    <img src="https://user-images.githubusercontent.com/68190553/187695957-e93d8722-3428-4e35-98a7-590b44801042.png"
    width="100%"/>
</div><br/>

If Condition: Hyperparameter tuning, the model is not trained and only tuning is performed.<br/>
The number of tuning can be controlled through input variables, and the tuning process can be checked through Weight & Biases Porject.

<br/>
</details>


<details open>
<summary> Condition 2. Train 🛠 </summary>

### A. Model Versioning with Mlflow 
<div align="center">
    <img src="https://user-images.githubusercontent.com/68190553/188034509-569543ec-938b-459f-aec3-f5348cbd6b77.png"
    width="70%"/>
</div><br/>

In "Condition: Train", the model is trained according to the input parameters, and if you do not enter the parameters, the model is trained with the default value.<br/>
The trained model is compared to the model registered in the mlflow, and if it has better accuracy, it is mlflow uploaded and versioning.
    
    
### B. Model Serving with Bentoml
<div align="center">
    <img src="https://user-images.githubusercontent.com/68190553/187844129-b09dd15a-2a70-44c8-9939-052eb2d58864.png"
    width="70%"/><br/>
</div><br/>

If the model is stored in Mlflow, the model will is pushed to BentoML.<br/>
Pushed models are deployed as desired by the user. (eg. CPU, GPU, Memory etc)<br/>


<br/>
</details>

### Cluster Monitoring 
<div align="center">
    <img src="https://user-images.githubusercontent.com/68190553/193259553-139912cc-3291-47ff-9a85-46f18cc46e5e.png"
    width="90%"/>
</div><br/>
Customize Prometheus & Grafana to monitor clusters.

### Model Inference
```bash
$ curl \                                                                                                                                                   
    -X POST \
    -H "content-type: application/json" \
    --data "[[[1.1, 2.2, 3.3, 4.4],
                      ... 
               5.5, 6.6, 7.7, 8.8]]]" \ # shape: N x 22 x 750
    https://demo-default-yatai-127-0-0-1.apps.yatai.dev/classify
    
>>> "left"
```
```
Currently, there are many problems with BentoML1.0 in the GKE environment, so it is registered as an issue, and the writing is temporarily written.
```
-> Issue: https://github.com/bentoml/Yatai/issues/322
If this issue is resolved, then i will apply bentoml again.

---

## Machine Learning Task
<div align="center">
    <img src="https://user-images.githubusercontent.com/68190553/187877740-ac248b69-046f-412a-b1d5-7e974b95f2db.png"
    width="100%"/><br/>
</div><br/>

### Overview
In this task, I aim to build a motor image (MI) task, which is mainly covered in the Brain Computer Interface, into MLOps <br/>
(ie. MI task: input brain waves generated when imagining moving into the model to derive results)

### Data
[BCI Competition IV 2a Dataset](https://www.bbci.de/competition/iv/) (Classification of EEG signals affected by eye movement artifacts)

### Task
1. Preprocessing 
    - Band Pass Filter 8~30Hz 
    - Segment the data into trainable shapes
2. Feature Extraction
    - Common Spatial Pattern
3. Modeling
    - Support Vector Machine

---

## Author
```yaml
Github: @jeirfe
Website: jerife.github.io
Email: jerife@naver.com

Copyright © 2022 jerife.
This project is Apache-2.0 licensed.
```
