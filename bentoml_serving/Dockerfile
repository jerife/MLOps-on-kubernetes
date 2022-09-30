FROM python:3.7.13

RUN apt-get -y update && apt-get install -y libzbar-dev
RUN pip install --upgrade pip
RUN pip install -U mlflow==1.28.0 boto3 bentoml==1.0.4 numpy==1.21.6 \
pandas==1.3.5 scikit-learn==1.0.2 mne==1.1.0 scipy==1.7.3 dill==0.3.4

RUN mkdir -p /app
COPY . /app/
WORKDIR /app
