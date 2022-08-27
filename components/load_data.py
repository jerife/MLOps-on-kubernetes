from functools import partial
from kfp.components import OutputPath, create_component_from_func
    
    
@partial(
    create_component_from_func,
    packages_to_install=["google-cloud-storage==1.44.0", "tqdm==4.64.0", "numpy==1.21.6", "mne==1.1.0", "dill==0.3.4"],
)
def component_load_data(
    data_path: OutputPath("dill"),
    target_path: OutputPath("dill"),
    gcs_bucket_name: str,
    ):
    """ Load data from google cloud storage """
    from google.cloud import storage
    import os
    
    prefix = 'data/'
    dl_dir = './data/'
    os.mkdir(dl_dir)
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name=gcs_bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)  # Get list of files
    for blob in blobs:
        filename = blob.name.replace(prefix, "")
        blob.download_to_filename(dl_dir + filename)  # Download
    
    
    """ Extract usefull data """  
    from glob import glob
    from tqdm import tqdm
    import numpy as np
    import mne
    
    data_paths = glob(dl_dir+"*T*") # Get train data
    data_signal_list = []
    data_label_list = []
    
    for data_path_each in tqdm(data_paths):
        data = mne.io.read_raw_gdf(data_path_each)
        
        data_signal = data.get_data() # Get data info
        data_label_info = mne.events_from_annotations(data) # Get label info
        data_label = data_label_info[0]

        data_label[:,2] = data_label[:,2]-1 # To allocate label for according data
        data_signal_list.append(data_signal)
        data_label_list.append(data_label)
    
    
    """ Convert array to dill to deliver data """
    import dill
    with open(data_path, "wb") as file_writer:
        dill.dump(data_signal_list, file_writer)
    with open(target_path, "wb") as file_writer:
        dill.dump(data_label_list, file_writer)