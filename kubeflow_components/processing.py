from functools import partial
from kfp.components import OutputPath, create_component_from_func    

@partial(
    create_component_from_func,
    packages_to_install=["google-cloud-storage==1.44.0", "tqdm==4.64.0", "numpy==1.21.6", "mne==1.1.0", "dill==0.3.4", "scikit_learn==1.0.2"],
)
def load_data_and_preprocess(
    secret: dict,
    cfg: dict,
    train_x_path: OutputPath("dill"),
    train_y_path: OutputPath("dill"),
    test_x_path: OutputPath("dill"),
    test_y_path: OutputPath("dill"),
):
    """ Load data from google cloud storage """
    from google.cloud import storage
    import os
    
    prefix = 'data/'
    dl_dir = './data/'
    os.mkdir(dl_dir)
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name=secret["GCS_BUCKET_NAME"])
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


    """ Data Preprocessing """
    from scipy import signal
    import numpy as np
    def apply_bandpass_filter(
        data: np.array,
        fs: int,
        band: list,
        band_order:int
    ):
        """ Apply Band Pass Filter """
        b2, a2 = signal.butter(band_order, band, 'band', fs=fs)  # band pass define

        data_list = []
        for each_channel in data: # 22 * 750
            bandpass_data = signal.lfilter(b2, a2, each_channel)
            data_list.append(bandpass_data)
            
        return np.array(data_list)
        
    class2index = {
        "left": 6,
        "right": 7,
        "foot": 8,
        "tongue": 9,   
    }
    class_1 = class2index["left"]
    class_2 = class2index["right"]
    
    trial_list = []
    label_list = []
    for data_signal, data_label in zip(data_signal_list, data_label_list):
        use_data = data_label[np.concatenate([np.where(data_label[:,2]==class_1), # left class
                        np.where(data_label[:,2]==class_2)], axis=-1)[0]] # right class
        preprocessing_data_label = use_data[:,2]
        label_list.append(preprocessing_data_label)
        
        data_signal = data_signal[:22] # Use only EEG data, index 23, 24, 25 are EOG data
        data_signal = apply_bandpass_filter(data_signal, fs=cfg["signal"]["fs"], band=[8,30], band_order=cfg["signal"]["band_order"]) # bandpass filter

        for use_data_each in tqdm(use_data):
            start = int(use_data_each[0])
            end = int(start + cfg["signal"]["window_size"])
            
            channel_list=[]
            for each_channel in data_signal:
                channel_list.append(each_channel[start:end])
            trial_list.append(channel_list)
        
        
    """ Train/Test data split & Convert array to dill to deliver data"""
    import dill
    from sklearn.model_selection import train_test_split
    
    train_x, test_x, train_y, test_y = train_test_split(np.array(trial_list), np.array(label_list).reshape(-1), test_size=0.2, random_state=cfg["random_state"])
    with open(train_x_path, "wb") as file_writer:
        dill.dump(train_x, file_writer)
    with open(test_x_path, "wb") as file_writer:
        dill.dump(test_x, file_writer)
    with open(train_y_path, "wb") as file_writer:
        dill.dump(train_y, file_writer)
    with open(test_y_path, "wb") as file_writer:
        dill.dump(test_y, file_writer)
        
        

