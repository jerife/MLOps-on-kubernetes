from functools import partial
from kfp.components import InputPath, OutputPath, create_component_from_func

from components.utils.filter import apply_bandpass_filter


@partial(
    create_component_from_func,
    packages_to_install=["tqdm==4.64.0", "numpy==1.21.6", "dill==0.3.4"],
)
def component_preprocessing(
    data_path: InputPath("dill"),
    target_path: InputPath("dill"),
    trial_data_path: OutputPath("dill"),
    trial_target_path: OutputPath("dill"),
    band_order: int,
    ):
    """ Load Data """
    import dill
    with open(data_path, 'rb') as file_writer:
        data_signal_all = dill.load(file_writer)
    with open(target_path, 'rb') as file_writer:
        data_label_all = dill.load(file_writer)


    """ Data Preprocessing """
    import numpy as np
    from tqdm import tqdm
    
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
    for data_signal, data_label in zip(data_signal_all, data_label_all):
        use_data = data_label[np.concatenate([np.where(data_label[:,2]==class_1), # left class
                        np.where(data_label[:,2]==class_2)], axis=-1)[0]] # right class
        preprocessing_data_label = use_data[:,2]
        label_list.append(preprocessing_data_label)
        
        data_signal = data_signal[:22] # Use only EEG data, index 23, 24, 25 are EOG data
        data_signal = apply_bandpass_filter(data_signal, fs=250, band=[8,30], band_order=band_order) # bandpass filter

        SEC_PER_SAMPLE = 250
        WINDOW_LENGTH = 9
        WINDOW_SIZE = 2
        WINDOW_INTERVAL = 0.1
        
        for use_data_each in tqdm(use_data):

            window_list = []
            for window in range(WINDOW_LENGTH):
                start = int(use_data_each[0] + window*WINDOW_INTERVAL*SEC_PER_SAMPLE)
                end = int(start + WINDOW_SIZE*SEC_PER_SAMPLE)

                channel_list=[]
                for each_channel in data_signal:
                    channel_list.append(np.array(each_channel[start:end]))

                window_list.append(np.array(channel_list))

            trial_list.append(np.array(window_list))
    
    
    """ Convert array to dill to deliver data """
    with open(trial_data_path, "wb") as file_writer:
        dill.dump(np.array(trial_list), file_writer)
    with open(trial_target_path, "wb") as file_writer:
        dill.dump(np.array(label_list).reshape(-1), file_writer)
        

