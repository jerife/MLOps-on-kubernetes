from scipy import signal
import numpy as np
    
def apply_bandpass_filter(data: np.array,
                                fs: int,
                                band: list,
                                band_order:int
                                ) -> np.array:
    """ Apply Band Pass Filter """
    
    
    b2, a2 = signal.butter(band_order, band, 'band', fs=fs)  # band pass define

    data_list = []
    for each_channel in data: # 64 * X
        bandpass_data = signal.lfilter(b2, a2, each_channel)
        data_list.append(bandpass_data)
        
    return np.array(data_list)