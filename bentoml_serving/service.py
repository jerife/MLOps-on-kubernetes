from scipy import signal
import numpy as np

import bentoml
from bentoml.io import NumpyNdarray

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

label2class = {
    6: "left",
    7: "right"
}
bentoml.picklable_model.get
bci_runner = bentoml.picklable_model.get("bci_clf:latest").to_runner()
svc = bentoml.Service("bci_classifier", runners=[bci_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(signals):
    signal_list = []
    for signal in signals:
        signal_list.append(apply_bandpass_filter(signal))
        
    result = bci_runner.run(np.array(signal_list))
    return np.array([label2class[i] for i in result])