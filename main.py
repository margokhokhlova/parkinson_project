from utils.data_loader import PatientsRawData
from utils.preprocessing import preprocess_signal
import os 

if __name__ == "__main__":
    data = PatientsRawData('../data/Исходные файлы/')
    data.load_data()
    print(data.Y)
    print(len(data.X))
    print(data.X[0].head())
    # check splitting function
    data.get_train_val_split()
    print(f'There are {len(data.y_train)} train sequences for labels {data.y_train} and {len(data.y_test)} sequences with labels {data.y_test}')
    # filter out unused data
    data.get_emg_data()
    print(len(data.X))
    data.convert_to_numpy()
    print(len(data.X[0]), len(data.X[0][0]))
    # process one signal
    signal_emg = data.X[0][0]
    print(signal_emg.shape)
    processed_signal =  preprocess_signal(signal_emg,  SamplingRate=500,  LF=60, HF=240, frequences_to_filter = [50, 100, 150, 200 ], order_butter=4, save_plot=True) 

