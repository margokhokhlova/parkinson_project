#import numpy as np
import pandas as pd

headers = [
    'FP1',# 1 канал
    'FP2',# канал
    'F3', #3
    'F4', #4
    'C3', #5
    'C4', #6
    'P3', #7
    'P4', #8
    'O1', #9
    'O2', #10
    'F7', #11
    'F8', #12
    'T3', #13
    'T4', #14
    'T5', #15
    'T6', #16
    'CZ', #17
    'EMG1', #18 канал - это левая рука испытуемого
    'EMG2', #19 - это правая рука испытуемого
    'EMG3', #20 - это левая нога испытуемого
    'EMG4'] # 21 - правая нога испытуемого

def save_as_csv(path_in, path_out):
    ''' converts txt to csv and adds headers '''
    # with open(path_in) as datafile:
    #     list_data = [int(x) for x in datafile.read().rstrip().split()]
    #     print(len(list_data))
    #     x = np.reshape(list_data, (-1, 21))
    #     print(x.shape)
    #     x = np.vstack([headers, x])
    #     save to csv file
    #     print(x[0,:])
    #     np.savetxt(path_out, x, delimiter=',')
    data = pd.read_csv(path_in, sep=" ", header=None)
    data = data.dropna('columns')
    data.columns = headers
    print(data.head())
    data.to_csv(path_out)

if __name__ == "__main__":
    save_as_csv('../data/Исходные файлы/Margo_DATA_Control/аста0101.txt', '../data/initial_data_as_csv/DATA_Control/аста0101.txt')
