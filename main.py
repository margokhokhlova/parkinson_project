from utils.data_loader import PatientsRawData
import os 

if __name__ == "__main__":
    data = PatientsRawData('../data/Исходные файлы/')
    data.load_data()
    print(data.Y)
    print(len(data.X))
    print(len(data.X[0]))
    # check splitting function
    #X_train, y_train, X_test, y_test = data.get_train_val_split() 
    data.get_train_val_split()
    print(f'There are {len(data.y_train)} train sequences for labels {data.y_train} and {len(data.y_test)} sequences with labels {data.y_test}')
