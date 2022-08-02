from utils.data_loader import PatientsRawData
import os 

if __name__ == "__main__":
    data = PatientsRawData('../data/Исходные файлы/')
    data.load_data()
    print(data.Y)
    print(len(data.X))
    print(len(data.X[0]))
