import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
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

eeg_columns = ['FP1','FP2','F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'CZ']

class PatientsRawData:
    def __init__(self, file_path):        
        self.path = file_path
        self.X = []
        self.Y = []
        self.patient = []
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def load_data(self):
        subfolders = [ f.path for f in os.scandir(self.path) if f.is_dir()]
        for s_folder in subfolders:
            for filename in os.listdir(s_folder):
                f = os.path.join(s_folder, filename)            
                data = pd.read_csv(f, sep=" ", header=None)
                data = data.iloc[: , :-1] # remove an extra column from the data
                data.columns = headers
                self.X.append(data)
                self.Y.append(s_folder.split('_')[-1])
                self.patient.append(filename)

    def get_emg_data(self):
        ''' function to run before training, filters out the unused data'''
        for i in range(len(self.X)):
            df = self.X[i]
            df = df.drop(columns=eeg_columns)
            self.X[i]=df #.to_numpy().transpose() # and converts data to numpy array for the following training

    def shuffle_data(self, random_state=0):
        self.X, self.Y = shuffle(self.X, self.Y, random_state)

    def get_train_val_split(self, test_size = 0.2, stratified = True):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y,
                                                    stratify=self.Y, test_size=test_size)


        





        