import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

class PatientsRawData:
    def __init__(self, file_path):        
        self.path = file_path
        self.X = []
        self.Y = []
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def load_data(self):
        subfolders = [ f.path for f in os.scandir(self.path) if f.is_dir()]
        for s_folder in subfolders:
            for filename in os.listdir(s_folder):
                f = os.path.join(s_folder, filename)
                with open(f) as datafile:
                    lst = [int(x) for x in datafile.read().rstrip().split()]
                    self.X.append(lst)
                    self.Y.append(s_folder.split('_')[-1])

    def shuffle_data(self, random_state=0):
        self.X, self.Y = shuffle(self.X, self.Y, random_state)

    def get_train_val_split(self, test_size = 0.2, stratified = True):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y,
                                                    stratify=self.Y, test_size=test_size)


        





        