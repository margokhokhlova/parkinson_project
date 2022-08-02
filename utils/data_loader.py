import os

class PatientsRawData:
    def __init__(self, file_path):        
        self.path = file_path
        self.X = []
        self.Y = []
    def load_data(self):
        subfolders = [ f.path for f in os.scandir(self.path) if f.is_dir()]
        for s_folder in subfolders:
            for filename in os.listdir(s_folder):
                f = os.path.join(s_folder, filename)
                with open(f) as datafile:
                    contents = datafile.read()
                    self.X.append(contents)
                    self.Y.append(s_folder.split('_')[-1])



        