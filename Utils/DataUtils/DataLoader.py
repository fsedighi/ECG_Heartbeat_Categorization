import pandas as pd

class ECGDataLoader:
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = None
        self.x = None
        self.y = None

    def load_ecg_dataset_csv(self):
        self.dataset = pd.read_csv(self.dataset_path, header=None)

    def split_x_y(self):
        self.x = self.dataset.iloc[:, :-1]  # Select all columns except the last one
        self.y = self.dataset.iloc[:, -1]   # Select only the last column
        return self.x, self.y

    def get_dataset(self):
        return self.dataset
    
