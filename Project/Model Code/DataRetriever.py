import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from CustomDataset import CustomDataset

class DataRetriever:
    def __init__(self, df_path, image_folder_path, batch_size, dataPercentage):
        self.df_path = df_path
        self.image_folder_path = image_folder_path
        self.batch_size = batch_size
        self.dataPercentage = dataPercentage
        self.df = pd.read_csv(self.df_path, dtype={'Age': 'float32', 'Gender': 'float32'})
        self.train_data, self.val_data, self.test_data = self.split_data()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def split_data(self):
        train_ratio = 0.7
        val_ratio = 0.2
        test_ratio = 0.1

        df_use, df_discard = train_test_split(self.df, test_size=1-self.dataPercentage, random_state=42)

        train_val, test = train_test_split(df_use, test_size=test_ratio, random_state=42)
        train, val = train_test_split(train_val, test_size=val_ratio/(train_ratio + val_ratio), random_state=42)

        return train, val, test

    def retrieve_datasets(self):
        self.train_dataset = CustomDataset(dataframe=self.train_data, image_folder=self.image_folder_path, transform='train')
        self.val_dataset = CustomDataset(dataframe=self.val_data, image_folder=self.image_folder_path, transform='val')
        self.test_dataset = CustomDataset(dataframe=self.test_data, image_folder=self.image_folder_path, transform='test')
        return self.train_dataset, self.val_dataset, self.test_dataset

    def retrieve_loaders(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader, val_loader, test_loader