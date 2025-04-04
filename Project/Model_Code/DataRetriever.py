import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from CustomDataset import CustomDataset
import os

class DataRetriever:
    def __init__(self, model_type, df_path, image_folder_path, batch_size, dataPercentage, split_done = False):
        self.model_type = model_type
        self.df_path = df_path
        self.image_folder_path = image_folder_path
        self.batch_size = batch_size
        self.dataPercentage = dataPercentage
        self.df = pd.read_csv(self.df_path, dtype={'Age': 'float32', 'Gender': 'float32'})
        if split_done:
            self.train_data, self.val_data, self.test_data = self.read_dataset_csv()
        else:
            self.train_data, self.val_data, self.test_data = self.split_data()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def split_data(self):
        train_ratio = 0.7
        val_ratio = 0.2
        test_ratio = 0.1

        if self.dataPercentage != 1:
            df_use, df_discard = train_test_split(self.df, test_size=1 - self.dataPercentage, random_state=42)
        else:
            df_use = self.df

        train_val, test = train_test_split(df_use, test_size=test_ratio, random_state=42)
        train, val = train_test_split(train_val, test_size=val_ratio/(train_ratio + val_ratio), random_state=42)

        # Create the directory if it doesn't exist
        output_directory = '../data/datasets/'
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        test.to_csv(output_directory + 'test_set.csv', index=False)
        train.to_csv(output_directory + 'train_set.csv', index=False)
        val.to_csv(output_directory + 'val_set.csv', index=False)

        return train, val, test

    def read_dataset_csv(self):
        directory = '../data/datasets/'
        train = pd.read_csv(directory + 'train_set.csv', dtype={'Age': 'float32', 'Gender': 'float32'})
        val = pd.read_csv(directory + 'val_set.csv', dtype={'Age': 'float32', 'Gender': 'float32'})
        test = pd.read_csv(directory + 'test_set.csv', dtype={'Age': 'float32', 'Gender': 'float32'})

        return train, val, test

    def retrieve_datasets(self):
        self.train_dataset = CustomDataset(model_type = self.model_type, dataframe=self.train_data, image_folder=self.image_folder_path, transform='train')
        self.val_dataset = CustomDataset(model_type = self.model_type, dataframe=self.val_data, image_folder=self.image_folder_path, transform='val')
        self.test_dataset = CustomDataset(model_type = self.model_type, dataframe=self.test_data, image_folder=self.image_folder_path, transform='test')
        return self.train_dataset, self.val_dataset, self.test_dataset

    def retrieve_loaders(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader, val_loader, test_loader