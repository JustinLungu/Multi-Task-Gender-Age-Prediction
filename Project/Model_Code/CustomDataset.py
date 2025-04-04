from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch

class CustomDataset(Dataset):
    def __init__(self, model_type, image_folder, dataframe, transform=None):
        self.model_type = model_type
        self.image_folder = image_folder
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]  # Assuming the image column is 'Image_ID'
        features = []
        if self.model_type == 'age_baseline' or self.model_type == 'final':
            age = torch.tensor(self.dataframe.iloc[idx, 1])
            features.append(age)
        if self.model_type == 'gender_baseline' or self.model_type == 'final':
            gender = torch.tensor(self.dataframe.iloc[idx, 2])
            features.append(gender)

        # Lazy loading: return the image path and label instead of loading the image
        return img_name, features

    def load_image(self, img_name):
        img_path = self.image_folder + '/' + img_name
        image = Image.open(img_path).convert('L')
        #need a transformer for test images just to convert to tensor and to resahape and allat
        if self.transform == 'train' or self.transform == 'val':
            # Define transformations for data augmentation
            transform = transforms.Compose([
                transforms.Resize((100, 100)),
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1),
                transforms.Normalize(0.5, 0.5),
            ])
            image = transform(image)
        elif self.transform == "test":
            transform = transforms.Compose([
                transforms.Resize((100, 100)),
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
            image = transform(image)
        return image