#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch
import tensorflow as tf
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from PIL import Image
import pandas as pd
from torchvision import transforms
from sklearn.model_selection import train_test_split
import preprocessing
import torch.optim as optim


# In[5]:


def split_data(df):
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1
    
    df_use, df_discard = train_test_split(df, test_size=0.8, random_state=42)

    train_val, test = train_test_split(df_use, test_size=test_ratio, random_state=42)
    train, val = train_test_split(train_val, test_size=val_ratio/(train_ratio + val_ratio), random_state=42)

    return train, val, test


# In[6]:


class CustomDataset2(Dataset):
    def __init__(self, image_folder, dataframe, transform=None):
        self.image_folder = image_folder
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0] + '.jpg'  # Assuming the image column is 'Image_ID'
        gender = torch.tensor(self.dataframe.iloc[idx,2])

        # Lazy loading: return the image path and label instead of loading the image
        return img_name, gender

    def load_image(self, img_name):
        img_path = self.image_folder + '/' + img_name
        image = Image.open(img_path).convert('RGB')
        #need a transformer for test images just to convert to tensor and to resahape and allat
        if self.transform == 'train' or self.transform == 'val':
            # Define transformations for data augmentation
            transform = transforms.Compose([
                transforms.Resize((100, 100)),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(20),
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1), 
                transforms.Normalize(mean=[0.5], std=[0.5]),
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


# In[7]:


data_path = '../data/UTKFace'
batch_size=32
df = pd.read_csv('../data/UTKFace_labels.csv', dtype={'Age':'float32', 'Gender':'float32'})
train_data, val_data, test_data = split_data(df) 


# In[8]:


train_dataset = CustomDataset2(dataframe=train_data, image_folder=data_path, transform='train')
val_dataset = CustomDataset2(dataframe=val_data, image_folder=data_path, transform='val')
test_dataset = CustomDataset2(dataframe=test_data, image_folder=data_path, transform='test')


# In[9]:


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# In[10]:


# Iterate through the train_loader to get a batch of data
for batch_idx, (img_names, targets) in enumerate(train_loader):
    data = [train_dataset.load_image(img_name) for img_name in img_names]
    example_data = data  # This will contain a batch of images
    example_targets = targets  # This will contain the corresponding labels/targets
    break


# In[11]:


import matplotlib.pyplot as plt
import numpy as np  # If needed for conversion
import random

# Assuming image_tensor is your tensor data
# Convert tensor to NumPy array
image_array = example_data[random.randint(0,31)].numpy()  # PyTorch example; adjust accordingly for TensorFlow

# Reverse the normalization by scaling back to the original range [0, 255]
image_array = image_array * 255.0

# Clip values to ensure they are within the valid range [0, 255]
image_array = np.clip(image_array, 0, 255)

# Convert data type to unsigned 8-bit integer (uint8)
image_array = image_array.astype(np.uint8)

# If the tensor has color channels as the first dimension (e.g., [C, H, W])
# and you want to rearrange it to be [H, W, C] for display:
if len(image_array.shape) == 3:
    image_array = np.transpose(image_array, (1, 2, 0))  # Change the order of dimensions

# Display the image using Matplotlib
plt.imshow(image_array)
plt.axis('off')  # Hide axis
plt.show()


# In[12]:


tf.print(example_data)


# In[13]:


tf.print(example_targets)


# In[14]:


import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
#self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.dense_shared = nn.Linear(50 * 50 * 64, 128)  # Calculate the input size based on your input_shape

        # Output layers
        self.classification_output = nn.Linear(128, 1)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Identity()  # No activation for linear output

    def forward(self, x):
        # Forward pass through convolutional layers
        x = self.relu(self.conv1(x))
        # x = self.relu(self.conv2(x))
        x = self.max_pool(x)

        # Flatten and pass through fully connected layers
        x = self.flatten(x)
        x = self.relu(self.dense_shared(x))

        # Classification branch
        classification_out = self.sigmoid(self.classification_output(x)).squeeze()
        
        return classification_out


# In[15]:


model = CustomModel()
loss_func = nn.BCELoss()


# In[16]:


# Define your optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)  # You can adjust the learning rate as needed

train_losses = []
val_losses = []
# Training loop
epochs = 10  # Define the number of epochs for training
for epoch in range(epochs):
    print("epoch:", epoch)
    model.train()  # Set the model to training mode
    total_loss = 0.0

    for batch_idx, (img_names, targets) in enumerate(train_loader):
        print(batch_idx)
        data = torch.stack([train_dataset.load_image(img_name) for img_name in img_names])
        # print(type(data))
        optimizer.zero_grad()  # Zero the gradients to prevent accumulation
        pred = model(data)  # Forward pass
         
        loss = loss_func(pred, targets)
        total_loss += loss.item()
        # Backpropagation
        loss.backward()
        optimizer.step()


    # Calculate average loss for the epoch
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_train_loss:.4f}")

    # Validation phase
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        total_val_loss = 0.0
        for batch_idx, (img_names, targets) in enumerate(val_loader):
            data = torch.stack([train_dataset.load_image(img_name) for img_name in img_names])
            pred = model(data)
            loss = loss_func(pred, targets)
            total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch [{epoch + 1}/{epochs}] - Validation Loss: {avg_val_loss:.4f}")
    

torch.save(model.state_dict(), 'gender_baseline_weights.pth')
print("Model weights saved successfully.")


# In[17]:


# Plotting losses over epochs
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


# In[18]:


# Set the model to evaluation mode
model.eval()

# Initialize variables to track correct predictions and count labels
total_correct = 0
total_samples = 0
label_0_count = 0
label_1_count = 0

# Iterate through the test dataset
with torch.no_grad():
    for batch_idx, (img_names, targets) in enumerate(test_loader):
        data = torch.stack([test_dataset.load_image(img_name) for img_name in img_names])
        predictions = model(data)  # Get model predictions
        predictions = (predictions > 0.5).float()  # Apply a threshold (assuming it's a binary classification)

        for i in range(len(targets)):
            if predictions[i] == targets[i]:
                total_correct += 1

            # Count occurrences of label 0 and label 1
            if targets[i] == 0:
                label_0_count += 1
            elif targets[i] == 1:
                label_1_count += 1

        total_samples += len(targets)

# Calculate overall accuracy
accuracy = total_correct / total_samples

# Calculate the ratio between labels 0 and 1
total_label_count = label_0_count + label_1_count
label_0_ratio = label_0_count / total_label_count
label_1_ratio = label_1_count / total_label_count

# Adjust random chance according to label ratio
random_chance = label_1_ratio  # Assuming label 1 is the positive class

# Print label counts and random chance based on label ratio
print(f"Label 0 count: {label_0_count}")
print(f"Label 1 count: {label_1_count}")
print(f"Random chance based on label ratio: {random_chance * 100:.2f}%")

# Print accuracy
print(f"Accuracy on test set: {accuracy * 100:.2f}%")

# Compare accuracy against random chance
if accuracy > random_chance:
    print("Model performs better than random chance.")
else:
    print("Model performs no better than random chance.")


# In[19]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

# Initialize variables to store predictions and actual labels
all_predictions = []
all_targets = []

# Iterate through the test dataset
with torch.no_grad():
    for batch_idx, (img_names, targets) in enumerate(test_loader):
        data = torch.stack([test_dataset.load_image(img_name) for img_name in img_names])
        predictions = model(data)  # Get model predictions
        predictions = (predictions > 0.5).float()  # Apply a threshold (assuming it's a binary classification)

        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

# Convert lists to NumPy arrays
all_predictions = np.array(all_predictions)
all_targets = np.array(all_targets)

# Convert predictions to binary (0 or 1)
y_class_pred_binary = (all_predictions > 0.5).astype(int)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(all_targets, y_class_pred_binary)

# Plot the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Female', 'Male'], yticklabels=['Female', 'Male'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# In[42]:


import torch
import torch.nn.functional as F

# Get the first batch from the test loader
batch_idx, (img_names, targets) = next(enumerate(test_loader))
data = torch.stack([test_dataset.load_image(img_name) for img_name in img_names])

# Get the gradients of the output with respect to the input
data.requires_grad = True
output = model(data)
loss = torch.nn.functional.mse_loss(output, targets)
loss.backward()

# Calculate the salience map from the gradients
gradients = data.grad
normalized_gradients = F.normalize(gradients.abs(), p=1, dim=(2, 3))
salience_map = normalized_gradients.max(dim=1)[0].squeeze().detach().cpu().numpy()

# Overlay salience map on the original image
fig, axes = plt.subplots(data.shape[0], 3, figsize=(12, 4 * data.shape[0]))  # 3 columns for the original image, salience map, and overlay

for i in range(data.shape[0]):
    # Plot the original image
    original_image = data[i].squeeze().detach().cpu().numpy()
    axes[i, 0].imshow(original_image, cmap='gray')
    axes[i, 0].set_title('Original Image')
    axes[i, 0].axis('off')

    # Plot the salience map
    axes[i, 1].imshow(salience_map[i], cmap='hot', interpolation='nearest')
    axes[i, 1].set_title('Salience Map')
    axes[i, 1].axis('off')

    # Overlay salience map on the original image
    axes[i, 2].imshow(original_image, cmap='gray')
    axes[i, 2].imshow(salience_map[i], cmap='hot', alpha=0.5, interpolation='nearest')  # Overlay the salience map
    axes[i, 2].set_title('Original Image with Salience Overlay')
    axes[i, 2].axis('off')

    # Check if the prediction is correct and add text to the title
    with torch.no_grad():
        prediction = torch.tensor(model(data[i].unsqueeze(0)).item())  # Convert the prediction to a PyTorch tensor
        predicted_gender = round(prediction.item())  # Round the predicted gender
        actual_gender = targets[i].item()

    # Map gender labels to strings
    gender_mapping = {0: 'Male', 1: 'Female'}
    predicted_gender_str = gender_mapping.get(predicted_gender, 'Unknown')
    actual_gender_str = gender_mapping.get(actual_gender, 'Unknown')

    # Add information to the titles
    axes[i, 0].set_title(f'Original Image\nActual Gender: {actual_gender_str}')
    axes[i, 1].set_title(f'Salience Map\nPredicted Gender: {predicted_gender_str}')
    axes[i, 2].set_title(f'Overlay\nPredicted Gender: {predicted_gender_str}')

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()

