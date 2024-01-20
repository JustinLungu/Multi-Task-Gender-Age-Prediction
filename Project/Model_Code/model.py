import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle as pkl
from DataRetriever import DataRetriever
from MultiTaskModel import MultiTaskModel
from MultiTaskLossWrapper import MultiTaskLossWrapper
from BaselineGenderModel import BaselineGenderModel
from BaselineAgeModel import BaselineAgeModel
import evaluation
import os

def plot_losses(train_losses, val_losses):
    # Plotting losses over epochs
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_graph.png')

def run_model(model_type, data_usage, split):
    if model_type == 'final':
        model = MultiTaskModel()
        loss_func = MultiTaskLossWrapper(2)
    elif model_type == 'age_baseline':
        model = BaselineAgeModel()
        loss_func = nn.MSELoss()
    elif model_type == 'gender_baseline':
        model = BaselineGenderModel()
        loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # You can adjust the learning rate as needed

    batch_size = 32
    # Get the current working directory
    current_directory = os.getcwd()
    # Print the current working directory
    print("Current Directory:", current_directory)
    image_folder_path = os.path.join(current_directory, 'Data', 'UTKFace')
    df_path = os.path.join(current_directory, 'Data', 'UTKFace_labels.csv')
    data_percentage = data_usage

    # write split_done = True if you already have csv files of the split data in ../data/datasets/
    # write False if you want these files to be created
    dr = DataRetriever(model_type, df_path, image_folder_path, batch_size, data_percentage, split_done=split)
    train_dataset, val_dataset, test_dataset = dr.retrieve_datasets()
    train_loader, val_loader, test_loader = dr.retrieve_loaders()

    # Training loop
    val_losses = []
    train_losses = []
    epochs = 5
    for epoch in range(epochs):
        print("epoch:", epoch)
        model.train()  # Set the model to training mode
        total_loss = 0.0

        for batch_idx, (img_names, targets) in enumerate(train_loader):
            print(batch_idx)
            data = torch.stack([train_dataset.load_image(img_name) for img_name in img_names])
            optimizer.zero_grad()  # Zero the gradients to prevent accumulation

            if model_type == 'final':
                age_pred, gen_pred = model(data)
                age_true = targets[0]
                gen_true = targets[1]
                loss = loss_func(age_pred, gen_pred, age_true, gen_true)
            else:
                pred = model(data)  # Forward pass
                loss = loss_func(pred,
                                 targets[0])  # the [0] is necessary to take the tensor out of the one-element list

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
                data = torch.stack([val_dataset.load_image(img_name) for img_name in img_names])

                if model_type == 'final':
                    age_pred, gen_pred = model(data)
                    age_true = targets[0]
                    gen_true = targets[1]
                    loss = loss_func(age_pred, gen_pred, age_true, gen_true)
                else:
                    pred = model(data)  # Forward pass
                    loss = loss_func(pred,
                                     targets[0])  # the [0] is necessary to take the tensor out of the one-element list
                total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            print(f"Epoch [{epoch + 1}/{epochs}] - Validation Loss: {avg_val_loss:.4f}")

    plot_losses(train_losses, val_losses)

    # Save the trained model using pickle
    filename = model_type + '_model'
    with open(filename + '.pkl', 'wb') as file:
        pkl.dump(model, file)

    # Convert to state dictionary
    state_dict = model.state_dict()

    # Save as PyTorch serialized model
    torch.save(state_dict, filename + '.pth')

    model.eval()
    evaluation.evaluate_model(model_type, model, train_loader, test_dataset, test_loader)