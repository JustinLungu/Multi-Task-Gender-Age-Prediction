import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle as pkl
from DataRetriever import DataRetriever
from MultiTaskModel import MultiTaskModel
from MultiTaskLossWrapper import MultiTaskLossWrapper
import evaluation

def plot_losses(train_losses, val_losses):
    # Plotting losses over epochs
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_graph.png')

def run_model():
    model = MultiTaskModel()
    loss_func = MultiTaskLossWrapper(2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # You can adjust the learning rate as needed

    batch_size = 32
    df_path = '../data/UTKFace_labels.csv'
    image_folder_path = '../data/UTKFace'
    data_percentage = 0.5

    dr = DataRetriever(df_path, image_folder_path, batch_size, data_percentage)
    train_dataset, val_dataset, test_dataset = dr.retrieve_datasets()
    train_loader, val_loader, test_loader = dr.retrieve_loaders()

    # Training loop
    val_losses = []
    train_losses = []
    epochs = 20
    for epoch in range(epochs):
        print("epoch:", epoch)
        model.train()  # Set the model to training mode
        total_loss = 0.0

        for batch_idx, (img_names, targets) in enumerate(train_loader):
            print(batch_idx)
            data = torch.stack([train_dataset.load_image(img_name) for img_name in img_names])
            optimizer.zero_grad()  # Zero the gradients to prevent accumulation
            age_pred, gen_pred = model(data)
            age_true = targets[0]
            gen_true = targets[1]
            loss = loss_func(age_pred, gen_pred, age_true, gen_true)
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
                age_pred, gen_pred = model(data)  # Forward pass
                age_true = targets[0]
                gen_true = targets[1]
                loss = loss_func(age_pred, gen_pred, age_true, gen_true)
                total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            print(f"Epoch [{epoch + 1}/{epochs}] - Validation Loss: {avg_val_loss:.4f}")

    plot_losses(train_losses, val_losses)

    # Save the trained model using pickle
    filename = 'trained_multi_model.pkl'
    with open(filename, 'wb') as file:
        pkl.dump(model, file)

    # Convert to state dictionary
    state_dict = model.state_dict()

    # Save as PyTorch serialized model
    torch.save(state_dict, 'trained_multi_model.pth')

    model.eval()
    evaluation.evaluate_model(model, train_loader, test_dataset, test_loader)