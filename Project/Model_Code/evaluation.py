import torch
import torch.nn.functional as F
def evaluate_model(model_type, model, train_loader, test_dataset, test_loader):
    if model_type == 'age_baseline':
        evaluate_age_baseline(model, train_loader, test_dataset, test_loader)
    elif model_type == 'gender_baseline':
        evaluate_gender_baseline(model, train_loader, test_dataset, test_loader)
    elif model_type == 'final':
        evaluate_final(model, train_loader, test_dataset, test_loader)

def evaluate_final(model, train_loader, test_dataset, test_loader):
    # Prediction loop
    predictions = []
    targets = []

    # Initialize variables to track correct predictions and count labels
    total_correct = 0
    total_samples = 0
    label_0_count = 0
    label_1_count = 0

    # Initialize variables for MSE calculation
    total_mse_model = 0.0
    total_mse_baseline = 0.0
    total_age_samples = 0

    # Calculate the mean value in the training set (to use as the baseline prediction)
    train_targets = []  # Collect training targets
    for _, targets in train_loader:
        train_targets.extend(targets[0].numpy())

    baseline_prediction = torch.tensor(train_targets).float().mean().item()

    with torch.no_grad():
        for (test_img_names, test_targets) in test_loader:
            test_data = torch.stack([test_dataset.load_image(img_name) for img_name in test_img_names])
            test_age_pred, test_gen_pred = model(test_data)  # Forward pass
            test_gen_pred = (test_gen_pred.squeeze() > 0.5).float()
            # print("Predictions", test_gen_pred)

            targets = test_targets[1]
            for i in range(len(targets)):
                if test_gen_pred[i] == targets[i]:
                    total_correct += 1

                # Count occurrences of label 0 and label 1
                if targets[i] == 0:
                    label_0_count += 1
                elif targets[i] == 1:
                    label_1_count += 1

            total_samples += len(targets)

            # Calculate MSE for model predictions
            mse_model = F.mse_loss(test_age_pred, test_targets[0], reduction='sum').item()
            total_mse_model += mse_model

            # Calculate MSE for baseline predictions (always predicting the mean)
            mse_baseline = F.mse_loss(torch.full_like(targets, baseline_prediction), targets, reduction='sum').item()
            total_mse_baseline += mse_baseline

            total_age_samples += len(targets)

    # Calculate overall accuracy
    accuracy = total_correct / total_samples

    # Calculate the ratio between labels 0 and 1
    total_label_count = label_0_count + label_1_count

    # Adjust random chance according to label ratio
    random_chance = max(label_0_count, label_1_count) / total_label_count  # Assuming label 1 is the positive class

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

    # Calculate average MSE for both model and baseline
    avg_mse_model = total_mse_model / total_age_samples
    avg_mse_baseline = total_mse_baseline / total_age_samples

    # Print MSE values
    print(f"Model MSE: {avg_mse_model:.4f}")
    print(f"Baseline (Predicting Mean) MSE: {avg_mse_baseline:.4f}")

    # Calculate and print the improvement in MSE over the baseline
    mse_improvement = avg_mse_baseline - avg_mse_model
    print(f"Improvement over baseline MSE: {mse_improvement:.4f}")

def evaluate_age_baseline(model, train_loader, test_dataset, test_loader):
    # Initialize variables for MSE calculation
    total_mse_model = 0.0
    total_mse_baseline = 0.0
    total_samples = 0

    # Calculate the mean value in the training set (to use as the baseline prediction)
    train_targets = []  # Collect training targets
    for _, targets in train_loader:
        targets = targets[0]
        train_targets.extend(targets.numpy())

    baseline_prediction = torch.tensor(train_targets).float().mean().item()

    # Iterate through the test dataset
    with torch.no_grad():
        for batch_idx, (img_names, targets) in enumerate(test_loader):
            targets = targets[0]
            data = torch.stack([test_dataset.load_image(img_name) for img_name in img_names])
            predictions = model(data)  # Get model predictions

            # Calculate MSE for model predictions
            mse_model = F.mse_loss(predictions, targets, reduction='sum').item()
            total_mse_model += mse_model

            # Calculate MSE for baseline predictions (always predicting the mean)
            mse_baseline = F.mse_loss(torch.full_like(targets, baseline_prediction), targets, reduction='sum').item()
            total_mse_baseline += mse_baseline

            total_samples += len(targets)

    # Calculate average MSE for both model and baseline
    avg_mse_model = total_mse_model / total_samples
    avg_mse_baseline = total_mse_baseline / total_samples

    # Print MSE values
    print(f"Model MSE: {avg_mse_model:.4f}")
    print(f"Baseline (Predicting Mean) MSE: {avg_mse_baseline:.4f}")

    # Calculate and print the improvement in MSE over the baseline
    mse_improvement = avg_mse_baseline - avg_mse_model
    print(f"Improvement over baseline MSE: {mse_improvement:.4f}")

def evaluate_gender_baseline(model, train_loader, test_dataset, test_loader):
    # Initialize variables to track correct predictions and count labels
    total_correct = 0
    total_samples = 0
    label_0_count = 0
    label_1_count = 0

    # Iterate through the test dataset
    with torch.no_grad():
        for batch_idx, (img_names, targets) in enumerate(test_loader):
            targets = targets[0]
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
