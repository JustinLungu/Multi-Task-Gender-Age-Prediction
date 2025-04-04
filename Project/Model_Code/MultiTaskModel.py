import torch.nn as nn

class MultiTaskModel(nn.Module):
    def __init__(self):
        super(MultiTaskModel, self).__init__()
        # TinyVGG Architecture
        # Convolutional layer set 1
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu1_2 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=2)

        # Convolutional layer set 2
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu2_2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.dense_shared = nn.Linear(25 * 25 * 64, 128)  # Calculate the input size based on your input_shape

        # Output layers
        self.classification_output = nn.Linear(128, 1)
        self.regression_output = nn.Linear(128, 1)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Identity()  # No activation for linear output

    def forward(self, x):
        # Forward pass through convolutional layer set 1
        x = self.relu1_1(self.conv1_1(x))
        x = self.relu1_2(self.conv1_2(x))
        x = self.max_pool1(x)

        # Forward pass through convolutional layer set 2
        x = self.relu2_1(self.conv2_1(x))
        x = self.relu2_2(self.conv2_2(x))
        x = self.max_pool2(x)

        # Flatten and pass through fully connected layers
        x = self.flatten(x)
        x = self.relu(self.dense_shared(x))

        # Classification branch
        classification_out = self.sigmoid(self.classification_output(x)).squeeze()

        # Regression branch
        regression_out = self.linear(self.regression_output(x)).squeeze()

        if regression_out.dim() == 0:
            regression_out = regression_out.unsqueeze(0)

        if classification_out.dim() == 0:
            classification_out = classification_out.unsqueeze(0)  # Convert the single value to a list

        return regression_out, classification_out