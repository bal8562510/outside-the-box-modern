import torch
import torch.nn as nn


class MNISTNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        # Second convolutional layer
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        # Fully connected hidden layer
        self.fc1 = nn.Linear(32 * 7 * 7, 40)

        # Final classification layer
        self.fc2 = nn.Linear(40, 10)

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x, return_hidden=False):

        # 28 x 28
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # 14 x 14
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # 7 x 7
        x = x.view(x.size(0), -1)

        # Hidden representation
        hidden = self.fc1(x)
        hidden = self.relu(hidden)

        # Final classification
        output = self.fc2(hidden)

        if return_hidden:
            return output, hidden

        return output