import torch
import torch.nn as nn


class MNISTCNY19(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 40, 5)
        self.conv2 = nn.Conv2d(40, 20, 5)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(320, 320)   # Dense 320
        self.fc2 = nn.Linear(320, 160)   # Dense 160  -> layer -4
        self.fc3 = nn.Linear(160, 80)    # Dense 80   -> layer -3
        self.fc4 = nn.Linear(80, 40)     # Dense 40   -> layer -2
        self.out = nn.Linear(40, n_classes)
        for m in self.modules():         # Keras defaults
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, return_layers=False):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        h160 = self.relu(self.fc2(x))
        h80 = self.relu(self.fc3(h160))
        h40 = self.relu(self.fc4(h80))
        logits = self.out(h40)
        if not return_layers:
            return logits
        # same layer indices as the original Keras model
        return logits, {-1: torch.softmax(logits, 1), -2: h40, -3: h80, -4: h160}