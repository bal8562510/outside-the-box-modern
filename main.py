import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import numpy as np

from model import MNISTNetwork
from monitor import BoxMonitor


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# ============================================================
# LOAD MNIST
# ============================================================

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=ToTensor()
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# CREATE MODEL
# ============================================================

model = MNISTNetwork()
model = model.to(DEVICE)

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAIN THE NEURAL NETWORK
# ============================================================

print("\nTraining neural network...\n")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Clear old gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = loss_function(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"| Loss: {total_loss:.3f} "
        f"| Accuracy: {accuracy:.2f}%"
    )


# ============================================================
# NORMAL TEST ACCURACY
# ============================================================

print("\nTesting neural network...\n")

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)


test_accuracy = 100 * correct / total

print(
    f"Normal MNIST test accuracy: "
    f"{test_accuracy:.2f}%"
)


# ============================================================
# COLLECT HIDDEN REPRESENTATIONS
# ============================================================

print("\nCollecting hidden representations...\n")

all_representations = []
all_labels = []

with torch.no_grad():

    for images, labels in train_loader:

        images = images.to(DEVICE)

        outputs, hidden = model(
            images,
            return_hidden=True
        )

        # Move from GPU to CPU
        hidden = hidden.cpu().numpy()

        labels = labels.numpy()

        all_representations.append(hidden)
        all_labels.append(labels)


all_representations = np.concatenate(
    all_representations,
    axis=0
)

all_labels = np.concatenate(
    all_labels,
    axis=0
)


print(
    "Hidden representation shape:",
    all_representations.shape
)

print(
    "Example hidden vector:",
    all_representations[0]
)


# ============================================================
# CREATE BOX MONITOR
# ============================================================

print("\nCreating box monitor...\n")

monitor = BoxMonitor()

monitor.fit(
    all_representations,
    all_labels
)


print("\nBoxes created.")

for class_id, box in monitor.boxes.items():

    print(
        f"Class {class_id}: "
        f"{len(box.lower)} dimensions"
    )


# ============================================================
# TEST THE MONITOR
# ============================================================

print("\nTesting monitor...\n")

known_correct = 0
known_total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs, hidden = model(
            images,
            return_hidden=True
        )

        predictions = outputs.argmax(dim=1)

        hidden = hidden.cpu().numpy()
        predictions = predictions.cpu().numpy()

        for i in range(len(images)):

            predicted_class = int(predictions[i])

            representation = hidden[i]

            accepted = monitor.check(
                representation,
                predicted_class
            )

            if accepted:
                known_correct += 1

            known_total += 1


monitor_acceptance = (
    100 * known_correct / known_total
)

print(
    f"Known test samples accepted by monitor: "
    f"{monitor_acceptance:.2f}%"
)