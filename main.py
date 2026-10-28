import numpy as np
import torch
from torch import nn

from dataset import get_mnist_dataloaders
from model import MNISTCNY19
from monitor import BoxMonitor as OriginalBoxMonitor


from rotatedbox import LayerMonitor, RotatedBox

# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dynamic configuration settings
CONFIG = {
    "known_classes": [0, 1, 2, 3, 4, 5],  # Digits 0-5 are known, 6-9 are novelties
    "train_ratio": 0.60,                  # 60% of knowns for network training
    "monitor_ratio": 0.20,                # 20% of knowns for monitor creation
    "seed": 42
}

print("Using device:", DEVICE)

# ============================================================
# LOAD MNIST
# ============================================================

# Fetch the DataLoaders in one line
model_train_loader, monitor_creation_loader, monitor_test_loader = get_mnist_dataloaders(
    known_classes=CONFIG["known_classes"],
    train_ratio=CONFIG["train_ratio"],
    monitor_ratio=CONFIG["monitor_ratio"],
    batch_size=BATCH_SIZE,
    seed=CONFIG["seed"]
)

# Verify the setup
print(f"Model Training Batches   : {len(model_train_loader)}")
print(f"Monitor Creation Batches : {len(monitor_creation_loader)}")
print(f"Testing Batches          : {len(monitor_test_loader)}")

# ============================================================
# CREATE MODEL
# ============================================================

model = MNISTCNY19(n_classes=len(CONFIG["known_classes"]))#MNISTNetwork()
model = model.to(DEVICE)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ============================================================
# TRAIN THE NEURAL NETWORK
# ============================================================

print("\nTraining neural network...\n")

for epoch in range(EPOCHS):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in model_train_loader:
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

# print("\nTesting neural network...\n")

# model.eval()

# correct = 0
# total = 0

# with torch.no_grad():
#     for images, labels in test_loader:
#         images = images.to(DEVICE)
#         labels = labels.to(DEVICE)
#         outputs = model(images)
#         predictions = outputs.argmax(dim=1)
#         correct += (predictions == labels).sum().item()
#         total += labels.size(0)

# test_accuracy = 100 * correct / total

# print(f"Normal MNIST test accuracy: {test_accuracy:.2f}%")


# ============================================================
# COLLECT HIDDEN REPRESENTATIONS
# ============================================================

print("\nCollecting hidden representations...\n")

all_representations = []
all_labels = []

with torch.no_grad():
    for images, labels in monitor_creation_loader:
        images = images.to(DEVICE)
        outputs, hidden = model(images, return_hidden=True)
        # Move from GPU to CPU
        hidden = hidden.cpu().numpy()
        labels = labels.numpy()
        all_representations.append(hidden)
        all_labels.append(labels)

all_representations = np.concatenate(all_representations, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

print("Hidden representation shape:", all_representations.shape)
print("Example hidden vector:", all_representations[0])

# ============================================================
# COLLECT TEST REPRESENTATIONS
# ============================================================

print("\nCollecting test representations...\n")

test_representations = []
test_labels = []

with torch.no_grad():
    for images, labels in monitor_test_loader:
        images = images.to(DEVICE)
        outputs, hidden = model(images, return_hidden=True)

        test_representations.append(hidden.cpu().numpy())
        test_labels.append(labels.numpy())

test_representations = np.concatenate(test_representations, axis=0)
test_labels = np.concatenate(test_labels, axis=0)

test_predictions = []

with torch.no_grad():
    for images, labels in monitor_test_loader:
        images = images.to(DEVICE)
        outputs, hidden = model(images, return_hidden=True)
        test_predictions.append(outputs.argmax(dim=1).cpu().numpy())

test_predictions = np.concatenate(test_predictions)



# ============================================================
# 1. NORMAL BOX MONITOR
# ============================================================

print("\n========== NORMAL BOX MONITOR ==========\n")

normal_monitor = LayerMonitor(
    shape_cls=Box,
    threshold=0.07,
    epsilon=0.0
)

normal_monitor.fit(
    all_representations,
    all_labels,
    n_classes=len(CONFIG["known_classes"])
)

normal_accepted = []

with torch.no_grad():
    for images, labels in monitor_test_loader:
        images = images.to(DEVICE)
        outputs, hidden = model(images, return_hidden=True)

        normal_accepted.extend(
            normal_monitor.accepts(
                hidden.cpu().numpy(),
                outputs.argmax(dim=1).cpu().numpy()
            )
        )

normal_accepted = np.asarray(normal_accepted, dtype=bool)

print(
    f"Normal box acceptance rate: "
    f"{100 * normal_accepted.mean():.2f}%"
)


# ============================================================
# 2. ROTATED BOX MONITOR
# ============================================================

print("\n========== ROTATED BOX MONITOR ==========\n")

rotated_monitor = LayerMonitor(
    shape_cls=RotatedBox,
    threshold=0.07,
    epsilon=0.0
)

rotated_monitor.fit(
    all_representations,
    all_labels,
    n_classes=len(CONFIG["known_classes"])
)

rotated_accepted = []

with torch.no_grad():
    for images, labels in monitor_test_loader:
        images = images.to(DEVICE)
        outputs, hidden = model(images, return_hidden=True)

        rotated_accepted.extend(
            rotated_monitor.accepts(
                hidden.cpu().numpy(),
                outputs.argmax(dim=1).cpu().numpy()
            )
        )

rotated_accepted = np.asarray(rotated_accepted, dtype=bool)

print(
    f"Rotated box acceptance rate: "
    f"{100 * rotated_accepted.mean():.2f}%"
)


# ============================================================
# 3. COMPARE RESULTS
# ============================================================

print("\n========== COMPARISON ==========")
print(f"Normal boxes:  {100 * normal_accepted.mean():.2f}%")
print(f"Rotated boxes: {100 * rotated_accepted.mean():.2f}%")

"""
# ============================================================
# CREATE BOX MONITOR
# ============================================================

print("\nCreating box monitor...\n")
monitor = BoxMonitor()
monitor.fit(all_representations, all_labels)

print("\nBoxes created.")

for class_id, box in monitor.boxes.items():
    print(f"Class {class_id}: {len(box.lower)} dimensions")


# ============================================================
# TEST THE MONITOR
# ============================================================

print("\nTesting monitor...\n")

known_correct = 0
known_total = 0

with torch.no_grad():
    for images, labels in monitor_test_loader:
        images = images.to(DEVICE)
        outputs, hidden = model(images, return_hidden=True)
        predictions = outputs.argmax(dim=1)
        hidden = hidden.cpu().numpy()
        predictions = predictions.cpu().numpy()

        for i in range(len(images)):
            predicted_class = int(predictions[i])
            representation = hidden[i]
            accepted = monitor.check(representation, predicted_class)

            if accepted:
                known_correct += 1

            known_total += 1

monitor_acceptance = 100 * known_correct / known_total

print(f"Known test samples accepted by monitor: {monitor_acceptance:.2f}%")
"""