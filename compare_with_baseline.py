
import h5py
import numpy as np
import torch
from torchvision import datasets, transforms

from model import MNISTCNY19


WEIGHTS_PATH = r"C:\work\original\models\CNY19id1_MNIST_0-2.h5"


def load_keras_weights(model, path):
    """Load the original Keras weights into the matching PyTorch layers."""
    with h5py.File(path, "r") as f:
        g = f["model_weights"]
        weighted_layers = []

        for name in g.attrs["layer_names"]:
            name = name.decode() if isinstance(name, bytes) else name
            weight_names = g[name].attrs["weight_names"]

            if len(weight_names) == 0:
                continue

            arrays = []
            for weight_name in weight_names:
                weight_name = (
                    weight_name.decode()
                    if isinstance(weight_name, bytes)
                    else weight_name
                )
                arrays.append(np.asarray(g[name][weight_name]))

            weighted_layers.append(arrays)

    assert len(weighted_layers) == 7, (
        f"Expected 7 weighted layers, found {len(weighted_layers)}"
    )

    layers = [
        model.conv1,
        model.conv2,
        model.fc1,
        model.fc2,
        model.fc3,
        model.fc4,
        model.out,
    ]

    with torch.no_grad():
        for i, (layer, weights) in enumerate(
            zip(layers, weighted_layers)
        ):
            kernel, bias = weights

            if i < 2:
                # Keras Conv2D: (height, width, in, out)
                # PyTorch Conv2d: (out, in, height, width)
                kernel = kernel.transpose(3, 2, 0, 1).copy()

            else:
                if i == 2:
                    # Keras Flatten uses H,W,C order.
                    # PyTorch Flatten uses C,H,W order.
                    kernel = (
                        kernel.reshape(4, 4, 20, -1)
                        .transpose(2, 0, 1, 3)
                        .reshape(320, -1)
                    )

                # Keras Dense: (in_features, out_features)
                # PyTorch Linear: (out_features, in_features)
                kernel = kernel.T.copy()

            expected_shape = tuple(layer.weight.shape)
            assert kernel.shape == expected_shape, (
                f"Layer {i}: expected {expected_shape}, "
                f"got {kernel.shape}"
            )
            assert bias.shape == tuple(layer.bias.shape), (
                f"Bias shape mismatch in layer {i}"
            )

            layer.weight.copy_(
                torch.as_tensor(kernel, dtype=layer.weight.dtype)
            )
            layer.bias.copy_(
                torch.as_tensor(bias, dtype=layer.bias.dtype)
            )

    print("Original Keras weights loaded successfully.")
    print("Verified weighted layers:", len(weighted_layers))


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = MNISTCNY19(n_classes=3)
    load_keras_weights(model, WEIGHTS_PATH)
    model.to(device)
    model.eval()

    # ToTensor maps pixels to [0,1].
    # Normalize to the original pipeline's [-1,1] range.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    # Evaluate only original classes 0, 1 and 2.
    indices = [
        i for i, label in enumerate(test_dataset.targets.tolist())
        if label in (0, 1, 2)
    ]

    test_subset = torch.utils.data.Subset(test_dataset, indices)
    loader = torch.utils.data.DataLoader(
        test_subset, batch_size=256, shuffle=False
    )

    correct = 0
    total = 0
    class_correct = {0: 0, 1: 0, 2: 0}
    class_total = {0: 0, 1: 0, 2: 0}

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            predictions = logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.numel()

            for digit in (0, 1, 2):
                mask = labels == digit
                class_total[digit] += mask.sum().item()
                class_correct[digit] += (
                    (predictions[mask] == labels[mask]).sum().item()
                )

    print(f"\nTest samples (digits 0, 1, 2): {total}")
    print(f"Overall accuracy: {100 * correct / total:.2f}%")

    for digit in (0, 1, 2):
        accuracy = 100 * class_correct[digit] / class_total[digit]
        print(
            f"Digit {digit} accuracy: {accuracy:.2f}% "
            f"({class_correct[digit]}/{class_total[digit]})"
        )


if __name__ == "__main__":
    main()
