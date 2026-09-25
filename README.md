# Outside the Box – Modern Educational Implementation

A beginner-friendly modern implementation of the core idea behind **Outside the Box: Abstraction-Based Monitoring of Neural Networks**.

This project is implemented using modern Python, PyTorch, torchvision, NumPy, and scikit-learn.

## Main idea

The neural network first predicts the class of an image.

A runtime monitor then checks whether the neural network's hidden representation looks similar to the representations seen during training.

The basic pipeline is:

```text
MNIST Image
     ↓
CNN
     ↓
40-dimensional hidden representation
     ↓
Class-specific box
     ↓
Runtime Monitor
     ↓
Inside / Outside
```

## Current implementation

The current version includes:

* MNIST dataset
* CNN-based neural network
* 40-dimensional hidden representation
* One axis-aligned box per class
* Runtime box checking
* Training and test evaluation

The current implementation is a simplified educational version and is **not an exact reproduction of the original research repository**.

## Model

The network contains:

```text
28 × 28 image
      ↓
Conv1
      ↓
16 feature maps
      ↓
Max Pooling
      ↓
Conv2
      ↓
32 feature maps
      ↓
Max Pooling
      ↓
32 × 7 × 7
      ↓
Flatten
      ↓
1568 values
      ↓
FC1
      ↓
40-dimensional representation
      ↓
FC2
      ↓
10 class scores
```

The 40-dimensional representation is used by the monitor.

## Results

Example run:

```text
Normal MNIST test accuracy: 98.71%

Hidden representation shape:
(60000, 40)

Known test samples accepted by monitor:
99.64%
```

These numbers may change slightly between runs.

## Installation

Create a Python environment:

```bash
conda create -n outside_box python=3.11 -y
conda activate outside_box
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

On Windows, if an OpenMP library conflict occurs, you may need:

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python main.py
```

## Project direction

Possible future extensions include:

* clustering
* multiple boxes per class
* the τ clustering parameter
* confidence scores
* multiple monitored layers
* rotated boxes
* PCA-based transformations
* comparison between axis-aligned and transformed boxes

## Reference

The project is inspired by:

**Outside the Box: Abstraction-Based Monitoring of Neural Networks**

Henzinger, Lukina, Schilling.

This repository is intended as a modern educational implementation of the core concept rather than a copy of the original research code.
