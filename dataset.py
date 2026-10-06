import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, transforms


def get_mnist_dataloaders(
    known_classes=None, # Do not use mutable data structures for argument defaults (by Ruff). but why?
    train_ratio=0.60,
    monitor_ratio=0.20,
    batch_size=64,
    seed=42,
    data_dir='./data'
):
    """
    Downloads, filters, splits, and returns 3 PyTorch DataLoaders:
    1. model_train_loader       : Known classes only (train_ratio portion)
    2. monitor_creation_loader  : Known classes only (monitor_ratio portion, unseen by network)
    3. test_loader              : Unseen known classes + ALL unknown classes
    """
    # Load full dataset
    if known_classes is None:
        known_classes = [0, 1, 2, 3, 4, 5]
    train_ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
    test_ds = datasets.MNIST(root=data_dir, train=False, download=True, transform=transforms.ToTensor())
    full_ds = ConcatDataset([train_ds, test_ds])

    # Extract targets in memory
    targets = torch.cat([train_ds.targets, test_ds.targets])

    # Dynamic mask for known vs unknown classes
    known_mask = torch.isin(targets, torch.tensor(known_classes))
    known_indices = torch.where(known_mask)[0]
    unknown_indices = torch.where(~known_mask)[0]

    # Shuffle known indices reproducibly
    gen = torch.Generator().manual_seed(seed)
    shuffled_known = known_indices[torch.randperm(len(known_indices), generator=gen)]

    # Compute dynamic split boundaries
    n_known = len(shuffled_known)
    train_end = int(train_ratio * n_known)
    monitor_end = int((train_ratio + monitor_ratio) * n_known)

    # Slice index subsets
    model_train_idx = shuffled_known[:train_end]
    monitor_creation_idx = shuffled_known[train_end:monitor_end]
    test_known_idx = shuffled_known[monitor_end:]

    # Merge remaining test knowns with ALL novel unknown classes
    test_combined_idx = torch.cat([test_known_idx, unknown_indices])

    # Create zero-copy PyTorch Subsets
    model_train_set = Subset(full_ds, model_train_idx)
    monitor_creation_set = Subset(full_ds, monitor_creation_idx)
    monitor_test_set = Subset(full_ds, test_combined_idx)

    # Construct DataLoaders
    model_train_loader = DataLoader(model_train_set, batch_size=batch_size, shuffle=True)
    monitor_creation_loader = DataLoader(monitor_creation_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(monitor_test_set, batch_size=batch_size, shuffle=False)

    return model_train_loader, monitor_creation_loader, test_loader