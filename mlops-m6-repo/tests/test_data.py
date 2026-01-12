import torch
import pytest
import os.path
from tests import _PATH_DATA
from mlops_m6_project.data import corrupt_mnist
from torch.utils.data import TensorDataset

@pytest.mark.skipif(not os.path.exists(_PATH_DATA) or not any(os.scandir(_PATH_DATA)), reason="Data folder not found or empty")
def test_my_dataset():
    """Test the MyDataset class."""
    train_set, test_set = corrupt_mnist()
    N_train = 30_000
    N_test = 5_000
    N_labels = 10
    assert len(train_set) == N_train, f"Expected {N_train} training samples, got {len(train_set)}"
    assert len(test_set) == N_test, f"Expected {N_test} test samples, got {len(test_set)}"
    for dataset in (train_set, test_set):
        assert isinstance(dataset, TensorDataset)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, drop_last=True)
        for imgs, targets in data_loader:
            assert imgs.shape == (16, 1, 28, 28)
            assert targets.shape == (16,)
            assert imgs.dtype == torch.float32
            assert targets.dtype == torch.int64
            assert targets.min().item() >= 0
            assert targets.max().item() < N_labels
    train_targets = torch.unique(train_set.tensors[1])
    assert (train_targets == torch.arange(0,10)).all(), "Training set does not contain all labels from 0 to 9"
    test_targets = torch.unique(test_set.tensors[1])
    assert (test_targets == torch.arange(0,10)).all(), "Test set does not contain all labels from 0 to 9"


def test_normalize():
    """Test the normalize function."""
    from mlops_m6_project.data import normalize

    # Create a dummy tensor
    images = torch.randn(100, 1, 28, 28) * 50 + 100  # mean ~100, std ~50
    normalized_images = normalize(images)

    # Check that the mean is approximately 0 and std is approximately 1
    mean = normalized_images.mean().item()
    std = normalized_images.std().item()
    assert pytest.approx(mean, abs=1e-5) == 0, f"Mean after normalization is not close to 0, got {mean}"
    assert pytest.approx(std, abs=1e-5) == 1, f"Std after normalization is not close to 1, got {std}"
