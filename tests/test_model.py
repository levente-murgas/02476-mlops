import pytest
import torch
from mlops_m6_project.model_lightning import Classifier


@pytest.mark.parametrize(
    "input_tensor",
    [
        torch.randn(16, 1, 28, 28),  # valid input
        torch.randn(32, 1, 28, 28),  # valid input with different batch size
    ],
)
def test_forward(input_tensor):
    """Test the forward method of the Classifier model."""
    model = Classifier()
    batch_size = input_tensor.shape[0]
    output = model(input_tensor)
    assert output.shape == (batch_size, 10), "Output shape is incorrect."


def test_forward_raises():
    """Test that the forward method raises ValueError for incorrect input shapes."""
    model = Classifier()
    # Test for 3D tensor
    invalid_input_3d = torch.randn(16, 28, 28)
    with pytest.raises(ValueError):
        model(invalid_input_3d)

    # Test for incorrect channel/height/width
    invalid_input_shape = torch.randn(16, 3, 28, 28)
    with pytest.raises(ValueError):
        model(invalid_input_shape)
