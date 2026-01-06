import torch
import typer
import numpy as np
from mlops_m6_project.model import Classifier
from mlops_m6_project.data import corrupt_mnist


def evaluate(model_checkpoint: str = "models/model.pth") -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    state_dict = torch.load(model_checkpoint)
    model = Classifier()
    model.load_state_dict(state_dict)
    model.eval()
    _, test_ds = corrupt_mnist()
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)

    all_equals = np.array([])

    for images, labels in test_loader:
        with torch.no_grad():
            log_ps = model(images)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            all_equals = np.concatenate((all_equals, equals.numpy().flatten()))
    accuracy = np.mean(all_equals)
    print(f"Test Accuracy: {accuracy * 100}%")


if __name__ == "__main__":
    typer.run(evaluate)
