import os
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb

from mlops_m6_project.data import corrupt_mnist
from mlops_m6_project.model import Classifier

# Get the absolute path to configs directory
config_path = str(Path(__file__).parent.parent.parent / "configs")


@hydra.main(version_base=None, config_path=config_path, config_name="eval.yaml")
def evaluate(cfg) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print("Current working dir:", os.getcwd())

    api = wandb.Api()
    artifact_name = f"{cfg.artifact_name}"
    artifact = api.artifact(name=artifact_name)
    artifact.download(cfg.artifact_dir)
    model = Classifier()
    model.load_state_dict(torch.load(f"{cfg.artifact_dir}/model.pth"))
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
    evaluate()
