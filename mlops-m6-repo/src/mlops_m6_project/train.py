import os
import numpy as np
import typer
import torch
import hydra
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
import wandb
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env file
wandb.login()

from mlops_m6_project.model import Classifier
from mlops_m6_project.data import corrupt_mnist

# Get the absolute path to configs directory
config_path = str(Path(__file__).parent.parent.parent / "configs")

@hydra.main(version_base=None, config_path=config_path, config_name="config.yaml")
def train(cfg) -> float:
    """Train a model on Corrupt MNIST.

    Parameters:
        lr (float): Learning rate for the optimizer.
        epochs (int): Number of epochs to train the model.
        batch_size (int): Batch size for training.
    Returns:
        float: Maximum training accuracy achieved during training.
    """
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger.add(f"{output_dir}/mytrain.log")

    model_params = cfg.model
    training_params = cfg.training
    combined_params = {**model_params, **training_params}
    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY"),
        project=os.environ.get("WANDB_PROJECT"),
        config=combined_params,
    )
    train_set, test_set = corrupt_mnist()
    model = Classifier(**model_params)
    # add rest of your training code here
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=training_params.batch_size, shuffle=True)
    lr = training_params.lr
    epochs = training_params.epochs
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lowest_loss = float("inf")
    running_loss = 0.0
    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = model(img)
            loss = criterion(y_pred, target)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)
            run.log({"train_loss": loss.item(), "train_accuracy": accuracy})

            if i % 100 == 0:
                logger.info(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")
        if running_loss / len(train_loader) < lowest_loss:
            lowest_loss = running_loss / len(train_loader)
            torch.save(model.state_dict(), "models/model.pth")

    logger.info("Training complete")
    torch.save(model.state_dict(), "models/model.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")
    run.log({"training_statistics": wandb.Image(fig)})
    accs = np.array(statistics["train_accuracy"])
    artifact = wandb.Artifact(name="cnn", type="model")
    artifact.add_file("models/model.pth")
    run.log_artifact(artifact)

    return np.max(accs)


if __name__ == "__main__":
    # typer.run(train)
    train()
