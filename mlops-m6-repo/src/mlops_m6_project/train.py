import numpy as np
import typer
import torch
import matplotlib.pyplot as plt

from mlops_m6_project.model import Classifier
from mlops_m6_project.data import corrupt_mnist


def train(lr: float = 1e-3, epochs: int = 5, batch_size: int = 64) -> float:
    """Train a model on Corrupt MNIST.

    Parameters:
        lr (float): Learning rate for the optimizer.
        epochs (int): Number of epochs to train the model.
        batch_size (int): Batch size for training.
    Returns:
        float: Maximum training accuracy achieved during training.
    """
    train_set, test_set = corrupt_mnist()
    model = Classifier()
    # add rest of your training code here
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    lr = 1e-3
    epochs = 5
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

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")
        if running_loss / len(train_loader) < lowest_loss:
            lowest_loss = running_loss / len(train_loader)
            torch.save(model.state_dict(), "models/model.pth")

    print("Training complete")
    torch.save(model.state_dict(), "models/model.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")
    accs = np.array(statistics["train_accuracy"])
    return np.max(accs)


if __name__ == "__main__":
    typer.run(train)
