import torch
import typer
from data import corrupt_mnist
from model import Classifier
from matplotlib import pyplot as plt
import numpy as np

app = typer.Typer()


@app.command()
def train(lr: float = 1e-3, epochs: int = 5) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    model = Classifier()
    train_loader, _ = corrupt_mnist()

    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    training_loss = []
    lowest_loss = float('inf')
    training_steps = 0
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader)}")
        training_loss.append(running_loss/len(train_loader))
        if running_loss/len(train_loader) < lowest_loss:
            lowest_loss = running_loss/len(train_loader)
            torch.save(model.state_dict(), "model.pth")
        training_steps += 1

    plt.plot(range(training_steps), training_loss)
    plt.xlabel("Training Steps")
    plt.ylabel("Training Loss")
    plt.show()

@app.command()
def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    state_dict = torch.load(model_checkpoint)
    model = Classifier()
    model.load_state_dict(state_dict)
    model.eval()
    _, test_loader = corrupt_mnist()

    all_equals = np.array([])

    for images, labels in test_loader:
        with torch.no_grad():
            log_ps = model(images)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            all_equals = np.concatenate((all_equals, equals.numpy().flatten()))
    accuracy = np.mean(all_equals)
    print(f"Model Accuracy: {accuracy * 100}%")


if __name__ == "__main__":
    app()