import torch
import typer
import matplotlib.pyplot as plt
from loguru import logger


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std()


def preprocess_data(raw_dir: str, processed_dir: str) -> None:
    """Process raw data and save it to processed directory."""
    train_images, train_target = [], []
    for i in range(6):
        train_images.append(torch.load(f"{raw_dir}/train_images_{i}.pt"))
        train_target.append(torch.load(f"{raw_dir}/train_target_{i}.pt"))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    test_images: torch.Tensor = torch.load(f"{raw_dir}/test_images.pt")
    test_target: torch.Tensor = torch.load(f"{raw_dir}/test_target.pt")

    # We add channel dimension, before it is (N, 28, 28), after it is (N, 1, 28, 28)
    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    train_images = normalize(train_images)
    test_images = normalize(test_images)

    torch.save(train_images, f"{processed_dir}/train_images.pt")
    torch.save(train_target, f"{processed_dir}/train_target.pt")
    torch.save(test_images, f"{processed_dir}/test_images.pt")
    torch.save(test_target, f"{processed_dir}/test_target.pt")


def dataset_statistics(dataset_path: str):
    train, test = MNISTDataset(dataset_path, train=True), MNISTDataset(dataset_path, train=False)
    logger.info(f"Train set size: {len(train)}")
    logger.info(f"Test set size: {len(test)}")
    # randomly take 12 samples from train set and plot them
    indices = torch.randperm(len(train))[:12]
    samples, labels = torch.stack([train[i][0] for i in indices]), torch.tensor([train[i][1] for i in indices])
    fig, axes = plt.subplots(3, 4, figsize=(8, 6))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(samples[i].squeeze(), cmap="gray")
        ax.set_title(f"Label: {labels[i].item()}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("reports/figures/dataset_samples.png")
    plt.close()

    labels = torch.tensor([train[i][1] for i in range(len(train))])
    plt.hist(labels.numpy(), bins=range(11), align="left", rwidth=0.8)
    plt.xticks(range(10))
    plt.xlabel("Class Label")
    plt.ylabel("Frequency")
    plt.title("Class Distribution in Training Set")
    plt.savefig("reports/figures/train_class_distribution.png")
    plt.close()

    test_labels = torch.tensor([test[i][1] for i in range(len(test))])
    plt.hist(test_labels.numpy(), bins=range(11), align="left", rwidth=0.8)
    plt.xticks(range(10))
    plt.xlabel("Class Label")
    plt.ylabel("Frequency")
    plt.title("Class Distribution in Test Set")
    plt.savefig("reports/figures/test_class_distribution.png")
    plt.close()


class MNISTDataset(torch.utils.data.Dataset):
    """Custom Dataset for MNIST data."""

    def __init__(self,train: bool,  dataset_path: str = "./data/processed") -> None:
        if train:
            self.images = torch.load(f"{dataset_path}/train_images.pt")
            self.targets = torch.load(f"{dataset_path}/train_target.pt")
        else:
            self.images = torch.load(f"{dataset_path}/test_images.pt")
            self.targets = torch.load(f"{dataset_path}/test_target.pt")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.targets[idx]



if __name__ == "__main__":
    # typer.run(preprocess_data)
    typer.run(dataset_statistics)
