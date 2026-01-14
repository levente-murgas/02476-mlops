import matplotlib.pyplot as plt
import torch
import typer
from sklearn.manifold import TSNE
from tqdm import tqdm

from mlops_m6_project.data import MNISTDataset
from mlops_m6_project.model import Classifier


def visualize_embeddings(model_checkpoint: str = "models/model.pth") -> None:
    """Visualize embeddings of a trained model using t-SNE."""
    print("Visualizing embeddings")
    print(model_checkpoint)

    state_dict = torch.load(model_checkpoint)
    model = Classifier()
    model.load_state_dict(state_dict)
    model.eval()
    train_set = MNISTDataset(train=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=False)

    all_embeddings = []
    all_labels = []

    for images, labels in tqdm(train_loader, total=len(train_loader), desc="Extracting embeddings"):
        with torch.no_grad():
            embeddings = model.extract_representations(images)
            all_embeddings.append(embeddings)
            all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings).numpy()
    all_labels = torch.cat(all_labels).numpy()

    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=all_labels, cmap="tab10", alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title("t-SNE visualization of embeddings")
    plt.savefig("reports/figures/embeddings_tsne.png")
    plt.show()


if __name__ == "__main__":
    typer.run(visualize_embeddings)
