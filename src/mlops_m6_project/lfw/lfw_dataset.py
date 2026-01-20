"""LFW dataloading."""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


class LFWDataset(Dataset):

    """Initialize LFW dataset."""

    def __init__(self, path_to_folder: str, transform) -> None:
        # TODO: fill out with what you need
        self.transform = transform
        image_folder = path_to_folder + "/lfw-deepfunneled/lfw-deepfunneled"
        self.image_paths = []
        for root, _, files in os.walk(image_folder):
            for file in files:
                if file.endswith(".jpg"):
                    self.image_paths.append(os.path.join(root, file))

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Get item from dataset."""
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-path_to_folder", default="./data/lfw", type=str)
    parser.add_argument("-batch_size", default=128, type=int)
    parser.add_argument("-num_workers", default=1, type=int)
    parser.add_argument("-visualize_batch", action="store_true")
    parser.add_argument("-get_timing", action="store_true")
    parser.add_argument("-batches_to_check", default=100, type=int)

    args = parser.parse_args()

    lfw_trans = transforms.Compose(
        [
            transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
        ]
    )

    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)

    # Define dataloader
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, multiprocessing_context="fork"
    )

    if args.visualize_batch:
        batch = next(iter(dataloader))
        grid = make_grid(batch)
        show(grid)

    if args.get_timing:
        # Test different numbers of workers
        worker_counts = [0, 1, 2, 4, 8]
        mean_times = []
        std_times = []

        for num_workers in worker_counts:
            print(f"Testing with {num_workers} workers...")
            # Create dataloader with specific number of workers
            test_dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=num_workers,
                multiprocessing_context="fork" if num_workers > 0 else None,
            )

            # Run 5 repetitions
            res = []
            for _ in range(5):
                start = time.time()
                for batch_idx, _batch in enumerate(test_dataloader):
                    if batch_idx > args.batches_to_check:
                        break
                end = time.time()
                res.append(end - start)

            res = np.array(res)
            mean_times.append(np.mean(res))
            std_times.append(np.std(res))
            print(f"  Timing: {np.mean(res):.4f}+-{np.std(res):.4f}")

        # Plot results
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            worker_counts, mean_times, yerr=std_times, marker="o", capsize=5, capthick=2, linewidth=2, markersize=8
        )
        plt.xlabel("Number of Workers")
        plt.ylabel("Time (s)")
        plt.title(f"DataLoader Timing vs Number of Workers (batch_size={args.batch_size})")
        plt.grid(True, alpha=0.3)
        plt.show()
