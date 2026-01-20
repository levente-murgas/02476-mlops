import copy
import time

import torch
from torch import nn
from torch.nn.utils import prune


class LeNet(nn.Module):

    """LeNet implementation."""

    def __init__(self) -> None:
        super().__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """Forward pass of the network."""
        x = nn.functional.max_pool2d(nn.functional.relu(self.conv1(x)), (2, 2))
        x = nn.functional.max_pool2d(nn.functional.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return self.fc3(x)


def check_prune_level(module: nn.Module):
    sparsity_level = 100 * float(torch.sum(module.weight == 0) / module.weight.numel())
    print(f"Sparsity level of module {sparsity_level}")


if __name__ == "__main__":
    model = LeNet()
    unpruned_model = copy.deepcopy(model)

    # print all the parameters of the model
    for m in model.named_modules():
        print(m)

    paramters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            paramters_to_prune.append((module, "weight"))

    parameters_to_prune = tuple(paramters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.2,
    )

    for layer in paramters_to_prune:
        prune.remove(layer[0], "weight")

    for weights in model.parameters():
        weights = weights.to_sparse()

    for m in [unpruned_model, model]:
        tic = time.time()
        for _ in range(100):
            _ = m(torch.randn(100, 1, 28, 28))
        toc = time.time()
        print(f"{toc - tic:.4f} seconds")

    torch.save(model.state_dict(), "./models/m31_pruned_network.pt")
    torch.save(unpruned_model.state_dict(), "./models/m31_network.pt")
