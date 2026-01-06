import torch
import os
from torch.utils.data import DataLoader

DATA_PATH = './data/corruptmnist'

def corrupt_mnist():
    """Return train and test dataloaders for corrupt MNIST."""
    train_loader = None
    test_loader = None
    all_train_images = torch.empty(0)
    all_train_labels = torch.empty(0,dtype=torch.long)
    test_images = torch.empty(0)
    test_labels = torch.empty(0,dtype=torch.long)
    for files in os.listdir(DATA_PATH):
        if files.endswith('.pt'):
            if 'train' in files:
                if 'images' in files:
                    train_images = torch.load(os.path.join(DATA_PATH, files))
                    all_train_images = torch.cat((all_train_images, train_images), dim=0)
                elif 'target' in files:
                    train_labels = torch.load(os.path.join(DATA_PATH, files))
                    all_train_labels = torch.cat((all_train_labels, train_labels), dim=0)
            else:
                print('test file:', files)
                if 'images' in files:
                    test_images = torch.load(os.path.join(DATA_PATH, files))
                elif 'target' in files:
                    test_labels = torch.load(os.path.join(DATA_PATH, files))
    
    # add channel dimension
    all_train_images = all_train_images.unsqueeze(1)
    test_images = test_images.unsqueeze(1)
    test_loader = DataLoader(list(zip(test_images, test_labels)), batch_size=16, shuffle=True)
    train_loader = DataLoader(list(zip(all_train_images, all_train_labels)), batch_size=64, shuffle=True)
    return  train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = corrupt_mnist()
    for images, labels in train_loader:
        print('Train batch - images shape:', images.shape)
        print('Train batch - labels shape:', labels.shape)
        break
    for images, labels in test_loader:
        print('Test batch - images shape:', images.shape)
        print('Test batch - labels shape:', labels.shape)
        break