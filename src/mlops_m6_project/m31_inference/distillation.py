import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

dataset = load_dataset("cifar10")

extractor = AutoFeatureExtractor.from_pretrained("aaraki/vit-base-patch16-224-in21k-finetuned-cifar10")
model = AutoModelForImageClassification.from_pretrained("aaraki/vit-base-patch16-224-in21k-finetuned-cifar10")

# Process entire training dataset
all_logits = []
for sample in tqdm(dataset["train"], total=len(dataset["train"])):
    preprocessed_img = extractor(sample["img"], return_tensors="pt")
    with torch.no_grad():  # Disable gradient computation for inference
        output = model(**preprocessed_img)
    all_logits.append(output.logits)

# Stack all logits into a single tensor
all_logits = torch.cat(all_logits, dim=0)

# Save the results
torch.save(all_logits, "./data/train_logits.pt")
print(f"Saved logits with shape: {all_logits.shape}")


class MyCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(16 * 8 * 8, 10)  # Assuming input images are 32x32

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 8 * 8)
        x = self.fc1(x)
        return x


# tensor([[ 3.3682, -0.3160, -0.2798, -0.5006, -0.5529, -0.5625, -0.6144, -0.4671, 0.2807, -0.3066]])
