import torch
import torch.nn as nn
from pytorch_lightning import LightningModule


class Classifier(LightningModule):
    def __init__(self, conv1_channels=32, conv2_channels=64, fc1_units=128, fc2_units=10, lr=1e-3):
        super(Classifier, self).__init__()

        self.conv1 = nn.Conv2d(1, conv1_channels, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(conv2_channels * 5 * 5, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.2)
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.criterion = nn.NLLLoss()
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Expected input to a 4D tensor")
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError("Expected each sample to have shape [1, 28, 28]")

        x = self.pool(self.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 64 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.log_softmax(self.fc2(x))
        return x

    def extract_representations(self, x):
        """Forward pass up to the second last fully connected layer to extract representations."""
        x = self.pool(self.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 64 * 5 * 5)
        x = self.relu(self.fc1(x))
        return x

    def training_step(self, batch, batch_idx):
        data, target = batch
        y_pred = self(data)
        loss = self.criterion(y_pred, target)
        self.log("train_loss", loss)
        # make sure target has shape (batch_size,)
        target = target.view(-1)
        acc = (y_pred.argmax(dim=1) == target).float().mean()
        self.log("train_accuracy", acc, prog_bar=True)
        # self.logger.experiment.log({'logits': wandb.Histogram(y_pred.cpu().detach().numpy())})

        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        y_pred = self(data)
        loss = self.criterion(y_pred, target)
        # make sure target has shape (batch_size,)
        target = target.view(-1)
        acc = (y_pred.argmax(dim=1) == target).float().mean()
        self.log("val_accuracy", acc, prog_bar=True, on_epoch=True)
        self.log("val_loss", loss, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    model = Classifier()
    print(model)

    log_ps = model(torch.randn(64, 1, 28, 28))
    ps = torch.exp(log_ps)
    # check if they sum to 1
    print(ps.sum(dim=1))
