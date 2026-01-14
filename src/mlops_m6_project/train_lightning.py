import os
from pathlib import Path

import hydra
import torch
import wandb
from dotenv import load_dotenv
from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from mlops_m6_project.data import MNISTDataset
from mlops_m6_project.model_lightning import Classifier

load_dotenv()  # take environment variables from .env file
wandb.login()


# Get the absolute path to configs directory
config_path = str(Path(__file__).parent.parent.parent / "configs")


@logger.catch
@hydra.main(version_base=None, config_path=config_path, config_name="config.yaml")
def train(cfg):
    """
    Train a model on Corrupt MNIST.

    Returns
    -------
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
    train_set, test_set = MNISTDataset(train=True), MNISTDataset(train=False)
    model = Classifier(**model_params, lr=training_params.lr)
    # add rest of your training code here
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=training_params.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(test_set, batch_size=training_params.batch_size, shuffle=False)

    checkpoint_callback = ModelCheckpoint(dirpath="models", monitor="val_loss", save_top_k=1, mode="min")
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3, mode="min", verbose=True)
    trainer = Trainer(
        max_epochs=training_params.epochs,
        limit_train_batches=0.2,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=WandbLogger(project=os.environ.get("WANDB_PROJECT")),
        profiler="simple",
        # precision="16-mixed",
        # accelerator="cpu"
    )

    trainer.fit(model, train_loader, val_loader)

    logger.info("Training complete")
    torch.save(model.state_dict(), "models/model.pth")
    artifact = wandb.Artifact(name="cnn", type="model")
    artifact.add_file("models/model.pth")
    run.log_artifact(artifact)


if __name__ == "__main__":
    # typer.run(train)
    train()
