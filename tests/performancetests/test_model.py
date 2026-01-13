import os
from dotenv import load_dotenv
import torch
import wandb
from loguru import logger
from mlops_m6_project.model_lightning import Classifier
import time
import typer

def main(path: str = None):
    load_dotenv()
    path = os.getenv("MODEL_NAME") if path is None else path
    api = wandb.Api(
        api_key=os.getenv("WANDB_API_KEY"),
        overrides={"entity": os.getenv("WANDB_ENTITY"), "project": os.getenv("WANDB_PROJECT")},
    )
    artifact = api.artifact(path, type="model")
    artifact_dir = artifact.download()
    file_name = artifact.files()[0].name
    logger.info(f"Model artifact downloaded to: {artifact_dir}")

    ckpt = torch.load(os.path.join(artifact_dir, file_name))
    torch.save(ckpt, os.path.join(artifact_dir, file_name))
    cls = Classifier()
    cls.load_state_dict(ckpt)
    cls.eval()
    logger.info("Model loaded successfully")

    with torch.no_grad():
        sample_input = torch.randn(100, 1, 28, 28)
        start = time.time()
        output = cls(sample_input)
        end = time.time()
        inferece_time = end - start
        assert inferece_time < 1.0  # Example assertion: inference should take less than 1 second
        logger.info(f"Inference time for 100 samples: {inferece_time} seconds")

if __name__ == "__main__":
    typer.run(main)