import os
from statistics import mean, stdev
import time

import torch
import hydra
import wandb
import onnx
import onnxruntime as ort

from mlops_m6_project.model_lightning import Classifier

@hydra.main(version_base=None, config_path='../../configs', config_name="eval.yaml")
def main(cfg):
    print(cfg)
    if not os.path.exists("./models/cnn.onnx"):
        export(cfg)
        check_model()

    input_sizes = range(1, 257, 32)
    for input_size in input_sizes:
        print(f"Input size: {input_size}")
        input_tensor = torch.randn(input_size, 1, 28, 28)
        print("PyTorch Inference:")
        output = test_pytorch(cfg, input_tensor)
        print("ONNX Inference:")
        output = test_onnx(input_tensor)


def export(cfg):

    api = wandb.Api()
    artifact_name = f"{cfg.artifact_name}"
    artifact = api.artifact(name=artifact_name)
    artifact.download(cfg.artifact_dir)
    model = Classifier()
    model.load_state_dict(torch.load(f"{cfg.artifact_dir}/model.pth"))
    model.eval()

    dummy_input = torch.randn(64, 1, 28, 28)
    model.to_onnx(
        file_path="./models/cnn.onnx",
        input_sample=dummy_input,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )

def check_model():
    model = onnx.load("./models/cnn.onnx")
    onnx.checker.check_model(model)
    print(onnx.printer.to_text(model.graph))


def timing_decorator(func, function_repeat: int = 10, timing_repeat: int = 5):
    """Decorator that times the execution of a function."""
    def wrapper(*args, **kwargs):
        timing_results = []
        for _ in range(timing_repeat):
            start_time = time.time()
            for _ in range(function_repeat):
                result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            timing_results.append(elapsed_time)
        print(f"Avg +- Stddev: {mean(timing_results):0.3f} +- {stdev(timing_results):0.3f} seconds")
        return result
    return wrapper


@timing_decorator
def inference_onnx(inputs,ort_session: ort.InferenceSession, output_names=None, ):
    return ort_session.run(output_names, inputs)

def test_onnx(input_tensor: torch.Tensor):
    ort_session = ort.InferenceSession("./models/cnn.onnx")
    output_names = [i.name for i in ort_session.get_outputs()]
    inputs = {ort_session.get_inputs()[0].name: input_tensor.numpy()}
    return inference_onnx(inputs, ort_session, output_names)

@timing_decorator
def inference_pytorch(input_tensor: torch.Tensor, model: Classifier):
    with torch.no_grad():
        return model(input_tensor)

def test_pytorch(cfg, input_tensor: torch.Tensor):
    api = wandb.Api()
    artifact_name = f"{cfg.artifact_name}"
    artifact = api.artifact(name=artifact_name)
    artifact.download(cfg.artifact_dir)
    model = Classifier()
    model.load_state_dict(torch.load(f"{cfg.artifact_dir}/model.pth"))
    model.eval()
    return inference_pytorch(input_tensor, model)


if __name__ == "__main__":
    main()

