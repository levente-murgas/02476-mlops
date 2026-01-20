from time import time

import torch
from torchvision.models import (
    EfficientNet_B5_Weights,
    ResNet50_Weights,
    Swin_T_Weights,
    efficientnet_b5,
    resnet50,
    swin_t,
)
from ptflops import get_model_complexity_info


def init_model(model_name: str):
    if model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
    elif model_name == "efficientnet_b5":
        weights = EfficientNet_B5_Weights.DEFAULT
        model = efficientnet_b5(weights=weights)
    elif model_name == "swin_t":
        weights = Swin_T_Weights.DEFAULT
        model = swin_t(weights=weights)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    model.eval()
    return model


def time_inference():
    model_names = ["resnet50", "efficientnet_b5", "swin_t"]
    dummy_input = torch.randn(100, 3, 256, 256)

    for model_name in model_names:
        model = init_model(model_name)
        print(get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=False))

        # Timing
        start_time = time()
        for _ in range(10):
            _ = model(dummy_input)
        end_time = time()

        avg_time = (end_time - start_time) / 10
        print(f"Average inference time for {model_name}: {avg_time:.6f} seconds")


if __name__ == "__main__":
    time_inference()
