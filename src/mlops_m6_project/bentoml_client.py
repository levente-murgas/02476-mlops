import bentoml
import torch
import numpy as np

def main():
    client = bentoml.SyncHTTPClient('http://localhost:3000')
    for _ in range(5):
        input = torch.randn(1, 1, 28, 28).numpy().astype(np.float32)  # Example input tensor for MNIST model
        outputs = client.predict(input=input)
        print(outputs)
    # Close the client to release resources
    client.close()


if __name__ == "__main__":
    main()
