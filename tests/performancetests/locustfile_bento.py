from locust import HttpUser, between, task
import bentoml
import numpy as np

class BentoUser(HttpUser):
    """A simple Locust user class that defines the tasks to be performed by the users."""
    wait_time = between(1, 2)

    @task
    def predict_batch(self) -> None:
            # For batchable endpoint, send with channel dimension (1, 28, 28)
            input = np.random.rand(1,1,28,28).astype(np.float32)
            payload = {
                'input': input.tolist()
            }
            self.client.post('/predict_batch',json=payload, headers={'Content-Type': 'application/json'}
            )

    # @task
    # def predict(self) -> None:
    #         input = np.random.rand(1,1,28,28).astype(np.float32)
    #         payload = {
    #             'input': input.tolist()
    #         }
    #         self.client.post('/predict',json=payload, headers={'Content-Type': 'application/json'}
    #         )

