import bentoml
import onnxruntime as ort
import numpy as np
import numpy.typing as npt


@bentoml.service(workers="cpu_count")
class MNISTClassifier():
    def __init__(self, onnx_model_path: str="./models/cnn.onnx"):
        self.ort_session = ort.InferenceSession(onnx_model_path)

    @bentoml.api(batchable=True, batch_dim=0, max_batch_size=128, max_latency_ms=100)
    def predict_batch(self, input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        # BentoML batches inputs from (1, 28, 28) to (batch_size, 1, 28, 28)
        inputs = {self.ort_session.get_inputs()[0].name: input}
        output_names = [i.name for i in self.ort_session.get_outputs()]
        outputs = self.ort_session.run(output_names, inputs)
        return np.array(outputs[0])

    @bentoml.api()
    def predict(self, input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        inputs = {self.ort_session.get_inputs()[0].name: input}
        output_names = [i.name for i in self.ort_session.get_outputs()]
        outputs = self.ort_session.run(output_names, inputs)
        return np.array(outputs[0])

@bentoml.service(workers="cpu_count")
class MNISTMapToLabel():
    inference_service = bentoml.depends(MNISTClassifier)
    label_map = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine"
    }

    @bentoml.api(batchable=True, batch_dim=0, max_batch_size=128, max_latency_ms=100)
    async def predict_batch(self, input: npt.NDArray[np.float32]) -> list[str]:
        preds = await self.inference_service.to_async.predict_batch(input)
        predicted_indices = np.argmax(preds, axis=1)
        return [self.label_map[idx] for idx in predicted_indices]

    @bentoml.api()
    async def predict(self, input: npt.NDArray[np.float32]) -> str:
        preds = await self.inference_service.to_async.predict(input)
        predicted_index = np.argmax(preds)
        return self.label_map[predicted_index]