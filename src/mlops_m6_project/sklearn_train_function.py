from datetime import datetime
import os
import pickle
import pandas as pd
from typing import Generator
from loguru import logger
import fastapi
from google.cloud import storage
from pydantic import BaseModel
from sklearn import datasets
from mlops_m6_project.data_drift import standardize_frames, generate_report
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST, make_asgi_app


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

BUCKET_NAME = "iris_models"
MODEL_FILE = "model.pkl"
c = Counter('my_failures', 'Description of counter')


def lifespan(app: fastapi.FastAPI) -> Generator[None]:
    """Load model and classes."""
    global model, classes
    classes = ["Iris-Setosa", "Iris-Versicolour", "Iris-Virginica"]
    client = storage.Client(project='mlops-483519')
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.get_blob(MODEL_FILE)
    model = pickle.loads(blob.download_as_string())

    if not os.path.exists("./reports/iris_predictions_log.csv"):
        os.makedirs("./reports")
        with open("./reports/iris_predictions_log.csv", "a") as f:
            f.write("timestamp,sepal_length,sepal_width,petal_length,petal_width,prediction\n")

    yield

    del model, classes

def save_prediction(timestamp: str, features: IrisFeatures, prediction: int) -> None:
    """Save prediction to a csv file."""
    with open("./reports/iris_predictions_log.csv", "a") as f:
        f.write(f"{timestamp},{features.sepal_length},{features.sepal_width},{features.petal_length},{features.petal_width},{prediction}\n")


app = fastapi.FastAPI(lifespan=lifespan)
app.mount("/metrics", make_asgi_app())

@app.post("/iris_v1/")
@c.count_exceptions()
def knn_classifier(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float, bg_task: fastapi.BackgroundTasks) -> dict:
    """Simple knn classifier function for iris prediction."""
    timestamp = datetime.utcnow().isoformat()
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data).tolist()[0]
    bg_task.add_task(save_prediction, timestamp, IrisFeatures(sepal_length=sepal_length, sepal_width=sepal_width, petal_length=petal_length, petal_width=petal_width), prediction)
    return {"prediction": classes[prediction], 'prediction_int': prediction}

@app.get("/monitoring")
def monitoring() -> dict:
    """Simple health check endpoint."""
    reference_data = datasets.load_iris(as_frame=True).frame
    current_data = pd.read_csv('./reports/iris_predictions_log.csv')
    reference_data_std, current_data_std = standardize_frames(reference_data, current_data)
    generate_report(reference_data_std, current_data_std)
    # serve data_drift.html
    html = open('./reports/data_drift.html', 'r').read()
    return fastapi.responses.HTMLResponse(content=html, status_code=200)

