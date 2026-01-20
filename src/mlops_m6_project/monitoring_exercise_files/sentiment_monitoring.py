import json
from pathlib import Path
import os
import anyio
import nltk
import pandas as pd
from evidently.legacy.metric_preset import TargetDriftPreset, TextEvals
from evidently.legacy.report import Report
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from google.cloud import storage

nltk.download("words")
nltk.download("wordnet")
nltk.download("omw-1.4")
PROJECT_NAME = "mlops-483519"
BUCKET_NAME = "gcp_monitoring_exercise_1219"

def run_analysis(reference_data: pd.DataFrame, current_data: pd.DataFrame):
    """Run the analysis and return the report."""
    text_overview_report = Report(metrics=[TextEvals(column_name="content"), TargetDriftPreset(columns=["sentiment"])])
    text_overview_report.run(reference_data=reference_data, current_data=current_data)
    text_overview_report.save_html("text_overview_report.html")


def lifespan(app: FastAPI):
    """Load the data and class names before the application starts."""
    global training_data, class_names
    training_data = pd.read_csv("reviews.csv")

    def to_sentiment(rating):
        """Convert rating to sentiment class."""
        rating = int(rating)
        if rating <= 2:
            return 0  # Negative
        if rating == 3:
            return 1  # Neutral
        return 2  # Positive

    training_data["sentiment"] = training_data.score.apply(to_sentiment)
    class_names = ["negative", "neutral", "positive"]

    yield

    del training_data, class_names


app = FastAPI(lifespan=lifespan)


def download_files(directory: Path, n: int = 5) -> None:
    """Download the N latest prediction files from the GCP bucket."""

    client = storage.Client(project=PROJECT_NAME)
    bucket = client.get_bucket(BUCKET_NAME)

    blobs = list(bucket.list_blobs(prefix='requests/'))
    blobs.sort(key=lambda x: x.time_created, reverse=True)

    os.makedirs(directory, exist_ok=True)

    for blob in blobs[:n]:
        # Extract just the filename from the blob path (e.g., "requests/file.json" -> "file.json")
        filename = blob.name.split('/')[-1]
        destination_path = directory / filename
        if not destination_path.exists():
            blob.download_to_filename(destination_path)
            print(f"Downloaded {blob.name} to {destination_path}")


def load_latest_files(directory: Path, n: int) -> pd.DataFrame:
    """Fetch latest data from the database."""
    download_files(directory=directory, n=n)
    dfs = []
    for file in sorted(directory.iterdir(), key=os.path.getmtime, reverse=True)[:n]:
        data_dict = json.loads(file.read_text())
        temp_df = pd.DataFrame([data_dict])  # Wrap in list to create a single-row DataFrame
        dfs.append(temp_df)
    df = pd.concat(dfs, ignore_index=True)
    return df


@app.get("/report")
async def get_report(n: int = 5):
    """Generate and return the report."""
    prediction_data = load_latest_files(Path("data/requests"), n=n)
    
    # Rename columns to match the reference data schema
    prediction_data = prediction_data.rename(columns={
        "review": "content",
        "predicted_sentiment": "sentiment"
    })
    
    # Convert sentiment class names to integers if they're strings
    if prediction_data["sentiment"].dtype == "object":
        sentiment_map = {"negative": 0, "neutral": 1, "positive": 2}
        prediction_data["sentiment"] = prediction_data["sentiment"].map(sentiment_map)
    
    run_analysis(training_data, prediction_data)

    async with await anyio.open_file("text_overview_report.html", encoding="utf-8") as f:
        html_content = await f.read()

    return HTMLResponse(content=html_content, status_code=200)