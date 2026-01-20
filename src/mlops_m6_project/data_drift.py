import pandas as pd
from sklearn import datasets
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from evidently.metrics import MissingValueCount

reference_data = datasets.load_iris(as_frame=True).frame
current_data = pd.read_csv('./reports/iris_predictions_log.csv')

def standardize_frames(ref_df: pd.DataFrame, curr_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize the dataframes to have the same columns and types."""
    ref_df = ref_df.rename(columns={
        'sepal length (cm)': 'sepal_length',
        'sepal width (cm)': 'sepal_width',
        'petal length (cm)': 'petal_length',
        'petal width (cm)': 'petal_width'
    })
    ref_df = ref_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    curr_df = curr_df.drop(columns=['timestamp', 'prediction'])
    curr_df = curr_df.astype(ref_df.dtypes.to_dict())
    return ref_df, curr_df


def filter_curret_df(n: int | None=None, t: int | None=None) -> pd.DataFrame:
    """Filter the current dataframe to the last n rows or last t seconds."""
    curr_df = current_data.copy()
    if n is not None:
        curr_df = curr_df.tail(n)
    if t is not None:
        curr_df['timestamp'] = pd.to_datetime(curr_df['timestamp'])
        time_threshold = pd.Timestamp.now() - pd.Timedelta(hours=t)
        curr_df = curr_df[curr_df['timestamp'] >= time_threshold]
    return curr_df

def generate_report(reference_data, current_data) -> None:
    report = Report(metrics=[
        DataSummaryPreset(),
        DataDriftPreset(),
        MissingValueCount(column='sepal_length'),
        MissingValueCount(column='sepal_width'),
        MissingValueCount(column='petal_length'),
        MissingValueCount(column='petal_width'),
    ], include_tests=True)
    snapshot = report.run(reference_data=reference_data, current_data=current_data)
    snapshot.save_html('./reports/data_drift.html')