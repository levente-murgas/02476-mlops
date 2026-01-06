from typing_extensions import Annotated
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import joblib
import typer

app = typer.Typer()
train_app = typer.Typer()
app.add_typer(train_app, name="train")

data = load_breast_cancer()
x = data.data
y = data.target

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

@train_app.command()
def svm(kernel='linear', output: Annotated[str, typer.Option("--output", "-o")]= 'models/model.ckpt'):
    """Train and evaluate the model."""
    # Train a Support Vector Machine (SVM) model
    model = SVC(kernel=kernel, random_state=42)
    model.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Save the trained model
    joblib.dump((model, scaler), output)

    # Print the results
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
    return accuracy, report

@train_app.command()
def knn(n_neighbors: int=5, output: Annotated[str, typer.Option("--output", "-o")]= 'models/model.ckpt'):
    """Train and evaluate the model."""
    # Train a K-Nearest Neighbors (KNN) model
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Save the trained model
    joblib.dump((model, scaler), output)

    # Print the results
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
    return accuracy, report

@app.command()
def evaluate(model_path: str):
    """Evaluate the trained model."""
    # Load the dataset
    model, scaler = joblib.load(model_path)

    # Make predictions on the test set
    y_pred = model.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Print the results
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
    return accuracy, report

# this "if"-block is added to enable the script to be run from the command line
if __name__ == "__main__":
    app()