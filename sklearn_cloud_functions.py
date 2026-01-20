# Load data
import pickle

import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from google.cloud import storage


iris_x, iris_y = datasets.load_iris(return_X_y=True)

# Split iris data in train and test data
# A random permutation, to split the data randomly
np.random.seed(0)
indices = np.random.permutation(len(iris_x))
iris_x_train = iris_x[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_x_test = iris_x[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]

# Create and fit a nearest-neighbor classifier

print(iris_x_train[0])

knn = KNeighborsClassifier()
knn.fit(iris_x_train, iris_y_train)
knn.predict(iris_x_test)

# save model

with open("model.pkl", "wb") as file:
    pickle.dump(knn, file)
    BUCKET_NAME = "iris_models"
    MODEL_FILE = "model.pkl"
    client = storage.Client(project='mlops-483519')
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_FILE)
    blob.upload_from_filename("model.pkl")