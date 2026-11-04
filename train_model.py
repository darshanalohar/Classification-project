import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


df = pd.read_csv("iris.csv")

print("=" * 60)
print("IRIS FLOWER CLASSIFICATION USING KNN")
print("=" * 60)

print("\nFirst 5 records:")
print(df.head())


print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())


df = df.dropna()

print("\nDataset after removing missing values:")
print(df.shape)


X = df[
    [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]
]

y = df["species"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))



model = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        ),

        (
            "knn",
            KNeighborsClassifier(n_neighbors=5)
        )
    ]
)


print("\nTraining KNN model...")

model.fit(X_train, y_train)

print("Model training completed successfully!")




y_pred = model.predict(X_test)




accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(
    "\nAccuracy:",
    round(accuracy * 100, 2),
    "%"
)




print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)




print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)




joblib.dump(
    model,
    "model.pkl"
)

print("\n" + "=" * 60)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 60)

print("Created file: model.pkl")