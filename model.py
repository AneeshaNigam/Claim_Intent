import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
df = pd.read_csv("Insurance claims data.csv")

# Features and label
selected_columns = ['subscription_length', 'vehicle_age', 'customer_age',
                    'region_code', 'fuel_type', 'ncap_rating', 'claim_status']
df = df[selected_columns]

X = df.drop("claim_status", axis=1)
y = df["claim_status"]

# Define categorical columns
categorical_cols = ['region_code', 'fuel_type']

# Define preprocessor
preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

# Build pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and save
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, "model.pkl")
