import os
import pandas as pd
from icu_watch_package.preprocessor import preprocess_input
from icu_watch_package.model_new import load_trained_model

# Load model
#model_path = os.path.join('icu_watch_package','models', 'sepsis_model2.h5')
#print(f"Attempting to load model from: {model_path}")
#model = load_model(model_path)
model = load_trained_model()
print("Model loaded successfully")

# Load and preprocess data
print("Loading data...")
df = pd.read_csv('raw_data/sample2.csv')
print(f"Data shape before preprocessing: {df.shape}")
df = preprocess_input(df)
print(f"Data shape after preprocessing: {df.shape}")

# Make predictions
print("Making predictions...")
print(model.predict)
predictions = model.predict(df, verbose=1, steps=1)
print("Predictions:")
print(predictions)
