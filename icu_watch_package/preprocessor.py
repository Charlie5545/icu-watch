import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def clean_data(df):
    # Fill missing values
    return df.interpolate().bfill().ffill()

def scale_data(df):
    columns_to_scale = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']
    min_max_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()

    # Apply MinMaxScaler to the O2Sat column
    df['O2Sat'] = min_max_scaler.fit_transform(df[['O2Sat']])

    # Apply StandardScaler to the rest of the columns
    columns_to_standardize = [col for col in columns_to_scale if col != 'O2Sat']
    df[columns_to_standardize] = standard_scaler.fit_transform(df[columns_to_standardize])

    return df

def create_sequences(X, time_steps=6):
    if len(X) < time_steps:
        raise ValueError(f"Input data must have at least {time_steps} time steps.")
    return np.array([X.values])

def preprocess_input(data):
    expected_columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'Age', 'Gender']

    if not all(col in data.columns for col in expected_columns):
        raise ValueError(f"Input data must contain all of these columns: {expected_columns}")

    # Select only the required features
    df = data[expected_columns]

    # Clean data
    df_cleaned = clean_data(df)

    # Scale data
    df_scaled = scale_data(df_cleaned)

    # Create sequences
    X_seq = create_sequences(df_scaled, time_steps=6)

    print('Input data successfully preprocessed')
    return X_seq

if __name__ == "__main__":
    # Test the preprocessing function
    sample_data = pd.read_csv('sample2.csv')
    processed_data = preprocess_input(sample_data)
    print("Preprocessed data shape:", processed_data.shape)
