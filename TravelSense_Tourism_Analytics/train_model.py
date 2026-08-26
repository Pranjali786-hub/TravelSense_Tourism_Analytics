"""Train and save the simple TensorFlow tourist footfall prediction model."""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "dataset" / "tourism_data.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_FILE = MODEL_DIR / "tourism_model.keras"
SCALER_FILE = MODEL_DIR / "scaler.pkl"
ENCODERS_FILE = MODEL_DIR / "encoders.pkl"
FEATURE_COLUMNS = ["Destination", "Season", "Year", "Month", "Domestic_Tourists", "Foreign_Tourists", "Hotel_Occupancy", "Festival_Event"]


def load_data():
    data = pd.read_csv(DATA_FILE)
    data = data.dropna().drop_duplicates().copy()
    return data


def prepare_features(data, encoders=None, scaler=None, fit=False):
    """Encode categories and scale numeric values without external ML libraries."""
    frame = data[FEATURE_COLUMNS].copy()
    if fit:
        encoders = {name: {value: index for index, value in enumerate(sorted(frame[name].unique()))}
                    for name in ("Destination", "Season")}
    for name, mapping in encoders.items():
        frame[name] = frame[name].map(mapping).fillna(0)
    values = frame.astype(float).to_numpy()
    if fit:
        scaler = {"mean": values.mean(axis=0), "std": values.std(axis=0)}
        scaler["std"][scaler["std"] == 0] = 1
    return (values - scaler["mean"]) / scaler["std"], encoders, scaler


def train_model():
    tf.keras.utils.set_random_seed(42)
    data = load_data()
    features, encoders, scaler = prepare_features(data, fit=True)
    target = data["Tourist_Footfall"].astype(float).to_numpy()
    indices = np.random.default_rng(42).permutation(len(features))
    split = int(len(features) * 0.80)
    train_ids, test_ids = indices[:split], indices[split:]
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(features.shape[1],)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(features[train_ids], target[train_ids], validation_data=(features[test_ids], target[test_ids]), epochs=140, batch_size=24, verbose=0)
    loss, mae = model.evaluate(features[test_ids], target[test_ids], verbose=0)
    MODEL_DIR.mkdir(exist_ok=True)
    model.save(MODEL_FILE)
    with SCALER_FILE.open("wb") as file: pickle.dump(scaler, file)
    with ENCODERS_FILE.open("wb") as file: pickle.dump(encoders, file)
    print(f"Model saved. Test MAE: {mae:,.0f} tourists")


if __name__ == "__main__":
    train_model()
