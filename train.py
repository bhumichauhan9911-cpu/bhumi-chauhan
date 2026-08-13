import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = "data/network_traffic.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "cyber_attack_model.pkl")
INFO_PATH = os.path.join(MODEL_DIR, "model_info.pkl")

FEATURES = [
    "duration",
    "packet_count",
    "byte_count",
    "src_bytes",
    "dst_bytes",
    "syn_count",
    "failed_connections",
]

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    missing = [c for c in FEATURES + ["label"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[FEATURES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred, labels=["Normal", "Attack"])
    report = classification_report(y_test, pred, output_dict=True)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump({
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classification_report": report,
        "features": FEATURES,
    }, INFO_PATH)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, pred))
    print("\nConfusion Matrix [Normal, Attack]:")
    print(cm)

if __name__ == "__main__":
    main()
