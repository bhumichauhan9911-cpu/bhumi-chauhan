import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

st.set_page_config(
    page_title="Cyber Attack Detection",
    page_icon="🛡️",
    layout="wide"
)

MODEL_PATH = "model/cyber_attack_model.pkl"
INFO_PATH = "model/model_info.pkl"
DATA_PATH = "data/network_traffic.csv"

FEATURES = [
    "duration",
    "packet_count",
    "byte_count",
    "src_bytes",
    "dst_bytes",
    "syn_count",
    "failed_connections",
]

st.title("🛡️ Cyber Attack Detection System")
st.caption("Machine Learning + Streamlit Mini Project")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

model = load_model()
df = load_data()

if model is None:
    st.warning("Model is not trained yet. Run `python train.py` first.")
    st.stop()

info = joblib.load(INFO_PATH) if os.path.exists(INFO_PATH) else {}

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Dashboard", "ℹ️ Project"])

with tab1:
    st.subheader("Enter Network Traffic Details")

    c1, c2, c3 = st.columns(3)
    with c1:
        duration = st.number_input("Duration (seconds)", min_value=0.0, value=2.0, step=0.1)
        packet_count = st.number_input("Packet Count", min_value=1, value=50, step=1)
        byte_count = st.number_input("Byte Count", min_value=0.0, value=25000.0, step=100.0)
    with c2:
        src_bytes = st.number_input("Source Bytes", min_value=0.0, value=15000.0, step=100.0)
        dst_bytes = st.number_input("Destination Bytes", min_value=0.0, value=10000.0, step=100.0)
    with c3:
        syn_count = st.number_input("SYN Count", min_value=0, value=4, step=1)
        failed_connections = st.number_input("Failed Connections", min_value=0, value=0, step=1)

    if st.button("🔍 Detect Attack", type="primary"):
        input_df = pd.DataFrame([{
            "duration": duration,
            "packet_count": packet_count,
            "byte_count": byte_count,
            "src_bytes": src_bytes,
            "dst_bytes": dst_bytes,
            "syn_count": syn_count,
            "failed_connections": failed_connections,
        }])

        prediction = model.predict(input_df[FEATURES])[0]

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df[FEATURES])[0]
            classes = list(model.classes_)
            confidence = float(max(probs)) * 100
        else:
            confidence = 0.0

        if prediction == "Attack":
            st.error(f"🚨 CYBER ATTACK DETECTED — Confidence: {confidence:.2f}%")
            st.write("Recommendation: investigate the traffic source and relevant logs.")
        else:
            st.success(f"✅ NORMAL TRAFFIC — Confidence: {confidence:.2f}%")

with tab2:
    st.subheader("Dataset Dashboard")

    if df is not None:
        a, b, c = st.columns(3)
        a.metric("Total Records", len(df))
        b.metric("Normal", int((df["label"] == "Normal").sum()))
        c.metric("Attacks", int((df["label"] == "Attack").sum()))

        st.write("### Attack Distribution")
        counts = df["label"].value_counts()
        fig, ax = plt.subplots()
        counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("Traffic Type")
        ax.set_ylabel("Number of Records")
        st.pyplot(fig)
        plt.close(fig)

        st.write("### Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

    st.write("### Model Performance")
    if "accuracy" in info:
        st.metric("Test Accuracy", f"{info['accuracy'] * 100:.2f}%")

    if "confusion_matrix" in info:
        cm = info["confusion_matrix"]
        fig, ax = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Normal", "Attack"],
            yticklabels=["Normal", "Attack"],
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)

    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        st.write("### Feature Importance")
        st.bar_chart(importance.set_index("Feature"))

with tab3:
    st.subheader("About the Project")
    st.write("""
    This mini project uses a Random Forest classifier to identify whether
    network traffic is normal or potentially malicious.

    **Workflow:** Dataset → Preprocessing → Train/Test Split → Random Forest →
    Evaluation → Streamlit Prediction Dashboard.
    """)

    st.write("**Features used:**")
    st.write(", ".join(FEATURES))

    st.info(
        "Educational project only. The bundled dataset is synthetic and is "
        "intended for learning Streamlit and machine learning workflows."
    )
