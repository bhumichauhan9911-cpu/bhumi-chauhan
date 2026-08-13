# Cyber Attack Detection System

A beginner-friendly Machine Learning mini project built with Python and Streamlit.

## Objective
Detect whether a network traffic record is **Normal** or a **Cyber Attack** using machine learning.

## Technologies
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Joblib

## Project Structure
```
Cyber_Attack_Detection_Streamlit/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── data/
│   └── network_traffic.csv
└── model/
    └── (generated after training)
```

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py
```

This creates:
- `model/cyber_attack_model.pkl`
- `model/model_info.pkl`

### 3. Start Streamlit
```bash
streamlit run app.py
```

## Input Features
- Duration
- Packet Count
- Byte Count
- Source Bytes
- Destination Bytes
- SYN Count
- Failed Connections

## Output
The application predicts:
- **Normal Traffic**
- **Cyber Attack**

It also displays model accuracy, confusion matrix, classification report, and feature importance.

## Important Note
The included CSV is a small educational/demo dataset generated for this project. For a real cybersecurity project, replace it with a properly labeled public network-security dataset such as CIC-IDS2017 or UNSW-NB15 and map its columns to the features used by the model.
