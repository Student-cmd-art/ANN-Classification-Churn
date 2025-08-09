# ANN-Classification-Churn

This guide documents the setup instructions used to get TensorFlow working on an Apple Silicon Mac (M1/M2/M3). It also includes the full workflow for running the ANN churn prediction project from start to finish, along with instructions for running it as a Streamlit web application.

## Environment Overview

* **Apple Silicon Mac (ARM64):** MacBook with M1/M2 chip using ARM-based architecture, not Intel x86.
* **Python 3.11 (ARM, via Homebrew):** Version of Python compatible with TensorFlow and optimized for Apple chips.
* **TensorFlow (Apple Silicon version):** Machine learning library where the Apple-optimized version (tensorflow-macos) is required.

## Setup Instructions (First Time)

1. **Install Python 3.11 using Homebrew**

   ```bash
   brew install python@3.11
   ```

2. **Create and activate the virtual environment**

   ```bash
   cd ANN\ PROJECT
   /opt/homebrew/bin/python3.11 -m venv venv
   source venv/bin/activate
   ```

3. **Install TensorFlow for Mac**

   ```bash
   pip install tensorflow-macos
   ```
   * tensorflow-macos: Apple Silicon version of TensorFlow

5. **Install Jupyter and other dependencies**

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn ipykernel
   ```


## Everyday Workflow

To start working on the project:

```bash
cd ANN\ PROJECT
source venv/bin/activate
jupyter notebook
```

---

## ANN Project Workflow

This project predicts bank customer churn using an Artificial Neural Network (ANN) trained on the **Churn\_Modelling.csv** dataset.

### 1. Preprocessing

* Drop irrelevant identifiers: RowNumber, CustomerId, Surname.
* Encode Gender using Label Encoding.
* One-hot encode Geography to avoid ordinal bias.
* Save encoders (LabelEncoder and OneHotEncoder) for later use.
* Standardize numerical features with StandardScaler and save it.

### 2. Model Architecture

* Input layer: size = number of features after preprocessing.
* Dense Layer 1: 64 neurons, ReLU activation, L2 regularization, Dropout (0.3).
* Dense Layer 2: 32 neurons, ReLU activation, Dropout (0.3).
* Output Layer: 1 neuron, Sigmoid activation (binary classification).

### 3. Training

* Loss: Binary Crossentropy.
* Optimizer: Adam (learning rate 0.0001).
* Early Stopping (patience = 5) to prevent overfitting.
* TensorBoard for training visualization.

### 4. Evaluation

* Confusion Matrix (TN, FP, FN, TP).
* Metrics: Precision, Recall, F1-score, ROC-AUC.
* Threshold tuning to optimize F1-score.
* At best threshold (0.37 in current run), F1 improved from 0.574 to 0.634.

### 5. Single-Customer Prediction

* Input details are preprocessed using the saved encoders and scaler.
* Model outputs churn probability.
* Compare probability to threshold to output "Likely to Churn" or "Likely to Not Churn" along with confidence.


## Streamlit Web Application

A Streamlit app is included to make predictions through a web interface.

### Setup and Run

1. Install Streamlit

   ```bash
   pip install streamlit
   ```

2. Ensure the following files are in the same folder:

   ```
   model.h5
   label_encoder_gender.pkl
   one_hot_encoder_geo.pkl
   scaler.pkl
   streamlitappann.py
   ```

3. Run the app

   ```bash
   streamlit run streamlitappann.py
   ```

4. The app will open in your browser at:

   ```
   http://localhost:8501
   ```
