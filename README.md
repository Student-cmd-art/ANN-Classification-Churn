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
----

Beyond the architecture described above, the model can also be optimized using hyperparameter tuning techniques.
Examples include:
a. Adjusting the number of neurons in each layer.
b. Changing the number of hidden layers in the network.
c. Tuning the learning rate, batch size, and dropout rate.
d. Using libraries like GridSearchCV (via KerasClassifier wrapper) to systematically search for the best combination of hyperparameters.

## Applications of ANNs

ANNs are inspired by the structure of the human brain and are designed to recognize complex patterns in data. They are  powerful when dealing with **non-linear relationships**, **high-dimensional data**, and **classification or prediction problems**. In the context of churn prediction, ANNs learn subtle behavioural signals from customer data (e.g., usage history, demographics, engagement metrics) to identify patterns that indicate whether a customer is likely to leave.

**Key strengths of ANNs:**
- Learn feature interactions that traditional models may miss  
- Handle noisy, messy, and large datasets   
- Provide probabilistic outputs (confidence scores / churn probabilities)  

## Use-Cases of ANNs in the Insurance Domain

Common applications can include:

- **Customer Churn Prediction:**  
  Identify policyholders likely to switch providers so targeted retention campaigns (e.g premium discounts, call-backs) can be launched.

- **Fraud Detection:**  
  Analyse claim patterns in real-time to detect suspicious/abnormal behaviour for faster flagging of fraudulent claims.

- **Risk Scoring & Underwriting:**  
  Predict claim likelihood by learning from historical policy and claims data (more accurate premium pricing & risk assessment).


Streamlist App: https://ann-classification-churn-9phn6hzcpozoqtbq9fxhif.streamlit.app/
