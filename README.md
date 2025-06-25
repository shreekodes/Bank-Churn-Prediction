# Bank Churn Prediction with XGBoost & Streamlit 

Predicting whether a customer will leave a bank before they even do.  
This project showcases a **production-ready ML pipeline** built to solve a real-world business problem using **XGBoost**, **SMOTE**, and **Streamlit**.

---

## 🔥 Demo  
 Try it live: [[Streamlit App Link] (https://banking-ch.streamlit.app/)] 

---

## 📊 Problem Statement  
Banks lose millions every year due to customer churn.  
This project builds a **classification model** that predicts the probability of a customer leaving, enabling proactive retention strategies.

---

## Features & Highlights

- Cleaned & preprocessed customer data  
- Encoded categorical features using `LabelEncoder` and `OneHotEncoder`  
- Addressed **class imbalance** with **SMOTE**  
- Built and tuned **XGBoost, Random Forest, and Logistic Regression** models  
- Used **RandomizedSearchCV** for hyperparameter optimization  
- Achieved high **ROC AUC Score & Recall**  
- Fully deployed **interactive Streamlit web app** for real-time prediction

---

## Tech Stack

**Languages & Tools**  
`Python`, `pandas`, `NumPy`, `scikit-learn`, `XGBoost`, `imbalanced-learn`, `Streamlit`, `matplotlib`, `seaborn`, `Git`, `Kaggle`

---

## ⚙ Model Performance (Post-SMOTE)

| Metric         | Value |
|----------------|-------|
| Accuracy       | 89%   |
| Recall (Churn) | 70%+  |
| ROC AUC        | 0.91  |
| Algorithm      | XGBoost (Tuned) |

---

## 📁 Project Structure
📦 Bank Churn Prediction
│
├── app.py # Streamlit app for deployment
├── churn_model.pkl # Trained XGBoost model
├── final_encoded_bank_churn.csv # Processed dataset used for training
├── requirements.txt # All necessary libraries
├── churn_notebook.ipynb# Full EDA, training & evaluation
└── README.md # You're here!


---

## 🚀 How to Run Locally

```bash
git clone https://github.com/yourusername/bank-churn-prediction
cd bank-churn-prediction
pip install -r requirements.txt
streamlit run app.py

---

## 📢 Want to Collaborate or Hire?

I'm actively seeking **AI/ML internship and research opportunities**.  
If you're working on something interesting — let’s connect!

📬 [LinkedIn](https://www.linkedin.com/in/shreeya-srivastava-b19437307/) • [Email](mailto:shreeyasrivastava4@gmail.com)

