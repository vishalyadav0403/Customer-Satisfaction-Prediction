# 📊 Customer Satisfaction Prediction

![Banner](banner)

[![Status](https://img.shields.io/badge/Project-Complete-brightgreen)]()
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-orange)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


---

## 📚 Table of Contents
- [📊 Overview](#-overview)
- [📁 Files Included](#-files-included)
- [🚀 Run the Project](#-run-the-project)
- [📈 Model Summary](#-model-summary)
- [🙌 Credits](#-credits)
- [📄 License](#-license)

---

## 📊 Overview

This project predicts **customer satisfaction ratings** from support ticket data using:

- ✍️ Text descriptions of issues (TF-IDF vectorized)
- 🔢 Structured features (age, product, priority, etc.)
- 🧠 Topic modeling (LDA) to group common issues
- 💬 Sentiment analysis with VADER
- 🎯 Random Forest classifier for prediction
- 🖼️ Streamlit app for real-time interaction

---

## 📁 Files Included

| File                                | Description                              |
|-------------------------------------|------------------------------------------|
| `customer_support_tickets.csv`      | Original support tickets dataset         |
| `ticket_topics.csv`                 | Topics assigned using LDA modeling       |
| `final_predictions.csv`             | Dataset with model predictions           |
| `hybrid_model_with_topics.pkl`      | Trained Random Forest model              |
| `tfidf_vectorizer_with_topics.pkl`  | TF-IDF vectorizer for ticket text        |
| `train_model_with_topics.py`        | Training + model export script           |
| `app.py`                            | Streamlit web app                        |
| `Customer_Satisfaction_Summary.ipynb` | Final exploratory notebook             |

---

## 📈 Model Summary

| Component     | Details                        |
| ------------- | ------------------------------ |
| Model         | Random Forest Classifier       |
| Accuracy      | \~XX% (Add once finalized)     |
| Text Features | TF-IDF vectorized ticket text  |
| NLP Add-ons   | Sentiment analysis, LDA topics |

---

## 🚀 Run the Project

### 🧪 1. Train the model

```bash
python train_model_with_topics.py
---

🌐 2. Launch Streamlit app

streamlit run app.py

The app allows you to:
Paste a ticket description
Choose age, priority, product
Get prediction + topic + sentiment live

---

