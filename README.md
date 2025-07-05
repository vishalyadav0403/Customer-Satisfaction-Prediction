# 📊 Customer Satisfaction Prediction

This project predicts customer satisfaction ratings using support ticket data. It uses:
- Text descriptions
- Structured features (age, product, priority)
- Topic modeling
- Sentiment analysis

---

## 🔧 Files Included

| File                           | Description                                |
|--------------------------------|--------------------------------------------|
| `customer_support_tickets.csv` | Raw support tickets dataset                |
| `ticket_topics.csv`            | Topics assigned using LDA                  |
| `final_predictions.csv`        | Model predictions with topics              |
| `hybrid_model_with_topics.pkl` | Trained RandomForest model                 |
| `tfidf_vectorizer_with_topics.pkl` | TF-IDF vectorizer                   |
| `train_model_with_topics.py`   | Full training + export script              |
| `app.py`                       | Streamlit interface                        |
| `Customer_Satisfaction_Summary.ipynb` | Final analysis notebook            |

---

## 🚀 Run the Project

### 1. Train the model
```bash
python train_model_with_topics.py
```

### 2. Run Streamlit App
```bash
streamlit run app.py
```

---

## 📊 Summary

This end-to-end machine learning project showcases:
- NLP (TF-IDF, sentiment)
- Topic modeling
- Classification
- App deployment using Streamlit
