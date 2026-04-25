# 🎫 Support Ticket Classification & Prioritization System
### Future Interns — Machine Learning Track | Task 02

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat-square)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

---

## 📌 About This Project

Real customer support teams receive **hundreds of tickets daily**.  
Without automation, urgent issues get buried — causing delays and poor customer satisfaction.

This project builds an **end-to-end ML system** that:

- 📂 **Reads** raw support ticket text
- 🏷️ **Classifies** tickets into categories (Billing, Technical Issue, Account, General Query)
- 🚦 **Assigns priority** levels (High / Medium / Low)
- 📊 **Evaluates** multiple ML models and selects the best one

---

## 🗂️ Project Structure

```
FUTURE_ML_02/
│
├── support_ticket_classification.py   ← Main ML script (run this)
├── requirements.txt                   ← Python dependencies
├── README.md                          ← You are here
│
├── models/                            ← Saved trained models
│   ├── category_model.pkl
│   ├── priority_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── label_encoder_category.pkl
│   └── label_encoder_priority.pkl
│
└── plots/                             ← Generated visualisations
    ├── 01_distribution.png
    ├── 02_heatmap.png
    ├── 03_model_comparison.png
    └── 04_confusion_matrices.png
```

---

## ⚙️ How to Run

### Step 1 — Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/FUTURE_ML_02.git
cd FUTURE_ML_02
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — (Optional) Download dataset from Kaggle
Download **Customer Support Ticket Dataset** from:  
🔗 https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset

Place the CSV file in the project folder and rename it:
```
customer_support_tickets.csv
```
> ℹ️ If you skip this step, the script auto-generates a synthetic dataset for demo purposes.

### Step 4 — Run the project
```bash
python support_ticket_classification.py
```

---

## 🧠 ML Pipeline

```
Raw Text
   ↓
Text Cleaning (lowercase → remove stopwords → lemmatization)
   ↓
TF-IDF Vectorization (5000 features, bigrams)
   ↓
┌──────────────────────┬─────────────────────┐
│  Category Classifier │  Priority Predictor  │
│  (4 models compared) │  (4 models compared) │
└──────────────────────┴─────────────────────┘
   ↓
Best Model Selected → Saved as .pkl
   ↓
Live Prediction on New Tickets
```

---

## 🤖 Models Compared

| Model | Task |
|---|---|
| Logistic Regression | Category + Priority |
| Random Forest | Category + Priority |
| LinearSVC | Category + Priority |
| Naive Bayes | Category + Priority |

---

## 📊 Key Results

| Metric | Value |
|---|---|
| Best Category Model | Logistic Regression / LinearSVC |
| Category Accuracy | ~90%+ |
| Best Priority Model | Logistic Regression |
| Priority Accuracy | ~85%+ |

---

## 🔑 Key Features Implemented

- ✅ Text cleaning (lowercasing, stopword removal, lemmatization)
- ✅ TF-IDF feature extraction with bigrams
- ✅ Multi-class ticket category classification
- ✅ Priority level prediction (High / Medium / Low)
- ✅ Comparison of 4 ML models
- ✅ Confusion matrices & class-wise reports
- ✅ Model saved with `joblib` for reuse
- ✅ Live prediction function

---

## 🖼️ Sample Output

```
🎫 Ticket   : My account was charged twice and I need a refund ASAP!
   Category : 💳 Billing
   Priority : 🔴 High

🎫 Ticket   : The app crashes every time I try to open reports.
   Category : 🔧 Technical Issue
   Priority : 🔴 High

🎫 Ticket   : What features are included in the pro plan?
   Category : ❓ General Query
   Priority : 🟢 Low
```

---

## 👨‍💻 Author
      SURYA PRAKASH
**Future Interns ML Intern**  
Task 02 — Support Ticket Classification  
Track: Machine Learning (ML)

---

> ⭐ If you found this useful, give it a star on GitHub!
