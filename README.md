# project-01_fake-news-detection
PBEL BATCH 7 AIML project on FAKE NEWS DETECTION MODEL
# 📰 Fake News Detection using Machine Learning

## 📌 Project Overview

This project is a Machine Learning-based Fake News Detection System that classifies news articles as **Real** or **Fake** using Natural Language Processing (NLP) techniques.

The system preprocesses news articles, converts text into numerical features using **TF-IDF Vectorization**, and predicts the news category using a **Logistic Regression** model.

A user-friendly **Streamlit Web Application** is also developed for real-time prediction.

---

## 🚀 Features

- Detects Fake and Real News
- Natural Language Processing (NLP)
- Text Cleaning & Preprocessing
- Stopword Removal
- Porter Stemming
- TF-IDF Feature Extraction
- Logistic Regression Classifier
- Streamlit Web Application
- Confidence Score Display
- Easy-to-use Interface

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Streamlit
- Jupyter Notebook
- VS Code
- Git & GitHub

---

## 📂 Dataset

The dataset contains the following columns:

| Column | Description |
|---------|-------------|
| Title | News headline |
| Text | Full news article |
| Label | 0 = Real News, 1 = Fake News |

Missing values are handled using:

```python
fillna('')
```

---

## ⚙️ Machine Learning Workflow

1. Load Dataset
2. Handle Missing Values
3. Combine Title + Text
4. Text Cleaning
5. Convert to Lowercase
6. Remove Stopwords
7. Apply Porter Stemming
8. TF-IDF Vectorization
9. Train-Test Split (80:20)
10. Train Logistic Regression Model
11. Evaluate Model
12. Deploy using Streamlit

---

## 📊 Model Performance

**Algorithm:** Logistic Regression

**Accuracy:** **94.49%**

The model provides reliable and fast prediction of fake news articles.

---

## 💻 How to Run the Project

### Clone the Repository

```bash
git clone https://github.com/yourusername/Fake-News-Detection.git
cd Fake-News-Detection
```

### Install Required Packages

```bash
pip install -r requirements.txt
```

### Run Streamlit Application

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
Fake-News-Detection/
│
├── app.py
├── data.ipynb
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── README.md
└── dataset.csv
```

---

## 🎯 Future Scope

- Deep Learning Models
- BERT & RoBERTa
- Live News APIs
- Mobile Application
- Cloud Deployment
- Multilingual Fake News Detection

---

## 👨‍💻 Developed By

**MD WALID KHAN**

B.Tech Computer Science & Engineering

IBM PBEL Batch 7

---

## 📜 License

This project is developed for educational and academic purposes as part of the IBM PBEL Program.
