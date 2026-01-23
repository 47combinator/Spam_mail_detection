# 📧 Spam Mail Detection – Machine Learning Project

A machine learning web application that classifies messages as **Spam** or **Not Spam** using **TF-IDF vectorization** and **Multinomial Naive Bayes**.  
The project includes a simple **Streamlit frontend** for real-time predictions.

This project was built as a first end-to-end ML application, covering data preprocessing, model training, evaluation, and deployment.

---

## 🚀 Features
- Detects spam messages in real time
- Simple and clean Streamlit web interface
- Pre-trained machine learning model
- Lightweight and fast predictions

---

## 🧠 How It Works
1. User enters a message in the web interface  
2. Text is converted into numerical features using **TF-IDF**
3. A **Multinomial Naive Bayes** classifier predicts whether the message is spam
4. Result is displayed instantly

---

## 🛠 Tech Stack
- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **Streamlit**

---

## ▶️ Run the App Locally

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
2️⃣ Run the Streamlit app
bash
Copy code
streamlit run app.py
The app will open in your browser at:

arduino
Copy code
http://localhost:8501
📁 Project Structure
bash
Copy code
.
├── app.py              # Streamlit frontend
├── spam_model.pkl      # Trained ML model
├── vectorizer.pkl      # TF-IDF vectorizer
├── requirements.txt    # Project dependencies
└── README.md
📚 Learning Outcomes
Through this project, I learned:

Text preprocessing using TF-IDF

Supervised classification with Naive Bayes

Handling train/test splits correctly

Saving and loading trained models

Deploying ML models using Streamlit

Managing projects with Git and GitHub

✨ Author
Pratyush (47combinator)
Machine Learning & AI Enthusiast
