import streamlit as st
import pickle

model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def check_spam(email_text):
    email_vec = vectorizer.transform([email_text])
    result = model.predict(email_vec)
    return "🚨 SPAM" if result[0] == 1 else "✅ NOT SPAM"

st.title("📧 Spam Detection App")

user_input = st.text_area("Enter message:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        st.success(check_spam(user_input))