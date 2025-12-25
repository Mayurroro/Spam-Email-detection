import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


#  NLTK downloads (IMPORTANT)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(ps.stem(i))

    return " ".join(y)

# Streamlit UI
st.set_page_config(page_title="Spam Email Detection", page_icon="📧")

st.title("📧 Spam Email Detection App")
st.subheader("Machine Learning Project")

input_email = st.text_area(
    "Enter the email/message content below:",
    height=200,
    placeholder="Type your email text here..."
)

if st.button("Predict"):
    if input_email.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        processed_text = transform_text(input_email)
        vectorized_text = vectorizer.transform([processed_text])
        prediction = model.predict(vectorized_text)[0]

        if prediction == 1:
            st.error("🚨 This message is **SPAM**")
        else:
            st.success("✅ This message is **NOT SPAM**")

st.markdown("---")
st.markdown(
    "🔍 **Model Used:** Logistic Regression  \n"
    "📊 **Text Processing:** Tokenization, Stopword Removal, Stemming  \n"
    "🛠 **Vectorization:** CountVectorizer "
)
