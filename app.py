import streamlit as st
import pickle

# Page config
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# Title
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>
    🎬 Movie Review Sentiment Analyzer
    </h1>
""", unsafe_allow_html=True)

st.write("Analyze whether a movie review is Positive or Negative")

# Load model safely
try:
    model = pickle.load(open('model/sentiment_model.pkl', 'rb'))
    vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")

# Sidebar info
st.sidebar.title("📊 Model Info")
st.sidebar.write("Algorithm: Logistic Regression")
st.sidebar.write("Vectorizer: TF-IDF")
st.sidebar.write("Dataset: IMDb Reviews")

# Example buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("👍 Try Positive Example"):
        review = "This movie was absolutely amazing with great acting!"

with col2:
    if st.button("👎 Try Negative Example"):
        review = "Worst movie ever, boring and waste of time"

# Input box
review = st.text_area("✍️ Enter your review:")

# Word count
if review:
    st.write(f"📝 Word Count: {len(review.split())}")

# Prediction
if st.button("🔍 Predict"):
    if review:
        review_vec = vectorizer.transform([review])
        prediction = model.predict(review_vec)
        probability = model.predict_proba(review_vec)

        confidence = max(probability[0]) * 100

        if prediction[0] == 'positive':
            st.success(f"😊 Positive Review ({confidence:.2f}% confidence)")
        else:
            st.error(f"😡 Negative Review ({confidence:.2f}% confidence)")
    else:
        st.warning("⚠️ Please enter a review first!")