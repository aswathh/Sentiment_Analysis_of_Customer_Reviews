import streamlit as st
from src.predict import predict_sentiment

st.title("Sentiment Analysis of Customer Reviews")

review = st.text_area("Enter a customer review")

if st.button("Analyze"):
    if review.strip():
        result = predict_sentiment(review)
        if "Positive" in result:
            st.success(result)
        else:
            st.error(result)
    else:
        st.warning("Please enter a review.")