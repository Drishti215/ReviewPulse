import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

# ---- App Header ----
st.markdown("<h1 style='text-align:center;'>📊 ReviewPulse</h1>", unsafe_allow_html=True)
st.write("Analyze customer reviews instantly with ML-powered sentiment analysis.")

# Load pre-trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Keep history in session state
if "history" not in st.session_state:
    st.session_state["history"] = []

# ---- Single Review Input ----
user_input = st.text_area("Enter your review here:")

if st.button("Analyze"):
    if user_input.strip() != "":
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        st.success(f"Sentiment: {sentiment}")

        # Save to history
        st.session_state["history"].append((user_input, sentiment))
    else:
        st.warning("Please enter a review first.")

# ---- File Upload for Bulk Analysis ----
uploaded_file = st.file_uploader("Upload a CSV file with reviews", type=["csv", "txt"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if "review" in df.columns:
        reviews = df["review"].tolist()
        sentiments = model.predict(vectorizer.transform(reviews))

        # Chart FIRST
        st.subheader("Sentiment Distribution")
        pos_count = sum(sentiments)
        neg_count = len(sentiments) - pos_count

        chart_data = pd.DataFrame({
            "Sentiment": ["Positive", "Negative"],
            "Count": [pos_count, neg_count]
        })

        fig = px.pie(chart_data, names="Sentiment", values="Count", title="Sentiment Split")
        st.plotly_chart(fig)

        # Then show results
        st.subheader("Bulk Analysis Results")
        results_df = pd.DataFrame({
            "Review": reviews,
            "Sentiment": ["Positive" if s == 1 else "Negative" for s in sentiments]
        })

        st.dataframe(results_df)

        # Download button
        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="sentiment_results.csv",
            mime="text/csv"
        )
    else:
        st.error("CSV must have a 'review' column.")

# ---- History Section ----
if st.session_state["history"]:
    st.subheader("Analysis History")
    for review, sentiment in st.session_state["history"]:
        st.write(f"Review: {review}")
        st.write(f"Result: {sentiment}")
        st.write("---")

    # Clear history button
    if st.button("Clear History"):
        st.session_state["history"] = []
        st.info("History cleared!")
