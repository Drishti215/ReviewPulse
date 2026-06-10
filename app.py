import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

# ---- App Header ----
st.markdown("<h1 style='text-align:center;'>📊 ReviewPulse</h1>", unsafe_allow_html=True)
st.write("Analyze customer reviews instantly with ML-powered sentiment analysis.")

# ---- Sidebar ----
st.sidebar.header("⚙️ Options")
model_choice = st.sidebar.selectbox("Choose Sentiment Model", ["Logistic Regression", "Naive Bayes"])
show_history = st.sidebar.checkbox("Show Analysis History", value=True)

# ---- Load Models ----
# Make sure you have logistic_model.pkl and naive_model.pkl saved in your repo
if model_choice == "Logistic Regression":
    model = joblib.load("logistic_model.pkl")
elif model_choice == "Naive Bayes":
    model = joblib.load("naive_model.pkl")

# Vectorizer (same for both models)
vectorizer = joblib.load("vectorizer.pkl")

# ---- Session State for History ----
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
if show_history and st.session_state["history"]:
    st.subheader("Analysis History")
    for review, sentiment in st.session_state["history"]:
        st.write(f"Review: {review}")
        st.write(f"Result: {sentiment}")
        st.write("---")

    # Clear history button
    if st.button("Clear History"):
        st.session_state["history"] = []
        st.info("History cleared!")
