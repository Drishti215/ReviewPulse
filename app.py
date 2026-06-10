import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- App Header ----
st.markdown("<h1 style='text-align:center;'>ReviewPulse</h1>", unsafe_allow_html=True)
st.write("Analyze customer reviews instantly with ML-powered sentiment analysis.")

# ---- Tabs ----
tab1, tab2 = st.tabs(["ReviewPulse", "Recommendation System"])

# ---- ReviewPulse Tab ----
with tab1:
    st.sidebar.header("Options")
    model_choice = st.sidebar.selectbox("Choose Sentiment Model", ["Logistic Regression", "Naive Bayes"])
    show_history = st.sidebar.checkbox("Show Analysis History", value=True)

    if model_choice == "Logistic Regression":
        model = joblib.load("logistic_model.pkl")
    elif model_choice == "Naive Bayes":
        model = joblib.load("naive_model.pkl")

    vectorizer = joblib.load("vectorizer.pkl")

    if "history" not in st.session_state:
        st.session_state["history"] = []

    user_input = st.text_area("Enter your review here:")

    if st.button("Analyze"):
        if user_input.strip() != "":
            input_tfidf = vectorizer.transform([user_input])
            prediction = model.predict(input_tfidf)[0]
            sentiment = "Positive" if prediction == 1 else "Negative"
            st.success(f"Sentiment: {sentiment}")

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_tfidf)[0]
                confidence = prob.max() * 100
                st.write(f"Confidence: {confidence:.2f}%")

            st.session_state["history"].append((user_input, sentiment))
        else:
            st.warning("Please enter a review first.")

    uploaded_file = st.file_uploader("Upload a CSV file with product reviews", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=["date"])
        if {"review", "product", "date"}.issubset(df.columns):
            reviews = df["review"].tolist()
            sentiments = model.predict(vectorizer.transform(reviews))

            st.subheader("Sentiment Distribution")
            pos_count = sum(sentiments)
            neg_count = len(sentiments) - pos_count

            chart_data = pd.DataFrame({
                "Sentiment": ["Positive", "Negative"],
                "Count": [pos_count, neg_count]
            })

            fig = px.pie(chart_data, names="Sentiment", values="Count", title="Sentiment Split")
            st.plotly_chart(fig)

            st.markdown("---")

            st.subheader("Bulk Analysis Results")
            results_df = pd.DataFrame({
                "product": df["product"],
                "review": reviews,
                "date": df["date"],
                "Sentiment": ["Positive" if s == 1 else "Negative" for s in sentiments]
            })

            st.dataframe(results_df, height=300)

            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="sentiment_results.csv",
                mime="text/csv"
            )

            # Bar chart of positive reviews per product
            pos_counts = results_df[results_df["Sentiment"] == "Positive"].groupby("product").size().reset_index(name="Positive Reviews")
            fig_bar = px.bar(pos_counts, x="product", y="Positive Reviews", title="Positive Reviews per Product")
            st.plotly_chart(fig_bar)

            # Trend analysis over time
            trend_df = results_df.groupby(["date", "Sentiment"]).size().reset_index(name="Count")
            fig_trend = px.line(trend_df, x="date", y="Count", color="Sentiment", title="Sentiment Trend Over Time")
            st.plotly_chart(fig_trend)

            st.session_state["sentiment_results"] = results_df

        else:
            st.error("CSV must have 'product', 'review', and 'date' columns.")

    if show_history and st.session_state["history"]:
        st.subheader("Analysis History")
        for review, sentiment in st.session_state["history"]:
            st.write(f"Review: {review}")
            st.write(f"Result: {sentiment}")
            st.write("---")

        if st.button("Clear History"):
            st.session_state["history"] = []
            st.info("History cleared!")

# ---- Recommendation System Tab ----
with tab2:
    st.subheader("Product Recommendation System")

    data = {
        "Product": ["Laptop", "Phone", "Headphones", "Smartwatch", "Tablet", "TV", "Camera", "Speakers"],
        "Category": ["Electronics", "Electronics", "Accessories", "Electronics", "Electronics", "Electronics", "Electronics", "Accessories"],
        "Description": [
            "Powerful laptop for work and gaming",
            "Smartphone with great camera and apps",
            "Wireless headphones for music lovers",
            "Smartwatch with health tracking features",
            "Tablet for reading and entertainment",
            "High-definition television with clear sound",
            "Digital camera with sharp photo quality",
            "Speakers with loud and clear sound"
        ],
        "Price": [60000, 30000, 2000, 15000, 25000, 40000, 35000, 5000]
    }
    products_df = pd.DataFrame(data)

    st.write("Available Products:")
    st.dataframe(products_df, height=300)

    st.markdown("---")

    st.subheader("Find Products by Category or Price")
    category_choice = st.selectbox("Select Category", products_df["Category"].unique())
    max_price = st.slider("Select Maximum Price", min_value=1000, max_value=70000, value=30000)

    recommended = products_df[(products_df["Category"] == category_choice) & (products_df["Price"] <= max_price)]
    st.write("Recommended Products:")
    st.dataframe(recommended, height=300)

    st.markdown("---")

    st.subheader("Get Smart Recommendations")
    user_pref = st.text_input("Describe what you want (e.g., 'music', 'gaming', 'health')")

    if user_pref:
        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(products_df["Description"])
        user_vec = tfidf.transform([user_pref])
        similarity = cosine_similarity(user_vec, tfidf_matrix).flatten()

        top_indices = similarity.argsort()[-3:][::-1]
        smart_recommendations = products_df.iloc[top_indices].copy()

        if "sentiment_results" in st.session_state:
            sentiment_df = st.session_state["sentiment_results"]
            product_pos_counts = sentiment_df[sentiment_df["Sentiment"] == "Positive"].groupby("product").size()
            smart_recommendations["Positive_Review_Count"] = smart_recommendations["Product"].map(product_pos_counts).fillna(0)
            smart_recommendations = smart_recommendations.sort_values(by="Positive_Review_Count", ascending=False)

        st.markdown("### Final Ranked Recommendations")
        st.dataframe(smart_recommendations[["Product", "Category", "Description", "Price", "Positive_Review_Count"]], height=300)

        rec_csv = smart_recommendations.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Recommendations as CSV",
            data=rec_csv,
            file_name="recommendations.csv",
            mime="text/csv"
        )
