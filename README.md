# ReviewPulse

ReviewPulse is a Streamlit app for sentiment analysis and product recommendations.

##  Live Demo
Try it here: [ReviewPulse App](https://reviewpulse-j5xbxceddwdbqh4nqyvjsi.streamlit.app/)

## Features
- Upload CSVs with product reviews
- ML-powered sentiment analysis (Logistic Regression & Naive Bayes)
- Visualizations: pie chart, bar chart, trend line
- Smart product recommendations based on user preferences
- Downloadable results and recommendations

## Files
- `app.py` – main Streamlit app
- `product_reviews.csv` – sample dataset with dates
- `logistic_model.pkl`, `naive_model.pkl`, `vectorizer.pkl` – trained models
- `requirements.txt` – dependencies

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
