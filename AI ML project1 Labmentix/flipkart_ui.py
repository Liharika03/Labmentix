import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
import joblib
import os

# Removed deprecated Streamlit option to avoid error

@st.cache_data
def load_data():
    # Adjust path as needed
    data_path = "Customer_support_data.csv"
    df = pd.read_csv(data_path)
    return df

def plot_csat_distribution(df):
    fig, ax = plt.subplots(figsize=(8,5))
    sns.countplot(x='CSAT Score', data=df, palette='viridis', ax=ax)
    ax.set_title('Distribution of Customer Satisfaction (CSAT) Scores')
    ax.set_xlabel('CSAT Score (1 = Low, 5 = High)')
    ax.set_ylabel('Number of Customers')
    st.pyplot(fig)

def plot_csat_by_channel(df):
    fig, ax = plt.subplots(figsize=(10,6))
    sns.boxplot(x='channel_name', y='CSAT Score', data=df, ax=ax)
    ax.set_title('CSAT Score by Support Channel')
    ax.set_xlabel('Channel')
    ax.set_ylabel('CSAT Score')
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_csat_by_category(df):
    top_categories = df['category'].value_counts().nlargest(10).index
    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x='category', y='CSAT Score', data=df[df['category'].isin(top_categories)], ax=ax)
    ax.set_title('CSAT Scores by Top 10 Issue Categories')
    ax.set_xlabel('Category')
    ax.set_ylabel('CSAT Score')
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_agent_performance(df):
    agent_scores = df.groupby('Agent_name')['CSAT Score'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10,5))
    agent_scores.head(10).plot(kind='barh', color='salmon', ax=ax)
    ax.set_title('Bottom 10 Performing Agents (Avg CSAT)')
    ax.set_xlabel('Average CSAT Score')
    ax.set_ylabel('Agent')
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10,5))
    agent_scores.tail(10).plot(kind='barh', color='seagreen', ax=ax)
    ax.set_title('Top 10 Performing Agents (Avg CSAT)')
    ax.set_xlabel('Average CSAT Score')
    ax.set_ylabel('Agent')
    st.pyplot(fig)

def plot_csat_vs_item_price(df):
    fig, ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(x='Item_price', y='CSAT Score', data=df, ax=ax)
    ax.set_title('Item Price vs CSAT Score')
    ax.set_xlabel('Item Price')
    ax.set_ylabel('CSAT Score')
    st.pyplot(fig)

def sentiment_analysis(df):
    df_text = df.dropna(subset=['Customer Remarks']).copy()
    df_text['Sentiment_Polarity'] = df_text['Customer Remarks'].apply(lambda x: TextBlob(x).sentiment.polarity)

    st.subheader("Sentiment Polarity Distribution")
    fig, ax = plt.subplots(figsize=(8,5))
    df_text['Sentiment_Polarity'].hist(bins=50, color='skyblue', ax=ax)
    ax.set_title('Distribution of Sentiment Polarity from Customer Remarks')
    ax.set_xlabel('Polarity Score (-1 = Negative, +1 = Positive)')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    st.subheader("Customer Sentiment vs CSAT Score")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.boxplot(x='CSAT Score', y='Sentiment_Polarity', data=df_text, ax=ax)
    ax.set_title('Customer Sentiment vs CSAT Score')
    st.pyplot(fig)

    negative_feedback = ' '.join(df_text[df_text['Sentiment_Polarity'] < 0]['Customer Remarks'].dropna())
    positive_feedback = ' '.join(df_text[df_text['Sentiment_Polarity'] > 0]['Customer Remarks'].dropna())

    wordcloud_neg = WordCloud(width=800, height=400, background_color='white').generate(negative_feedback)
    wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_feedback)

    st.subheader("WordClouds of Customer Feedback")
    fig, axs = plt.subplots(1, 2, figsize=(16,6))
    axs[0].imshow(wordcloud_neg, interpolation='bilinear')
    axs[0].axis('off')
    axs[0].set_title('Negative Feedback WordCloud')
    axs[1].imshow(wordcloud_pos, interpolation='bilinear')
    axs[1].axis('off')
    axs[1].set_title('Positive Feedback WordCloud')
    st.pyplot(fig)

def load_models():
    try:
        rf_model = joblib.load("random_forest_model.pkl")
        dt_model = joblib.load("decision_tree_model.pkl")
        feature_columns = joblib.load("feature_columns.pkl")
        return rf_model, dt_model, feature_columns
    except Exception as e:
        st.error(f"Error loading models or feature columns: {e}")
        return None, None, None

def prediction_ui(rf_model, dt_model, feature_columns):
    st.subheader("Customer Satisfaction Prediction")

    channel_name = st.selectbox("Support Channel", ["Chat", "Call", "Email", "Social Media", "Other"])
    category = st.selectbox("Issue Category", ["Returns", "Delivery Delays", "Product Issues", "Payment Issues", "Others"])

    # Load agent names from dataset for dropdown with debug logs
    try:
        df_agents = pd.read_csv("Customer_support_data.csv")
        st.write(f"Loaded dataset with {len(df_agents)} rows")
        agent_names = df_agents['Agent_name'].dropna().unique().tolist()
        st.write(f"Found {len(agent_names)} unique agent names")
    except Exception as e:
        st.error(f"Error loading agent names: {e}")
        agent_names = []
    agent_name = st.selectbox("Agent Name", agent_names)

    tenure_bucket = st.selectbox("Tenure Bucket", ["0-6 months", "6-12 months", "1-2 years", "2+ years"])
    agent_shift = st.selectbox("Agent Shift", ["Morning", "Afternoon", "Night"])

    input_df = pd.DataFrame({
        "channel_name": [channel_name],
        "category": [category],
        "Agent_name": [agent_name],
        "Tenure Bucket": [tenure_bucket],
        "Agent Shift": [agent_shift]
    })

    input_encoded = pd.get_dummies(input_df, drop_first=True)
    if feature_columns is not None:
        input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
    else:
        st.warning("Feature columns not loaded, prediction may not work correctly.")

    model_choice = st.radio("Choose ML Model", ["Random Forest", "Decision Tree"])
    model = rf_model if model_choice == "Random Forest" else dt_model

    if st.button("Predict"):
        if model is None:
            st.error("Model not loaded, cannot perform prediction.")
            return
        try:
            # Debug output hidden as per user request
            prediction = model.predict(input_encoded)[0]
            # st.write(f"Raw model prediction output: {prediction}")
            if prediction == 1:
                st.error("❌ Predicted: Low Customer Satisfaction")
            else:
                st.success("✅ Predicted: High Customer Satisfaction")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

def main():
    st.title("🛒 Flipkart Customer Satisfaction Analysis and Prediction")

    df = load_data()

    tabs = st.tabs(["Exploratory Data Analysis", "Sentiment Analysis", "Prediction"])

    with tabs[0]:
        st.header("Exploratory Data Analysis (EDA)")
        plot_csat_distribution(df)
        plot_csat_by_channel(df)
        plot_csat_by_category(df)
        plot_agent_performance(df)
        plot_csat_vs_item_price(df)


    with tabs[1]:
        st.header("Sentiment Analysis")
        sentiment_analysis(df)

    with tabs[2]:
        rf_model, dt_model, feature_columns = load_models()
        prediction_ui(rf_model, dt_model, feature_columns)

if __name__ == "__main__":
    main()
