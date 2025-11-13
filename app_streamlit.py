"""
Streamlit UI for Hybrid Sentiment Analysis.
"""

import streamlit as st
import requests
import json
import os
from typing import Dict, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Hybrid Sentiment Analysis",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration - use environment variable or default
# In Docker, use service name; locally, use localhost
DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")
API_BASE_URL = st.sidebar.text_input(
    "API Base URL",
    value=DEFAULT_API_URL,
    help="Base URL for the FastAPI backend"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #ffc107;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


def check_api_health() -> bool:
    """Check if API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def predict_sentiment(text: str) -> Optional[Dict]:
    """Call API to predict sentiment."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"text": text},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling API: {str(e)}")
        return None


def predict_batch(texts: List[str]) -> Optional[List[Dict]]:
    """Call API to predict sentiment for multiple texts."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            json={"texts": texts},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling API: {str(e)}")
        return None


def get_api_info() -> Optional[Dict]:
    """Get API information."""
    try:
        response = requests.get(f"{API_BASE_URL}/info", timeout=5)
        response.raise_for_status()
        return response.json()
    except:
        return None


def get_sentiment_color(sentiment: str) -> str:
    """Get color for sentiment label."""
    colors = {
        "positive": "#28a745",
        "negative": "#dc3545",
        "neutral": "#ffc107"
    }
    return colors.get(sentiment.lower(), "#6c757d")


def display_sentiment_result(result: Dict):
    """Display sentiment analysis result."""
    sentiment = result.get("sentiment", "unknown")
    confidence = result.get("confidence", 0.0)
    scores = result.get("scores", {})
    method = result.get("method", "unknown")
    
    # Sentiment label with color
    color = get_sentiment_color(sentiment)
    st.markdown(f"### Sentiment: <span style='color: {color}; font-weight: bold;'>{sentiment.upper()}</span>", unsafe_allow_html=True)
    
    # Confidence score
    st.progress(confidence)
    st.caption(f"Confidence: {confidence:.2%}")
    
    # Method used
    st.info(f"Method: {method}")
    
    # Scores breakdown
    if scores:
        st.subheader("Score Breakdown")
        scores_df = pd.DataFrame([
            {"Sentiment": k, "Score": v}
            for k, v in scores.items()
        ])
        
        # Bar chart
        fig = px.bar(
            scores_df,
            x="Sentiment",
            y="Score",
            color="Sentiment",
            color_discrete_map={
                "positive": "#28a745",
                "negative": "#dc3545",
                "neutral": "#ffc107"
            },
            title="Sentiment Scores"
        )
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed scores table
        st.dataframe(scores_df, use_container_width=True)
    
    # Model-specific scores
    col1, col2 = st.columns(2)
    
    with col1:
        if result.get("distilbert_scores"):
            st.subheader("DistilBERT Scores")
            db_df = pd.DataFrame([
                {"Sentiment": k, "Score": v}
                for k, v in result["distilbert_scores"].items()
            ])
            st.dataframe(db_df, use_container_width=True)
    
    with col2:
        if result.get("rnn_scores"):
            st.subheader("RNN Scores")
            rnn_df = pd.DataFrame([
                {"Sentiment": k, "Score": v}
                for k, v in result["rnn_scores"].items()
            ])
            st.dataframe(rnn_df, use_container_width=True)


# Main UI
st.markdown('<p class="main-header">😊 Hybrid Sentiment Analysis</p>', unsafe_allow_html=True)

# Check API health
if not check_api_health():
    st.error("⚠️ API is not available. Please ensure the FastAPI server is running.")
    st.info(f"Expected API URL: {API_BASE_URL}")
    st.stop()

# Sidebar info
with st.sidebar:
    st.header("ℹ️ API Information")
    api_info = get_api_info()
    if api_info:
        st.json(api_info)
    
    st.header("📊 Navigation")
    page = st.radio(
        "Select Page",
        ["Single Text", "Batch Analysis", "About"]
    )

# Main content based on selected page
if page == "Single Text":
    st.header("Single Text Analysis")
    
    # Text input
    text_input = st.text_area(
        "Enter text to analyze",
        height=150,
        placeholder="Type or paste your text here..."
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_button = st.button("Analyze Sentiment", type="primary")
    
    if analyze_button and text_input:
        with st.spinner("Analyzing sentiment..."):
            result = predict_sentiment(text_input.strip())
            
            if result:
                st.divider()
                display_sentiment_result(result)
    
    elif analyze_button:
        st.warning("Please enter some text to analyze.")

elif page == "Batch Analysis":
    st.header("Batch Text Analysis")
    
    # Input method selection
    input_method = st.radio(
        "Input Method",
        ["Text Area (one per line)", "CSV Upload"],
        horizontal=True
    )
    
    texts = []
    
    if input_method == "Text Area (one per line)":
        batch_text = st.text_area(
            "Enter texts (one per line)",
            height=200,
            placeholder="Text 1\nText 2\nText 3\n..."
        )
        if batch_text:
            texts = [line.strip() for line in batch_text.split("\n") if line.strip()]
    
    else:  # CSV Upload
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            help="CSV file should have a 'text' column"
        )
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                if "text" in df.columns:
                    texts = df["text"].dropna().tolist()
                    st.success(f"Loaded {len(texts)} texts from CSV")
                    st.dataframe(df.head(), use_container_width=True)
                else:
                    st.error("CSV file must have a 'text' column")
            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")
    
    if texts:
        st.info(f"Ready to analyze {len(texts)} texts")
        
        if st.button("Analyze Batch", type="primary"):
            with st.spinner(f"Analyzing {len(texts)} texts..."):
                results = predict_batch(texts)
                
                if results:
                    st.success(f"Analyzed {len(results)} texts")
                    
                    # Summary statistics
                    st.subheader("Summary Statistics")
                    sentiments = [r["sentiment"] for r in results]
                    sentiment_counts = pd.Series(sentiments).value_counts()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total", len(results))
                    with col2:
                        st.metric("Positive", sentiment_counts.get("positive", 0))
                    with col3:
                        st.metric("Negative", sentiment_counts.get("negative", 0))
                    
                    # Sentiment distribution chart
                    fig = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title="Sentiment Distribution",
                        color_discrete_map={
                            "positive": "#28a745",
                            "negative": "#dc3545",
                            "neutral": "#ffc107"
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Results table
                    st.subheader("Detailed Results")
                    results_df = pd.DataFrame([
                        {
                            "Text": texts[i][:50] + "..." if len(texts[i]) > 50 else texts[i],
                            "Sentiment": r["sentiment"],
                            "Confidence": f"{r['confidence']:.2%}",
                            "Method": r["method"]
                        }
                        for i, r in enumerate(results)
                    ])
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="sentiment_results.csv",
                        mime="text/csv"
                    )

else:  # About page
    st.header("About")
    st.markdown("""
    ## Hybrid Sentiment Analysis
    
    This application provides sentiment analysis using a hybrid approach that combines:
    
    - **DistilBERT**: Efficient transformer model (60% smaller than BERT) for contextual understanding
    - **RNN (BiLSTM)**: Bidirectional LSTM with attention mechanism for sequential pattern recognition
    - **Ensemble Methods**: Intelligent weighted combination of predictions for robust results
    
    ### Features
    
    - Single text analysis with detailed breakdown
    - Batch processing for multiple texts
    - Real-time API integration
    - Visualizations and statistics
    
    ### API Endpoints
    
    - `GET /health` - Health check
    - `POST /predict` - Single text prediction
    - `POST /predict/batch` - Batch prediction
    - `POST /train` - Train ML component
    - `GET /info` - Get analyzer configuration
    
    ### Usage
    
    1. Ensure the FastAPI backend is running
    2. Enter text in the Single Text page for individual analysis
    3. Use Batch Analysis for processing multiple texts
    4. View detailed results with confidence scores and model breakdowns
    """)
