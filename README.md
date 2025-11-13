# Hybrid Sentiment Analyzer

A professional, efficient, and secure hybrid sentiment analysis system combining **DistilBERT** transformer and **RNN (BiLSTM)** models.

## Features

- **DistilBERT Integration**: Efficient transformer model (60% smaller than BERT) for contextual understanding
- **RNN (BiLSTM) with Attention**: Bidirectional LSTM with attention mechanism for sequential pattern recognition
- **Hybrid Ensemble**: Intelligent weighted combination of predictions for robust results
- **Security**: Input validation, sanitization, and length limits
- **Efficiency**: Batch processing, model caching, and optimized inference
- **Production Ready**: FastAPI backend and Streamlit UI

## Architecture

### Models

1. **DistilBERT**: Pre-trained transformer model fine-tuned for sentiment analysis
   - Efficient: 60% smaller and 60% faster than BERT
   - Contextual understanding of language
   - Pre-trained on large corpora

2. **RNN (BiLSTM)**: Bidirectional LSTM with attention mechanism
   - Captures sequential dependencies
   - Attention mechanism focuses on important parts
   - Trainable on custom datasets

### Ensemble Methods

- **Weighted Average** (default): Combines predictions with weights (65% DistilBERT, 35% RNN)
- **Majority Vote**: Uses the prediction with highest confidence from each model
- **Max Confidence**: Uses the prediction with the highest overall confidence

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd hybrid-sentiment

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from hybrid_sentiment_harness import create_analyzer

# Create analyzer
analyzer = create_analyzer(
    use_distilbert=True,
    use_rnn=True,
    ensemble_method="weighted_average"
)

# Predict sentiment
result = analyzer.predict("I love this product!")
print(result)
# {
#     "sentiment": "positive",
#     "confidence": 0.95,
#     "scores": {"positive": 0.95, "negative": 0.03, "neutral": 0.02},
#     "method": "hybrid",
#     "distilbert_scores": {...},
#     "rnn_scores": {...}
# }
```

### Training RNN Model

```python
# Prepare training data
texts = [
    "I love this product!",
    "This is terrible.",
    "The weather is nice."
]
labels = ["positive", "negative", "neutral"]

# Train RNN component
analyzer.fit(texts, labels, rnn_epochs=10)
```

### Batch Prediction

```python
texts = [
    "I love this!",
    "This is terrible.",
    "The weather is nice."
]
results = analyzer.predict_batch(texts, batch_size=32)
```

## API Deployment

### FastAPI Server

```bash
# Start the FastAPI server
python deploy_fastapi.py

# Or using uvicorn
uvicorn deploy_fastapi:app --host 0.0.0.0 --port 8000
```

### API Endpoints

- `GET /`: Root endpoint
- `GET /health`: Health check
- `POST /predict`: Single text prediction
- `POST /predict/batch`: Batch prediction
- `POST /train`: Train RNN model
- `GET /info`: Get analyzer configuration

### Example API Request

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "I love this product!"}'
```

## Streamlit UI

```bash
# Start the Streamlit UI
streamlit run app_streamlit.py
```

The UI provides:
- Single text analysis
- Batch text analysis
- CSV upload support
- Visualization of results
- Model performance metrics

## Security

See [SECURITY.md](SECURITY.md) for detailed security considerations.

Key security features:
- Input validation and sanitization
- Text length limits (5,000 characters)
- Type validation
- Error handling without information leakage
- Resource management

## Configuration

### Model Parameters

```python
analyzer = create_analyzer(
    distilbert_model="distilbert-base-uncased-finetuned-sst-2-english",
    use_distilbert=True,
    use_rnn=True,
    ensemble_method="weighted_average",  # or "majority_vote", "max_confidence"
    rnn_vocab_size=10000,
    rnn_embedding_dim=128,
    rnn_lstm_units=128,
    rnn_max_length=200,
    device="cpu"  # or "cuda" for GPU
)
```

## Performance

- **DistilBERT**: ~60% faster than BERT with 97% of performance
- **RNN**: Fast inference after training
- **Batch Processing**: Optimized for batch predictions
- **Memory**: Efficient memory usage with model caching

## Testing

```bash
# Run tests
pytest test_hybrid_sentiment.py -v

# With coverage
coverage run -m pytest test_hybrid_sentiment.py
coverage report
```

## Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build
```

## Requirements

- Python 3.10+
- TensorFlow 2.14+
- Transformers 4.45+
- See `requirements.txt` for full list

## License

[Add your license here]

## Contributing

[Add contributing guidelines here]

## Acknowledgments

- DistilBERT by HuggingFace
- TensorFlow/Keras for RNN implementation
- FastAPI for API framework
- Streamlit for UI framework
