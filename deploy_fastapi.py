"""
FastAPI deployment for Hybrid Sentiment Analysis API.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
from hybrid_sentiment_harness import HybridSentimentAnalyzer, create_analyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Hybrid Sentiment Analysis API",
    description="API for hybrid sentiment analysis combining transformer and ML models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global analyzer instance
analyzer: Optional[HybridSentimentAnalyzer] = None


# Pydantic models for request/response
class SentimentRequest(BaseModel):
    text: str = Field(..., description="Text to analyze for sentiment", min_length=1)


class BatchSentimentRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to analyze", min_items=1)


class SentimentResponse(BaseModel):
    sentiment: str = Field(..., description="Predicted sentiment label")
    confidence: float = Field(..., description="Confidence score (0-1)")
    scores: Dict[str, float] = Field(..., description="Scores for all sentiment classes")
    method: str = Field(..., description="Method used for prediction")
    distilbert_scores: Optional[Dict[str, float]] = Field(None, description="DistilBERT model scores")
    rnn_scores: Optional[Dict[str, float]] = Field(None, description="RNN model scores")


class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse] = Field(..., description="List of sentiment predictions")
    total: int = Field(..., description="Total number of predictions")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether models are loaded")
    distilbert_enabled: bool = Field(..., description="Whether DistilBERT is enabled")
    rnn_enabled: bool = Field(..., description="Whether RNN is enabled")
    rnn_fitted: bool = Field(..., description="Whether RNN model is fitted")


@app.on_event("startup")
async def startup_event():
    """Initialize the analyzer on startup."""
    global analyzer
    try:
        logger.info("Initializing Hybrid Sentiment Analyzer...")
        analyzer = create_analyzer(
            use_distilbert=True,
            use_rnn=True,
            ensemble_method="weighted_average"
        )
        logger.info("Analyzer initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing analyzer: {e}")
        # Create analyzer with DistilBERT only as fallback
        analyzer = create_analyzer(
            use_distilbert=True,
            use_rnn=False,
            ensemble_method="max_confidence"
        )
        logger.warning("Initialized analyzer with DistilBERT only (RNN disabled)")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Hybrid Sentiment Analysis API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    global analyzer
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
    
    return HealthResponse(
        status="healthy",
        model_loaded=analyzer is not None,
        distilbert_enabled=analyzer.use_distilbert,
        rnn_enabled=analyzer.use_rnn,
        rnn_fitted=analyzer.is_fitted
    )


@app.post("/predict", response_model=SentimentResponse, tags=["Sentiment Analysis"])
async def predict_sentiment(request: SentimentRequest):
    """
    Predict sentiment for a single text.
    
    Args:
        request: SentimentRequest containing the text to analyze
        
    Returns:
        SentimentResponse with prediction results
    """
    global analyzer
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
    
    try:
        result = analyzer.predict(request.text)
        return SentimentResponse(**result)
    except Exception as e:
        logger.error(f"Error predicting sentiment: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.post("/predict/batch", response_model=BatchSentimentResponse, tags=["Sentiment Analysis"])
async def predict_sentiment_batch(request: BatchSentimentRequest):
    """
    Predict sentiment for multiple texts in batch.
    
    Args:
        request: BatchSentimentRequest containing list of texts
        
    Returns:
        BatchSentimentResponse with list of predictions
    """
    global analyzer
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
    
    try:
        results = analyzer.predict_batch(request.texts)
        return BatchSentimentResponse(
            results=[SentimentResponse(**result) for result in results],
            total=len(results)
        )
    except Exception as e:
        logger.error(f"Error predicting sentiment batch: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing batch request: {str(e)}")


@app.post("/train", tags=["Training"])
async def train_ml_model(
    texts: List[str],
    labels: List[str],
    background_tasks: BackgroundTasks
):
    """
    Train the ML component of the hybrid analyzer.
    
    Args:
        texts: List of training texts
        labels: List of corresponding sentiment labels
        background_tasks: FastAPI background tasks
        
    Returns:
        Success message
    """
    global analyzer
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
    
    if not analyzer.use_rnn:
        raise HTTPException(status_code=400, detail="RNN component is disabled")
    
    if len(texts) != len(labels):
        raise HTTPException(status_code=400, detail="Texts and labels must have the same length")
    
    if len(texts) == 0:
        raise HTTPException(status_code=400, detail="Training data cannot be empty")
    
    try:
        # Train in background to avoid blocking
        def train():
            analyzer.fit(texts, labels)
        
        background_tasks.add_task(train)
        
        return {
            "message": "Training started",
            "samples": len(texts),
            "status": "processing"
        }
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")


@app.get("/info", tags=["Info"])
async def get_info():
    """Get information about the analyzer configuration."""
    global analyzer
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
    
    return {
        "distilbert_model": analyzer.distilbert_model_name,
        "use_distilbert": analyzer.use_distilbert,
        "use_rnn": analyzer.use_rnn,
        "ensemble_method": analyzer.ensemble_method,
        "rnn_fitted": analyzer.is_fitted
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
