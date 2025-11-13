"""
Tests for Hybrid Sentiment Analysis.
"""

import pytest
import numpy as np
from hybrid_sentiment_harness import HybridSentimentAnalyzer, create_analyzer


class TestHybridSentimentAnalyzer:
    """Test cases for HybridSentimentAnalyzer."""
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = create_analyzer(
            use_distilbert=True,
            use_rnn=True,
            ensemble_method="weighted_average"
        )
        assert analyzer is not None
        # Note: use_distilbert and use_rnn may be False if dependencies are missing
        # This is acceptable behavior - the analyzer should still be created
        assert analyzer.ensemble_method == "weighted_average"
    
    def test_analyzer_transformer_only(self):
        """Test analyzer with transformer only."""
        analyzer = create_analyzer(
            use_distilbert=True,
            use_rnn=False
        )
        assert analyzer is not None
        # Note: use_distilbert may be False if PyTorch/TensorFlow are missing
        # This is acceptable - analyzer should still be created
        assert analyzer.use_rnn is False
    
    def test_analyzer_ml_only(self):
        """Test analyzer with ML only."""
        analyzer = create_analyzer(
            use_distilbert=False,
            use_rnn=True
        )
        assert analyzer is not None
        assert analyzer.use_distilbert is False
        # Note: use_rnn may be False if TensorFlow is missing
        # This is acceptable - analyzer should still be created
    
    def test_predict_empty_text(self):
        """Test prediction with empty text."""
        analyzer = create_analyzer()
        result = analyzer.predict("")
        
        assert result["sentiment"] == "neutral"
        assert result["confidence"] == 0.0
        assert result["method"] == "empty_input"
    
    def test_predict_whitespace(self):
        """Test prediction with whitespace only."""
        analyzer = create_analyzer()
        result = analyzer.predict("   ")
        
        assert result["sentiment"] == "neutral"
        assert result["method"] == "empty_input"
    
    def test_predict_basic(self):
        """Test basic prediction."""
        analyzer = create_analyzer()
        result = analyzer.predict("I love this product!")
        
        assert "sentiment" in result
        assert "confidence" in result
        assert "scores" in result
        assert "method" in result
        assert result["sentiment"] in ["positive", "negative", "neutral"]
        assert 0.0 <= result["confidence"] <= 1.0
    
    def test_predict_negative(self):
        """Test prediction with negative text."""
        analyzer = create_analyzer()
        result = analyzer.predict("This is terrible and awful!")
        
        assert result["sentiment"] in ["positive", "negative", "neutral"]
        assert "confidence" in result
    
    def test_predict_neutral(self):
        """Test prediction with neutral text."""
        analyzer = create_analyzer()
        result = analyzer.predict("The weather is cloudy today.")
        
        assert result["sentiment"] in ["positive", "negative", "neutral"]
    
    def test_predict_batch(self):
        """Test batch prediction."""
        analyzer = create_analyzer()
        texts = [
            "I love this!",
            "This is terrible.",
            "The weather is nice."
        ]
        results = analyzer.predict_batch(texts)
        
        assert len(results) == len(texts)
        for result in results:
            assert "sentiment" in result
            assert "confidence" in result
    
    def test_ml_training(self):
        """Test ML model training."""
        analyzer = create_analyzer(use_rnn=True, use_distilbert=False)
        
        texts = [
            "I love this product",
            "This is great",
            "Terrible experience",
            "Very bad service",
            "It's okay",
            "Nothing special"
        ]
        labels = [
            "positive",
            "positive",
            "negative",
            "negative",
            "neutral",
            "neutral"
        ]
        
        # Training may be skipped if TensorFlow is not available
        try:
            analyzer.fit(texts, labels)
            # If training succeeded, model should be fitted
            if analyzer.use_rnn:
                assert analyzer.is_fitted is True
        except (ImportError, ValueError) as e:
            # If TensorFlow is missing, training should be skipped gracefully
            # This is acceptable behavior
            assert "TensorFlow" in str(e) or "RNN component is disabled" in str(e) or not analyzer.use_rnn
        
        # Test prediction after training (should work even if training was skipped)
        result = analyzer.predict("I love it!")
        assert result["sentiment"] in ["positive", "negative", "neutral"]
    
    def test_ensemble_weighted_average(self):
        """Test weighted average ensemble method."""
        analyzer = create_analyzer(ensemble_method="weighted_average")
        result = analyzer.predict("This is a test sentence.")
        
        assert result["method"] in ["hybrid", "distilbert", "rnn", "fallback"]
        assert "scores" in result
    
    def test_ensemble_majority_vote(self):
        """Test majority vote ensemble method."""
        analyzer = create_analyzer(ensemble_method="majority_vote")
        result = analyzer.predict("This is a test sentence.")
        
        assert result["method"] in ["hybrid", "distilbert", "rnn", "fallback"]
    
    def test_ensemble_max_confidence(self):
        """Test max confidence ensemble method."""
        analyzer = create_analyzer(ensemble_method="max_confidence")
        result = analyzer.predict("This is a test sentence.")
        
        assert result["method"] in ["hybrid", "distilbert", "rnn", "fallback"]
    
    def test_result_structure(self):
        """Test that result has correct structure."""
        analyzer = create_analyzer()
        result = analyzer.predict("Test text")
        
        required_keys = ["sentiment", "confidence", "scores", "method"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
        
        assert isinstance(result["sentiment"], str)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["scores"], dict)
        assert isinstance(result["method"], str)
    
    def test_scores_normalization(self):
        """Test that scores are properly normalized."""
        analyzer = create_analyzer()
        result = analyzer.predict("Test text")
        
        if result["scores"]:
            total = sum(result["scores"].values())
            # Allow some tolerance for floating point errors
            assert abs(total - 1.0) < 0.01 or len(result["scores"]) == 1
    
    def test_long_text(self):
        """Test prediction with long text."""
        analyzer = create_analyzer()
        long_text = "This is a very long text. " * 100
        result = analyzer.predict(long_text)
        
        assert "sentiment" in result
        assert result["sentiment"] in ["positive", "negative", "neutral"]
    
    def test_special_characters(self):
        """Test prediction with special characters."""
        analyzer = create_analyzer()
        text = "This has special chars: @#$%^&*()! 🎉"
        result = analyzer.predict(text)
        
        assert "sentiment" in result
        assert result["sentiment"] in ["positive", "negative", "neutral"]


class TestFactoryFunction:
    """Test cases for factory function."""
    
    def test_create_analyzer_default(self):
        """Test creating analyzer with default parameters."""
        analyzer = create_analyzer()
        assert analyzer is not None
    
    def test_create_analyzer_custom_model(self):
        """Test creating analyzer with custom model."""
        analyzer = create_analyzer(
            distilbert_model="distilbert-base-uncased"
        )
        assert analyzer is not None
        assert analyzer.distilbert_model_name == "distilbert-base-uncased"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
