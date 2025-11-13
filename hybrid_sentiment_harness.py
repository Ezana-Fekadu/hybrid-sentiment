"""
Hybrid Sentiment Analysis Harness
Combines DistilBERT transformer with RNN (BiLSTM) for robust sentiment analysis.
Designed for efficiency, security, and professional deployment.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import re
import logging
import warnings
from functools import lru_cache
import hashlib
import time

# Optional pandas import
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# TensorFlow and Keras for RNN (optional)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    # Create dummy classes for type hints
    tf = None
    keras = None
    layers = None
    models = None
    optimizers = None
    callbacks = None
    Tokenizer = None
    pad_sequences = None

# Transformers for DistilBERT
try:
    from transformers import (
        AutoTokenizer,
        AutoModel,
        AutoModelForSequenceClassification,
        pipeline,
        DistilBertTokenizer,
        TFDistilBertModel,
        TFAutoModel
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Check for PyTorch (needed for transformers models)
try:
    import torch
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
if HAS_TENSORFLOW:
    tf.get_logger().setLevel('ERROR')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Secure and efficient text preprocessing with input validation.
    """
    
    # Security: Maximum text length to prevent DoS attacks
    MAX_TEXT_LENGTH = 5000
    MAX_WORDS = 1000
    
    def __init__(self):
        """Initialize text preprocessor."""
        self.patterns_to_remove = [
            (r'http\S+|www\.\S+', ''),  # URLs
            (r'@\w+', ''),  # Mentions
            (r'#\w+', ''),  # Hashtags
            (r'[^\w\s]', ' '),  # Special characters (keep alphanumeric and spaces)
        ]
    
    def sanitize(self, text: str) -> str:
        """
        Sanitize input text for security and preprocessing.
        
        Args:
            text: Input text to sanitize
            
        Returns:
            Sanitized text
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        # Security: Length validation
        if len(text) > self.MAX_TEXT_LENGTH:
            logger.warning(f"Text truncated from {len(text)} to {self.MAX_TEXT_LENGTH} characters")
            text = text[:self.MAX_TEXT_LENGTH]
        
        # Remove null bytes and control characters (security)
        text = text.replace('\x00', '')
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        # Basic cleaning
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        
        return text
    
    def preprocess(self, text: str, remove_urls: bool = True) -> str:
        """
        Preprocess text for sentiment analysis.
        
        Args:
            text: Input text
            remove_urls: Whether to remove URLs
            
        Returns:
            Preprocessed text
        """
        text = self.sanitize(text)
        
        if remove_urls:
            for pattern, replacement in self.patterns_to_remove:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        text = text.strip()
        
        # Return empty string if nothing left
        if not text:
            return ""
        
        return text


class RNNSentimentModel:
    """
    RNN-based sentiment analyzer using BiLSTM with attention mechanism.
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 128,
        lstm_units: int = 128,
        max_length: int = 200,
        num_classes: int = 3,
        dropout_rate: float = 0.3
    ):
        """
        Initialize RNN sentiment model.
        
        Args:
            vocab_size: Vocabulary size for tokenizer
            embedding_dim: Embedding dimension
            lstm_units: Number of LSTM units
            max_length: Maximum sequence length
            num_classes: Number of sentiment classes
            dropout_rate: Dropout rate for regularization
        """
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required for RNNSentimentModel. Please install TensorFlow.")
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.max_length = max_length
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        self.tokenizer = None
        self.model = None
        self.is_fitted = False
        self.label_to_index = {}
        self.index_to_label = {}
    
    def _build_model(self) -> keras.Model:
        """
        Build BiLSTM model with attention mechanism.
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = layers.Input(shape=(self.max_length,), name='text_input')
        
        # Embedding layer
        embedding = layers.Embedding(
            self.vocab_size,
            self.embedding_dim,
            input_length=self.max_length,
            name='embedding'
        )(inputs)
        
        # Dropout after embedding
        embedding = layers.Dropout(self.dropout_rate)(embedding)
        
        # Bidirectional LSTM
        bilstm = layers.Bidirectional(
            layers.LSTM(
                self.lstm_units,
                return_sequences=True,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate,
                name='bilstm'
            )
        )(embedding)
        
        # Attention mechanism (self-attention)
        # Compute attention weights using a dense layer
        attention_dense = layers.Dense(1, activation='tanh', name='attention_dense')(bilstm)
        attention_weights = layers.Softmax(axis=1, name='attention_softmax')(attention_dense)
        
        # Apply attention: multiply bilstm outputs by attention weights
        # Reshape attention_weights to match bilstm dimensions for broadcasting
        attention_weights_expanded = layers.Lambda(
            lambda x: tf.expand_dims(x, axis=-1),
            name='attention_expand'
        )(attention_weights)
        attention = layers.Multiply(name='attention_apply')([bilstm, attention_weights_expanded])
        
        # Sum over sequence dimension (weighted sum)
        pooled = layers.Lambda(
            lambda x: tf.reduce_sum(x, axis=1),
            name='attention_pooling'
        )(attention)
        
        # Dense layers
        dense1 = layers.Dense(64, activation='relu', name='dense1')(pooled)
        dense1 = layers.Dropout(self.dropout_rate)(dense1)
        
        dense2 = layers.Dense(32, activation='relu', name='dense2')(dense1)
        dense2 = layers.Dropout(self.dropout_rate)(dense2)
        
        # Output layer
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax',
            name='sentiment_output'
        )(dense2)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='rnn_sentiment')
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def fit(
        self,
        texts: List[str],
        labels: List[str],
        epochs: int = 10,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: int = 1
    ) -> None:
        """
        Train the RNN model.
        
        Args:
            texts: List of training texts
            labels: List of sentiment labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Validation split ratio
            verbose: Verbosity level
        """
        if len(texts) != len(labels):
            raise ValueError("Texts and labels must have the same length")
        
        if not texts:
            raise ValueError("Training data cannot be empty")
        
        logger.info(f"Training RNN model on {len(texts)} samples")
        
        # Create label mappings
        unique_labels = sorted(set(labels))
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
        self.num_classes = len(unique_labels)
        
        # Initialize tokenizer
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(texts)
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        
        # Convert labels to categorical
        y = np.array([self.label_to_index[label] for label in labels])
        y = keras.utils.to_categorical(y, num_classes=self.num_classes)
        
        # Build model
        self.model = self._build_model()
        
        # Early stopping callback
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=verbose
        )
        
        # Train model
        self.model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=verbose
        )
        
        self.is_fitted = True
        logger.info("RNN model trained successfully")
    
    def predict_proba(self, text: str) -> Dict[str, float]:
        """
        Get sentiment probabilities for a text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping labels to probabilities
        """
        if not self.is_fitted or self.model is None or self.tokenizer is None:
            return {}
        
        try:
            # Tokenize and pad
            sequence = self.tokenizer.texts_to_sequences([text])
            X = pad_sequences(sequence, maxlen=self.max_length, padding='post', truncating='post')
            
            # Predict
            probs = self.model.predict(X, verbose=0)[0]
            
            # Map to labels
            scores = {
                self.index_to_label[i]: float(prob)
                for i, prob in enumerate(probs)
            }
            
            return scores
        except Exception as e:
            logger.error(f"Error in RNN prediction: {e}")
            return {}
    
    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        if self.model is None:
            return "Model not built yet"
        
        from io import StringIO
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = buffer = StringIO()
        self.model.summary()
        summary = buffer.getvalue()
        sys.stdout = old_stdout
        
        return summary


class HybridSentimentAnalyzer:
    """
    Hybrid sentiment analyzer combining DistilBERT and RNN (BiLSTM).
    """
    
    def __init__(
        self,
        distilbert_model: str = "distilbert-base-uncased",
        use_distilbert: bool = True,
        use_rnn: bool = True,
        ensemble_method: str = "weighted_average",
        rnn_vocab_size: int = 10000,
        rnn_embedding_dim: int = 128,
        rnn_lstm_units: int = 128,
        rnn_max_length: int = 200,
        device: str = "cpu"
    ):
        """
        Initialize the hybrid sentiment analyzer.
        
        Args:
            distilbert_model: HuggingFace DistilBERT model identifier
            use_distilbert: Whether to use DistilBERT model
            use_rnn: Whether to use RNN model
            ensemble_method: Method for combining predictions
            rnn_vocab_size: Vocabulary size for RNN
            rnn_embedding_dim: Embedding dimension for RNN
            rnn_lstm_units: Number of LSTM units
            rnn_max_length: Maximum sequence length for RNN
            device: Device to use ('cpu' or 'cuda')
        """
        self.distilbert_model_name = distilbert_model
        self.use_distilbert = use_distilbert
        self.use_rnn = use_rnn
        self.ensemble_method = ensemble_method
        self.device = device
        
        # Initialize components
        self.preprocessor = TextPreprocessor()
        self.distilbert_pipeline = None
        self.distilbert_tokenizer = None
        self.distilbert_model = None
        self.rnn_model = None
        self.is_fitted = False
        
        # Initialize DistilBERT if enabled
        if self.use_distilbert:
            if not HAS_PYTORCH and not HAS_TENSORFLOW:
                logger.warning("Neither PyTorch nor TensorFlow is available. DistilBERT cannot be loaded.")
                logger.warning("Please install PyTorch or TensorFlow to use DistilBERT model.")
                self.use_distilbert = False
            else:
                try:
                    logger.info(f"Loading DistilBERT model: {distilbert_model}")
                    
                    # Try to load a sentiment-specific DistilBERT model using pipeline
                    # Pipeline will use PyTorch by default if available, otherwise TensorFlow
                    framework = "pt" if HAS_PYTORCH else "tf"
                    try:
                        self.distilbert_pipeline = pipeline(
                            "sentiment-analysis",
                            model="distilbert-base-uncased-finetuned-sst-2-english",
                            tokenizer="distilbert-base-uncased-finetuned-sst-2-english",
                            return_all_scores=True,
                            device=-1 if device == "cpu" else 0,
                            framework=framework
                        )
                        logger.info("Loaded DistilBERT sentiment pipeline")
                    except Exception as e1:
                        logger.warning(f"Failed to load sentiment pipeline: {e1}")
                        # Try alternative model
                        try:
                            self.distilbert_pipeline = pipeline(
                                "sentiment-analysis",
                                model="distilbert-base-uncased",
                                return_all_scores=True,
                                device=-1 if device == "cpu" else 0,
                                framework=framework
                            )
                            logger.info("Loaded DistilBERT base pipeline")
                        except Exception as e2:
                            logger.error(f"Failed to load DistilBERT: {e2}")
                            self.use_distilbert = False
                    
                    if self.use_distilbert:
                        logger.info("DistilBERT model loaded successfully")
                except Exception as e:
                    logger.error(f"Error loading DistilBERT model: {e}")
                    self.use_distilbert = False
        
        # Initialize RNN model if enabled
        if self.use_rnn:
            if not HAS_TENSORFLOW:
                logger.warning("TensorFlow is not available. RNN model cannot be initialized.")
                logger.warning("Falling back to DistilBERT-only mode.")
                self.use_rnn = False
            else:
                self.rnn_model = RNNSentimentModel(
                    vocab_size=rnn_vocab_size,
                    embedding_dim=rnn_embedding_dim,
                    lstm_units=rnn_lstm_units,
                    max_length=rnn_max_length
                )
                logger.info("RNN model initialized")
    
    def fit(
        self,
        texts: List[str],
        labels: List[str],
        rnn_epochs: int = 10,
        rnn_batch_size: int = 32
    ) -> None:
        """
        Train the RNN component on labeled data.
        
        Args:
            texts: List of text samples
            labels: List of sentiment labels
            rnn_epochs: Number of epochs for RNN training
            rnn_batch_size: Batch size for RNN training
        """
        if not self.use_rnn:
            error_msg = "RNN component is disabled. Skipping training."
            logger.warning(error_msg)
            if not HAS_TENSORFLOW:
                raise ImportError("TensorFlow is required for RNN training but is not available. " + error_msg)
            return
        
        if not texts or not labels:
            raise ValueError("Training data cannot be empty")
        
        # Preprocess texts
        preprocessed_texts = [self.preprocessor.preprocess(text) for text in texts]
        
        # Filter out empty texts
        valid_indices = [i for i, text in enumerate(preprocessed_texts) if text]
        preprocessed_texts = [preprocessed_texts[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]
        
        if not preprocessed_texts:
            raise ValueError("No valid texts after preprocessing")
        
        try:
            self.rnn_model.fit(
                preprocessed_texts,
                labels,
                epochs=rnn_epochs,
                batch_size=rnn_batch_size
            )
            self.is_fitted = True
            logger.info("RNN model trained successfully")
        except Exception as e:
            logger.error(f"Error training RNN model: {e}")
            raise
    
    def _predict_distilbert(self, text: str) -> Dict[str, float]:
        """
        Get sentiment prediction from DistilBERT model.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping sentiment labels to scores
        """
        if not self.use_distilbert:
            return {}
        
        try:
            # Use pipeline if available
            if self.distilbert_pipeline is not None:
                results = self.distilbert_pipeline(text)[0]
                scores = {item['label'].lower(): item['score'] for item in results}
                return self._normalize_labels(scores)
            
            # Otherwise use base model with custom classification
            # This is a simplified version - in production, you'd want a fine-tuned classifier head
            if self.distilbert_tokenizer is not None and self.distilbert_model is not None:
                # Tokenize
                inputs = self.distilbert_tokenizer(
                    text,
                    return_tensors="tf",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                
                # Get embeddings
                outputs = self.distilbert_model(inputs)
                # Use [CLS] token embedding for classification
                # Note: This is a simplified approach. For production, use a fine-tuned model.
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                
                # For now, return neutral as fallback
                # In production, you should have a trained classifier head
                return {"neutral": 1.0}
            
            return {}
        except Exception as e:
            logger.error(f"Error in DistilBERT prediction: {e}")
            return {}
    
    def _predict_rnn(self, text: str) -> Dict[str, float]:
        """
        Get sentiment prediction from RNN model.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping sentiment labels to scores
        """
        if not self.use_rnn or not self.is_fitted or self.rnn_model is None:
            return {}
        
        try:
            preprocessed_text = self.preprocessor.preprocess(text)
            if not preprocessed_text:
                return {}
            
            scores = self.rnn_model.predict_proba(preprocessed_text)
            return scores
        except Exception as e:
            logger.error(f"Error in RNN prediction: {e}")
            return {}
    
    def _normalize_labels(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize sentiment labels to standard format.
        
        Args:
            scores: Dictionary of label-score pairs
            
        Returns:
            Normalized dictionary with standard labels
        """
        normalized = {}
        label_mapping = {
            'positive': ['positive', 'pos', '1', 'positivity', 'label_1'],
            'negative': ['negative', 'neg', '-1', '0', 'negativity', 'label_0'],
            'neutral': ['neutral', 'neu', '0', 'neutrality', 'label_2']
        }
        
        for label, score in scores.items():
            label_lower = label.lower()
            mapped = False
            for standard_label, variants in label_mapping.items():
                if any(variant in label_lower for variant in variants):
                    normalized[standard_label] = normalized.get(standard_label, 0.0) + score
                    mapped = True
                    break
            if not mapped:
                # If no mapping found, keep original
                normalized[label_lower] = score
        
        # Normalize probabilities to sum to 1
        total = sum(normalized.values())
        if total > 0:
            normalized = {k: v / total for k, v in normalized.items()}
        else:
            normalized = {"neutral": 1.0}
        
        return normalized
    
    def _ensemble_predictions(
        self,
        distilbert_scores: Dict[str, float],
        rnn_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Combine predictions from DistilBERT and RNN models.
        
        Args:
            distilbert_scores: Scores from DistilBERT model
            rnn_scores: Scores from RNN model
            
        Returns:
            Combined sentiment scores
        """
        if self.ensemble_method == "weighted_average":
            # Weight DistilBERT more heavily (0.65) as it's typically more accurate
            # RNN provides complementary sequential understanding (0.35)
            weight_distilbert = 0.65 if distilbert_scores else 0.0
            weight_rnn = 0.35 if rnn_scores else 0.0
            
            # Normalize weights
            total_weight = weight_distilbert + weight_rnn
            if total_weight == 0:
                return {}
            
            weight_distilbert /= total_weight
            weight_rnn /= total_weight
            
            # Combine scores
            all_labels = set(distilbert_scores.keys()) | set(rnn_scores.keys())
            combined = {}
            
            for label in all_labels:
                db_score = distilbert_scores.get(label, 0.0)
                rnn_score = rnn_scores.get(label, 0.0)
                combined[label] = weight_distilbert * db_score + weight_rnn * rnn_score
            
            return combined
        
        elif self.ensemble_method == "majority_vote":
            # Get predictions with highest confidence from each model
            db_pred = max(distilbert_scores.items(), key=lambda x: x[1]) if distilbert_scores else None
            rnn_pred = max(rnn_scores.items(), key=lambda x: x[1]) if rnn_scores else None
            
            votes = {}
            if db_pred:
                votes[db_pred[0]] = votes.get(db_pred[0], 0) + 1
            if rnn_pred:
                votes[rnn_pred[0]] = votes.get(rnn_pred[0], 0) + 1
            
            # Return majority vote or highest confidence if tie
            if votes:
                majority = max(votes.items(), key=lambda x: x[1])[0]
                return {majority: 1.0}
            return {}
        
        elif self.ensemble_method == "max_confidence":
            # Use prediction with highest confidence
            all_scores = {**distilbert_scores, **rnn_scores}
            if all_scores:
                max_label = max(all_scores.items(), key=lambda x: x[1])[0]
                return {max_label: all_scores[max_label]}
            return {}
        
        else:
            # Default: simple average
            all_labels = set(distilbert_scores.keys()) | set(rnn_scores.keys())
            combined = {}
            
            for label in all_labels:
                scores = []
                if label in distilbert_scores:
                    scores.append(distilbert_scores[label])
                if label in rnn_scores:
                    scores.append(rnn_scores[label])
                
                if scores:
                    combined[label] = np.mean(scores)
            
            return combined
    
    def predict(self, text: str) -> Dict[str, any]:
        """
        Predict sentiment for a single text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment prediction, scores, and metadata
        """
        # Security: Input validation
        if not isinstance(text, str):
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "scores": {"neutral": 1.0},
                "method": "invalid_input",
                "error": "Invalid input type"
            }
        
        # Preprocess and validate
        preprocessed_text = self.preprocessor.preprocess(text)
        if not preprocessed_text:
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "scores": {"neutral": 1.0},
                "method": "empty_input"
            }
        
        # Get predictions from both models
        distilbert_scores = self._predict_distilbert(preprocessed_text) if self.use_distilbert else {}
        rnn_scores = self._predict_rnn(preprocessed_text) if self.use_rnn else {}
        
        # Combine predictions
        if distilbert_scores and rnn_scores:
            combined_scores = self._ensemble_predictions(distilbert_scores, rnn_scores)
            method = "hybrid"
        elif distilbert_scores:
            combined_scores = distilbert_scores
            method = "distilbert"
        elif rnn_scores:
            combined_scores = rnn_scores
            method = "rnn"
        else:
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "scores": {"neutral": 1.0},
                "method": "fallback"
            }
        
        # Get prediction
        if combined_scores:
            sentiment = max(combined_scores.items(), key=lambda x: x[1])[0]
            confidence = combined_scores[sentiment]
        else:
            sentiment = "neutral"
            confidence = 0.0
            combined_scores = {"neutral": 1.0}
        
        return {
            "sentiment": sentiment,
            "confidence": float(confidence),
            "scores": combined_scores,
            "method": method,
            "distilbert_scores": distilbert_scores if distilbert_scores else None,
            "rnn_scores": rnn_scores if rnn_scores else None
        }
    
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, any]]:
        """
        Predict sentiment for multiple texts with efficient batching.
        
        Args:
            texts: List of texts to analyze
            batch_size: Batch size for processing
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = [self.predict(text) for text in batch]
            results.extend(batch_results)
        return results


def create_analyzer(
    distilbert_model: Optional[str] = None,
    use_distilbert: bool = True,
    use_rnn: bool = True,
    ensemble_method: str = "weighted_average",
    device: str = "cpu"
) -> HybridSentimentAnalyzer:
    """
    Factory function to create a HybridSentimentAnalyzer instance.
    
    Args:
        distilbert_model: Optional DistilBERT model identifier (uses default if None)
        use_distilbert: Whether to use DistilBERT model
        use_rnn: Whether to use RNN model
        ensemble_method: Ensemble method to use
        device: Device to use ('cpu' or 'cuda')
        
    Returns:
        Configured HybridSentimentAnalyzer instance
    """
    default_model = "distilbert-base-uncased-finetuned-sst-2-english"
    model = distilbert_model if distilbert_model else default_model
    
    return HybridSentimentAnalyzer(
        distilbert_model=model,
        use_distilbert=use_distilbert,
        use_rnn=use_rnn,
        ensemble_method=ensemble_method,
        device=device
    )
