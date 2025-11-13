"""
Training script for Hybrid Sentiment Analyzer RNN component.
This script trains the RNN model on a curated sentiment dataset.
"""

import logging
from hybrid_sentiment_harness import create_analyzer
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_training_data():
    """
    Generate a comprehensive training dataset for sentiment analysis.
    Returns texts and labels for training.
    """
    texts = [
        # Positive sentiments
        "I love this product! It's amazing and works perfectly.",
        "This is the best purchase I've ever made. Highly recommended!",
        "Excellent quality and great customer service. Very satisfied!",
        "Outstanding product with fantastic features. Love it!",
        "Amazing experience! This exceeded all my expectations.",
        "Wonderful service and top-notch quality. Highly impressed!",
        "Great value for money. I'm very happy with this purchase.",
        "Perfect! Exactly what I was looking for. Very pleased!",
        "Fantastic product! Works great and looks beautiful.",
        "I'm thrilled with this purchase. Excellent quality!",
        "Superb quality and fast delivery. Very satisfied customer!",
        "This is incredible! Best product I've used in years.",
        "Love it! Great design and excellent functionality.",
        "Amazing! This product is a game changer.",
        "Outstanding! Highly recommend to everyone.",
        "Excellent! Better than I expected. Very happy!",
        "Perfect product! Great quality and value.",
        "Fantastic! This is exactly what I needed.",
        "Wonderful experience! Very satisfied with my purchase.",
        "Great product! I'm very impressed with the quality.",
        
        # Negative sentiments
        "This is terrible! Poor quality and doesn't work at all.",
        "Very disappointed with this purchase. Waste of money.",
        "Awful product! Broke after just one use. Not recommended.",
        "Terrible experience. Customer service was unhelpful.",
        "Poor quality product. Not worth the price at all.",
        "This is a complete waste of money. Very unsatisfied.",
        "Horrible product! Does not meet expectations at all.",
        "Very bad quality. I regret buying this item.",
        "Disappointing purchase. Product is defective.",
        "Terrible! This product is a scam. Avoid at all costs.",
        "Poor service and bad quality. Very unhappy customer.",
        "This is awful! Not what I expected. Very disappointed.",
        "Bad product! Does not work as advertised.",
        "Terrible quality! I want a refund immediately.",
        "Very poor experience. Would not recommend to anyone.",
        "Disappointing! Product arrived damaged and broken.",
        "Awful! This is the worst purchase I've ever made.",
        "Poor quality and terrible customer service.",
        "Bad product! Not worth buying at all.",
        "Terrible! I'm very unsatisfied with this purchase.",
        
        # Neutral sentiments
        "The weather is cloudy today. It might rain later.",
        "I received the package yesterday. It arrived on time.",
        "The product is okay. Nothing special but it works.",
        "It's a standard item. Does what it's supposed to do.",
        "The service was average. Nothing exceptional.",
        "This is a regular product. Meets basic expectations.",
        "The item arrived as expected. Standard quality.",
        "It's fine. Not great but not bad either.",
        "The product is acceptable. Does the job.",
        "This is a normal item. Nothing to complain about.",
        "The service was adequate. Met my basic needs.",
        "It's an ordinary product. Works as described.",
        "The item is standard. No issues but no surprises.",
        "This is okay. Nothing remarkable about it.",
        "The product is fine. It serves its purpose.",
        "It's a typical item. Average quality and performance.",
        "The service was normal. Nothing special.",
        "This is a regular product. Standard features.",
        "The item is acceptable. Meets minimum requirements.",
        "It's fine. Does what it needs to do.",
    ]
    
    labels = [
        # Positive (20 samples)
        "positive", "positive", "positive", "positive", "positive",
        "positive", "positive", "positive", "positive", "positive",
        "positive", "positive", "positive", "positive", "positive",
        "positive", "positive", "positive", "positive", "positive",
        
        # Negative (20 samples)
        "negative", "negative", "negative", "negative", "negative",
        "negative", "negative", "negative", "negative", "negative",
        "negative", "negative", "negative", "negative", "negative",
        "negative", "negative", "negative", "negative", "negative",
        
        # Neutral (20 samples)
        "neutral", "neutral", "neutral", "neutral", "neutral",
        "neutral", "neutral", "neutral", "neutral", "neutral",
        "neutral", "neutral", "neutral", "neutral", "neutral",
        "neutral", "neutral", "neutral", "neutral", "neutral",
    ]
    
    return texts, labels


def main():
    """Main training function."""
    logger.info("=" * 60)
    logger.info("Hybrid Sentiment Analyzer - Training Script")
    logger.info("=" * 60)
    
    # Create analyzer with RNN enabled
    logger.info("Initializing analyzer...")
    analyzer = create_analyzer(
        use_distilbert=True,
        use_rnn=True,
        ensemble_method="weighted_average"
    )
    
    # Generate training data
    logger.info("Generating training dataset...")
    texts, labels = generate_training_data()
    logger.info(f"Training dataset: {len(texts)} samples")
    logger.info(f"  - Positive: {labels.count('positive')} samples")
    logger.info(f"  - Negative: {labels.count('negative')} samples")
    logger.info(f"  - Neutral: {labels.count('neutral')} samples")
    
    # Train the RNN model
    logger.info("\nStarting RNN model training...")
    start_time = time.time()
    
    try:
        analyzer.fit(
            texts=texts,
            labels=labels,
            rnn_epochs=15,
            rnn_batch_size=16
        )
        
        training_time = time.time() - start_time
        logger.info(f"\nTraining completed successfully!")
        logger.info(f"Training time: {training_time:.2f} seconds")
        logger.info(f"Model is fitted: {analyzer.is_fitted}")
        
        # Test predictions
        logger.info("\n" + "=" * 60)
        logger.info("Testing trained model with sample predictions...")
        logger.info("=" * 60)
        
        test_texts = [
            "I absolutely love this!",
            "This is terrible and awful.",
            "The weather is nice today."
        ]
        
        for text in test_texts:
            result = analyzer.predict(text)
            logger.info(f"\nText: '{text}'")
            logger.info(f"  Sentiment: {result['sentiment']}")
            logger.info(f"  Confidence: {result['confidence']:.4f}")
            logger.info(f"  Method: {result['method']}")
            if result.get('rnn_scores'):
                logger.info(f"  RNN Scores: {result['rnn_scores']}")
        
        logger.info("\n" + "=" * 60)
        logger.info("Training completed successfully!")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

