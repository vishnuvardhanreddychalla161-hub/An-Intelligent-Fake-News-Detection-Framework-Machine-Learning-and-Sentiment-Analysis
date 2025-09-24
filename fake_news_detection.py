#!/usr/bin/env python3
"""
Intelligent Fake News Detection Framework using Machine Learning and Sentiment Analysis
Author: Manus AI
Date: September 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from textblob import TextBlob
import re
import warnings
warnings.filterwarnings('ignore')

class FakeNewsDetector:
    """
    A comprehensive fake news detection system using machine learning and sentiment analysis.
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'svm': SVC(random_state=42, probability=True),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        self.trained_models = {}
        self.feature_names = []
        
    def preprocess_text(self, text):
        """
        Preprocess text data by cleaning and normalizing.
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            str: Cleaned and preprocessed text
        """
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_sentiment_features(self, text):
        """
        Extract sentiment-based features from text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Dictionary containing sentiment features
        """
        blob = TextBlob(text)
        
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity,
            'word_count': len(text.split()),
            'char_count': len(text),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
        }
    
    def prepare_features(self, df):
        """
        Prepare features for machine learning models.
        
        Args:
            df (DataFrame): Input dataframe with text data
            
        Returns:
            tuple: Feature matrix and labels
        """
        # Preprocess text
        df['cleaned_text'] = df['text'].apply(self.preprocess_text)
        
        # Extract TF-IDF features
        tfidf_features = self.vectorizer.fit_transform(df['cleaned_text'])
        
        # Extract sentiment features
        sentiment_features = df['cleaned_text'].apply(self.extract_sentiment_features)
        sentiment_df = pd.DataFrame(sentiment_features.tolist())
        
        # Combine features
        feature_matrix = np.hstack([
            tfidf_features.toarray(),
            sentiment_df.values
        ])
        
        # Store feature names
        self.feature_names = (
            list(self.vectorizer.get_feature_names_out()) + 
            list(sentiment_df.columns)
        )
        
        return feature_matrix, df['label'].values
    
    def train_models(self, X_train, y_train):
        """
        Train multiple machine learning models.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print("Training models...")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            
        print("Model training completed!")
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate trained models on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Evaluation results for each model
        """
        results = {}
        
        for name, model in self.trained_models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
        return results
    
    def predict(self, text, model_name='logistic_regression'):
        """
        Predict if a news article is fake or real.
        
        Args:
            text (str): News article text
            model_name (str): Name of the model to use
            
        Returns:
            dict: Prediction result with confidence
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet!")
        
        # Preprocess text
        cleaned_text = self.preprocess_text(text)
        
        # Extract features
        tfidf_features = self.vectorizer.transform([cleaned_text])
        sentiment_features = self.extract_sentiment_features(cleaned_text)
        
        # Combine features
        features = np.hstack([
            tfidf_features.toarray(),
            np.array(list(sentiment_features.values())).reshape(1, -1)
        ])
        
        # Make prediction
        model = self.trained_models[model_name]
        prediction = model.predict(features)[0]
        confidence = model.predict_proba(features)[0].max() if hasattr(model, 'predict_proba') else None
        
        return {
            'prediction': 'Fake' if prediction == 1 else 'Real',
            'confidence': confidence,
            'sentiment_features': sentiment_features
        }

def create_sample_dataset():
    """
    Create a sample dataset for demonstration purposes.
    This simulates the structure of the FakeNewsNet dataset.
    """
    np.random.seed(42)
    
    # Sample real news headlines and content
    real_news = [
        "Scientists discover new species of marine life in deep ocean exploration",
        "Local community comes together to support flood victims with donations and volunteers",
        "New renewable energy project to provide clean power to thousands of homes",
        "University researchers develop innovative treatment for rare genetic disorder",
        "City council approves budget for new public transportation system",
        "International cooperation leads to breakthrough in climate change research",
        "Local school receives award for excellence in STEM education programs",
        "New archaeological findings shed light on ancient civilization practices",
        "Healthcare workers receive recognition for their dedication during pandemic",
        "Technology company announces partnership with environmental conservation group"
    ]
    
    # Sample fake news headlines and content
    fake_news = [
        "SHOCKING: Government secretly controls weather using hidden technology!!!",
        "BREAKING: Celebrity reveals aliens told them about upcoming invasion",
        "URGENT: Doctors don't want you to know this ONE WEIRD TRICK",
        "EXPOSED: Big pharma hiding miracle cure that costs only pennies",
        "ALERT: New study proves vaccines contain mind control chips",
        "SCANDAL: Politicians caught in massive conspiracy to hide the truth",
        "WARNING: Your smartphone is slowly killing you - here's proof",
        "REVEALED: Ancient secret that billionaires use to stay young forever",
        "CRISIS: Scientists confirm the world will end next month",
        "EXCLUSIVE: Leaked documents show government plans to ban all freedom"
    ]
    
    # Create dataset
    data = []
    
    # Add real news (label = 0)
    for news in real_news:
        data.append({
            'text': news,
            'label': 0,
            'source': 'reliable_news_source'
        })
    
    # Add fake news (label = 1)
    for news in fake_news:
        data.append({
            'text': news,
            'label': 1,
            'source': 'questionable_source'
        })
    
    # Create additional samples with variations
    for i in range(40):  # Add more samples for better training
        if i % 2 == 0:
            # Real news variations
            real_templates = [
                f"Research team at university makes breakthrough in {['medicine', 'technology', 'science'][i%3]}",
                f"Local government announces new initiative for {['education', 'healthcare', 'infrastructure'][i%3]}",
                f"International conference discusses solutions for {['climate change', 'poverty', 'disease'][i%3]}"
            ]
            data.append({
                'text': real_templates[i%3],
                'label': 0,
                'source': 'reliable_news_source'
            })
        else:
            # Fake news variations
            fake_templates = [
                f"SHOCKING discovery: {['Scientists', 'Doctors', 'Experts'][i%3]} hide this from you!!!",
                f"BREAKING: Secret {['government', 'corporate', 'alien'][i%3]} conspiracy exposed",
                f"URGENT: This {['miracle cure', 'hidden truth', 'secret method'][i%3]} will change everything"
            ]
            data.append({
                'text': fake_templates[i%3],
                'label': 1,
                'source': 'questionable_source'
            })
    
    return pd.DataFrame(data)

def main():
    """
    Main function to run the fake news detection system.
    """
    print("=== Intelligent Fake News Detection Framework ===")
    print("Loading and preparing dataset...")
    
    # Create sample dataset (in real implementation, load FakeNewsNet dataset)
    df = create_sample_dataset()
    
    print(f"Dataset loaded: {len(df)} articles")
    print(f"Real news: {len(df[df['label'] == 0])}")
    print(f"Fake news: {len(df[df['label'] == 1])}")
    
    # Initialize detector
    detector = FakeNewsDetector()
    
    # Prepare features
    print("\nPreparing features...")
    X, y = detector.prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train models
    detector.train_models(X_train, y_train)
    
    # Evaluate models
    print("\nEvaluating models...")
    results = detector.evaluate_models(X_test, y_test)
    
    # Print results
    print("\n=== Model Performance ===")
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
    
    # Test prediction on sample text
    print("\n=== Sample Predictions ===")
    
    test_articles = [
        "Scientists at MIT have developed a new method for detecting cancer cells early.",
        "SHOCKING: Government hiding alien technology in secret underground base!!!"
    ]
    
    for article in test_articles:
        result = detector.predict(article)
        print(f"\nArticle: {article[:60]}...")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
    
    return detector, results

if __name__ == "__main__":
    detector, results = main()

