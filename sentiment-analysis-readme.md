# IMDB Movie Review Sentiment Analysis

## Project Overview

This project performs sentiment analysis on movie reviews from the IMDB dataset using machine learning techniques. The goal is to classify movie reviews as either positive or negative based on their text content.

## Data Preprocessing

### Text Cleaning Steps:
1. **Lowercasing**: Convert all text to lowercase
2. **HTML Tag Removal**: Remove `<br>` tags and other HTML elements
3. **URL Removal**: Remove HTTP/HTTPS links  
4. **Special Character Removal**: Clean special characters and punctuation
5. **Tokenization**: Split text into individual words using NLTK
6. **Stop Word Removal**: Remove common English stop words
7. **Stemming**: Apply Porter Stemmer to reduce words to root forms
8. **Duplicate Removal**: Remove 421 duplicate entries

### Feature Engineering:
- **Word Count Analysis**: Calculate number of words per review
- **TF-IDF Vectorization**: Convert text to numerical features using TF-IDF
- **Vocabulary Analysis**: Most frequent words in positive vs negative reviews

## Machine Learning Models

Three classification algorithms were implemented and compared:

### 1. Logistic Regression
- **Accuracy**: 89.06%
- **Precision**: 0.90 (negative), 0.88 (positive)
- **Recall**: 0.88 (negative), 0.90 (positive)
- **F1-Score**: 0.89 for both classes

### 2. Support Vector Machine (LinearSVC)
- **Accuracy**: 89.22% (default), 89.41% (C=1, hinge loss)
- **Precision**: 0.90 (negative), 0.89 (positive)
- **Recall**: 0.88 (negative), 0.90 (positive)  
- **F1-Score**: 0.89 for both classes

### 3. Multinomial Naive Bayes
- **Accuracy**: 86.44%
- **Precision**: 0.86 (negative), 0.87 (positive)
- **Recall**: 0.87 (negative), 0.86 (positive)
- **F1-Score**: 0.86 for both classes

## Key Findings

### Model Performance:
- **Best Performer**: LinearSVC with 89.41% accuracy
- **Most Balanced**: Logistic Regression with consistent metrics
- **Baseline**: Naive Bayes at 86.44% accuracy

### Text Analysis Insights:
- **Positive Reviews**: Common words include "film", "movie", "good", "great", "story"
- **Negative Reviews**: More varied vocabulary with complaint-focused language
- **Review Length**: Positive reviews tend to be slightly longer on average
- **Word Distribution**: Shows clear patterns between sentiment classes

## Technical Implementation

### Libraries Used:
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical operations
- **scikit-learn**: Machine learning algorithms and metrics
- **nltk**: Natural language processing
- **matplotlib/seaborn**: Data visualization
- **re**: Regular expressions for text cleaning

### Model Training:
- **Train/Test Split**: 70% training, 30% testing
- **Cross-validation**: Consistent performance across models
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score, Confusion Matrix

## Results

All three models achieved strong performance with accuracies between 86-89%. The LinearSVC model performed best overall, while Logistic Regression provided the most interpretable and balanced results.

