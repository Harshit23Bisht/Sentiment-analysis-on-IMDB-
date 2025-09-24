# IMDb Sentiment Analysis

## Project Overview
This project classifies IMDb movie reviews as positive or negative using machine learning. I trained three different models on 50,000 reviews to determine which approach works best.

## My Approach

### Data Preprocessing
- Cleaned 50,000 IMDb reviews (balanced positive/negative)
- Removed HTML tags, URLs, and special characters
- Applied tokenization, stopword removal, and stemming
- Used TF-IDF vectorization to convert text to numerical features

### Models Trained
1. **Logistic Regression** - Simple linear approach
2. **Support Vector Machine (SVM)** - Pattern recognition with hyperparameter tuning
3. **Naive Bayes** - Probabilistic classifier

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| SVM | **89.41%** | 0.90/0.89 | 0.88/0.90 | 0.89 |
| Logistic Regression | 89.06% | 0.90/0.88 | 0.88/0.90 | 0.89 |
| Naive Bayes | 86.44% | 0.86/0.87 | 0.87/0.86 | 0.86 |

## Key Findings
- **SVM performed best** with 89.41% accuracy after hyperparameter tuning
- All models showed balanced performance for both positive and negative reviews
- Proper text preprocessing significantly improved model performance
- TF-IDF vectorization effectively captured important text features

## Tools Used
- **Python**: pandas, scikit-learn, nltk, matplotlib
- **Techniques**: Text preprocessing, TF-IDF, GridSearchCV, confusion matrices
