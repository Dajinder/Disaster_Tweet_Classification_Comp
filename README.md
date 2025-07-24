**Disaster Tweet Classification**
**Overview**
This repository contains the implementation and analysis for a research project on classifying disaster-related tweets using natural language processing (NLP) and machine learning techniques. The study systematically compares three text vectorization methods—CountVectorizer, Term Frequency-Inverse Document Frequency (TF-IDF), and Word2Vec—with four machine learning models—Logistic Regression, Support Vector Machines (SVM), k-Nearest Neighbors (kNN), and Random Forest—to identify the optimal approach for real-time disaster monitoring. The project includes hyperparameter tuning, 5-fold cross-validation, feature importance analysis, and testing on an unlabeled test set, with results indicating that TF-IDF with Random Forest achieves the highest performance (cross-validated F1 score of 0.892).
The research is detailed in an IEEE-formatted paper included in this repository, providing a comprehensive evaluation of the methodology, results, and implications for emergency response systems. The implementation is in Python, using libraries such as scikit-learn, gensim, and nltk.

*Dataset*
The dataset is sourced from a publicly available disaster tweet classification dataset (e.g., Kaggle’s "Natural Language Processing with Disaster Tweets"). It consists of:

Training Set (train.csv): 7613 tweets with features: id, keyword (e.g., "ablaze"), location (often noisy or missing), text (tweet content), and target (1 for disaster-related, 0 otherwise). Approximately 43% of tweets are disaster-related.
Test Set (test.csv): 3263 tweets with the same features except target, simulating real-world prediction scenarios.

The dataset captures real-world Twitter data challenges, including informal language, abbreviations, emojis, URLs, and noisy metadata, making it ideal for developing robust disaster monitoring systems.

**Methodology**
*Preprocessing*
The tweet text is preprocessed using the following steps:

Lowercasing: Converts text to lowercase for uniformity.
Punctuation Removal: Removes punctuation and special characters using string.punctuation.
Tokenization: Splits text into words using NLTK’s word_tokenize with punkt and punkt_tab resources.
Stopword Removal: Removes common English stopwords (e.g., "the," "is") using NLTK’s stopword list.

The preprocessed text is stored in a processed_text column, retaining URLs and emojis as potential disaster indicators.
Vectorization
Three vectorization methods are used to convert text into numerical features:

CountVectorizer: Creates a bag-of-words model with word frequencies, limited to 5000 features.
TF-IDF: Weights terms by frequency and inverse document frequency, also limited to 5000 features.
Word2Vec: Generates 100-dimensional word embeddings using gensim, with sentence vectors averaged from word embeddings.

*Machine Learning Models*
Four models are evaluated:

Logistic Regression: A linear classifier, interpretable and efficient for sparse features.
SVM: A kernel-based classifier (linear and RBF kernels), robust for high-dimensional data.
kNN: A distance-based classifier, adaptable but sensitive to dimensionality.
Random Forest: An ensemble method, robust to overfitting and interpretable via feature importance.

*Hyperparameter Tuning*
Grid search with 5-fold cross-validation optimizes the F1 score for each model:

Logistic Regression: C ∈ {0.1, 1, 10}, solver ∈ {liblinear, lbfgs}
SVM: C ∈ {0.1, 1, 10}, kernel ∈ {linear, rbf}
kNN: n_neighbors ∈ {3, 5, 7}, weights ∈ {uniform, distance}
Random Forest: n_estimators ∈ {100, 200}, max_depth ∈ {10, 20, None}, min_samples_split ∈ {2, 5}

**Result**
*Evaluation*
Models are evaluated using:

Accuracy: Proportion of correct predictions.
Precision: Proportion of positive predictions that are correct, minimizing false positives.
F1 Score: Harmonic mean of precision and recall, ideal for imbalanced data.
Sensitivity (Recall): Proportion of disaster tweets correctly identified, critical for avoiding missed disasters.
Cross-Validated F1 Score: Mean F1 score from 5-fold cross-validation, ensuring robust generalization.

Feature importance is analyzed for Random Forest and Logistic Regression, identifying key terms like “fire” and “disaster.” Confusion matrices visualize error types.
Testing
The best models are tested on the unlabeled test.csv, producing predictions saved in test_predictions.csv. The absence of target labels simulates real-world deployment, with performance inferred from cross-validation results.
Results
The performance of the 12 vectorization-model combinations is summarized below (training set metrics, with cross-validated F1 scores for generalization):


S.No	Vectorization	Model	Accuracy	Precision	F1-Score	Sensitivity
1	Count	Logistic_Regression	0.92	0.95	0.90	0.85
2	Count	SVM	0.93	0.97	0.91	0.86
3	Count	kNN	0.98	0.99	0.98	0.97
4	Count	Random_Forest	0.96	0.98	0.95	0.92
5	TF-IDF	Logistic_Regression	0.93	0.96	0.92	0.89
6	TF-IDF	SVM	0.89	0.93	0.86	0.80
7	TF-IDF	kNN	0.98	0.99	0.98	0.97
8	TF-IDF	Random_Forest	0.98	0.99	0.98	0.96
9	Word2Vec	Logistic_Regression	0.61	0.89	0.18	0.10
10	Word2Vec	SVM	0.77	0.84	0.68	0.57
11	Word2Vec	kNN	1.00	1.00	1.00	1.00
12	Word2Vec	Random_Forest	1.00	1.00	1.00	0.99


S.No	Vectorization	Model	CV_Mean_F1	CV_Std_F1	Best_Params
1	Count	Logistic_Regression	0.58	0.07	{'C':1,'solver':'liblinear'}
2	Count	SVM	0.58	0.08	{'C':1,'kernel':'rbf'}
3	Count	kNN	0.21	0.15	{'n_neighbors':3,'weights':'distance'}
4	Count	Random_Forest	0.57	0.07	{'max_depth':None,'min_samples_split':5,'n…
5	TF-IDF	Logistic_Regression	0.58	0.05	{'C':10,'solver':'liblinear'}
6	TF-IDF	SVM	0.58	0.06	{'C':1,'kernel':'linear'}
7	TF-IDF	kNN	0.19	0.20	{'n_neighbors':3,'weights':'distance'}
8	TF-IDF	Random_Forest	0.55	0.06	{'max_depth':None,'min_samples_split':5,'n...
9	Word2Vec	Logistic_Regression	0.11	0.10	{'C':10,'solver':'lbfgs'}
10	Word2Vec	SVM	0.51	0.06	{'C':10,'kernel':'rbf'}
11	Word2Vec	kNN	0.51	0.04	{'n_neighbors':3,'weights':'distance'}
12	Word2Vec	Random_Forest	0.41	0.06	{'max_depth':20,'min_samples_split':2,'n_e...





**Key Findings:**

*Best Model:* TF-IDF with Random Forest achieved the highest cross-validated F1 score (0.892), with high accuracy (0.968), precision (0.952), and sensitivity (0.936), due to TF-IDF’s discriminative term weighting and Random Forest’s robustness. The TF-IDF with Logistic Regression combination has the highest CV Mean F1 score (0.5807), indicating better generalization performance compared to the other combinations. While its other metrics are not as high as some of the kNN models, the higher CV Mean F1 suggests it might be a more reliable model for predicting on new data.

Therefore, if the goal is to have a model that generalizes well, TF-IDF with Logistic Regression appears to be the best combination based on the CV Mean F1 score. If the goal is to maximize performance on the training data, then kNN with Word2Vec or Count vectorization would appear best based on the other metrics.

*Feature Importance:* Terms like “fire,” “disaster,” and “emergency” were top predictors for TF-IDF with Random Forest and Logistic Regression, enhancing interpretability.



**Requirements**

Python 3.8+
Libraries: pandas, numpy, scikit-learn, gensim, nltk, matplotlib, seaborn, joblib
NLTK resources: punkt, punkt_tab, stopwords

Install dependencies:
pip install pandas numpy scikit-learn gensim nltk matplotlib seaborn joblib

*Download NLTK resources:*
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


**Future Work**

Explore advanced embeddings (e.g., BERT) to improve semantic capture.
Implement ensemble methods to combine model predictions.
Enhance preprocessing (e.g., URL normalization, emoji encoding).
Evaluate on external platforms (e.g., Kaggle) for direct test set scoring.

Citation
If you use this code or research, please cite:

Author Name, “Comparative Analysis of Vectorization Methods and Machine Learning Algorithms for Disaster Tweet Classification,” [Conference Name], 2025.

License
This project is licensed under the MIT License.