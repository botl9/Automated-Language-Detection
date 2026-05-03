# Automated Language Detection System

## Overview
This project implements a machine learning system to identify the language of a given text. It utilizes a Multinomial Naive Bayes classifier trained on character-level n-gram features. The system achieved an accuracy of 98.07% across 17 languages during testing.

## Technical Methodology
* **Feature Extraction**: The system uses character bigrams and trigrams (n-gram range 2-3) to capture linguistic patterns.
* **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert text into numerical vectors, limited to the top 5000 features for computational efficiency.
* **Classification**: A Multinomial Naive Bayes algorithm is employed for probabilistic classification.

## Performance
The model provides high-precision identification for common languages. In industrial contexts (Industry 4.0), similar text classification tasks may utilize Support Vector Machines (SVM) or Random Forest for increased precision on complex unstructured data.

## Repository Contents
* `language_detection_system.ipynb`: Complete Jupyter Notebook with data cleaning, training, and evaluation.
* `dataset/language-detection.csv`: The labeled language dataset sourced from Wikipedia.
* `web/`: Minimal project link page for source code, live demo, and data visualizations.

## Requirements
* Python 3.x
* pandas
* scikit-learn
* matplotlib
* seaborn
