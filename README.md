# Automated Language Detection System

## Overview
This project implements a machine learning system to identify the language of a given text. It utilizes a Multinomial Naive Bayes classifier trained on character-level n-gram features. The system achieved an accuracy of 98.07% across 17 languages during testing.

## Technical Methodology
* **Feature Extraction**: The system uses character bigrams and trigrams (n-gram range 2-3) to capture linguistic patterns.
* **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert text into numerical vectors, limited to the top 5000 features for computational efficiency.
* **Classification**: A Multinomial Naive Bayes algorithm is employed for probabilistic classification.

## Mathematical Foundation

### Bayes' Theorem
The classifier is built on Bayes' Theorem, which computes the probability of a language given the input text:

$$P(L \mid T) = \frac{P(T \mid L) \cdot P(L)}{P(T)}$$

Where:
- $P(L \mid T)$ is the posterior probability of language $L$ given text $T$
- $P(T \mid L)$ is the likelihood of observing text $T$ under language $L$
- $P(L)$ is the prior probability of language $L$
- $P(T)$ is the marginal probability of text $T$

### TF-IDF Vectorization
Character n-grams are weighted using TF-IDF to reflect their importance:

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\left(\frac{N}{\text{df}(t)}\right)$$

Where $\text{TF}(t, d)$ is the frequency of n-gram $t$ in document $d$, $N$ is the total number of documents, and $\text{df}(t)$ is the number of documents containing $t$.

### Laplace Smoothing
To handle n-grams unseen during training, additive (Laplace) smoothing is applied:

$$P(w \mid L) = \frac{\text{count}(w, L) + \alpha}{\text{count}(L) + \alpha \cdot V}$$

Where $\alpha = 1.0$ (default), and $V$ is the vocabulary size.

### Evaluation Metrics
| Metric | Formula | Description |
|--------|---------|-------------|
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Overall correctness |
| **Precision** | $\frac{TP}{TP + FP}$ | Exactness of predictions |
| **Recall** | $\frac{TP}{TP + FN}$ | Completeness of predictions |
| **F1 Score** | $2 \cdot \frac{P \cdot R}{P + R}$ | Harmonic mean of precision and recall |

## Performance
The model provides high-precision identification for common languages. In industrial contexts (Industry 4.0), similar text classification tasks may utilize Support Vector Machines (SVM) or Random Forest for increased precision on complex unstructured data.

## Repository Contents
* `language_detection_system.ipynb`: Complete Jupyter Notebook with data cleaning, training, and evaluation.
* `dataset/language-detection.csv`: The labeled language dataset sourced from Wikipedia.
* `web/`: Minimal project link page for source code, live demo, and data visualizations.
* `streamlit_app/`: Interactive Streamlit demo for language prediction.

## Requirements
* Python 3.x
* pandas
* scikit-learn
* matplotlib
* seaborn
* streamlit

## Running the Live Demo
Install the Streamlit app requirements:

```bash
pip install -r streamlit_app/requirements.txt
```

Run the demo:

```bash
streamlit run streamlit_app/app.py
```

The app trains the same TF-IDF character n-gram and Multinomial Naive Bayes pipeline from the dataset, then lets users enter text and view the predicted language with confidence scores.
