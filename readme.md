# Text Similarity App

This project is a **Text Similarity Application** that predicts whether two questions are duplicates based on their textual similarity. It combines **traditional text-based features**, **TF-IDF cosine similarity**, and **sentence embeddings** to train a machine learning model (XGBoost) for prediction.


## Features

The app computes the following features for each pair of questions:

1. **Length-based features**
   - `lenq1`: Number of words in question 1
   - `lenq2`: Number of words in question 2
   - `len_diff`: Absolute difference in the number of words between question 1 and question 2

2. **Lexical overlap**
   - `common_words`: Count of common words between the two questions

3. **TF-IDF cosine similarity**
   - Measures similarity between questions using TF-IDF vectors

4. **Sentence embeddings similarity**
   - Cosine similarity of embeddings generated using [Sentence-BERT](https://www.sbert.net/) (`all-MiniLM-L6-v2`)


## Dataset

The project uses the **Quora Question Pairs** dataset:

- `train.csv`: Contains labeled question pairs (`is_duplicate` column indicates duplicates)
- `test.csv`: Contains unlabeled question pairs
- `sample_submission.csv`: Example format for submission

> **Note:** Large dataset CSV files should not be committed to GitHub. Use `.gitignore` to exclude them from the repository.


## Installation

1. Clone the repository:

git clone https://github.com/MonikaChaulagain/TextSimilarityApp.git
cd TextSimilarityApp


2. Install dependencies:

pip install -r requirements.txt


**Required packages include:**

* `pandas`, `numpy`
* `torch`
* `scikit-learn`
* `nltk`
* `sentence-transformers`
* `xgboost`

3. Download NLTK resources (optional, handled in code):

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


## Usage

1. **Data preprocessing:**

   * Lowercase conversion
   * Remove special characters
   * Remove stopwords
   * Lemmatization

2. **Feature computation:**

   * Length-based features
   * Common words
   * TF-IDF cosine similarity
   * Sentence embeddings cosine similarity

3. **Model training:**

   * Standardize features using `StandardScaler`
   * Train an XGBoost classifier (`gpu_hist` if GPU is available)
   * Evaluate using cross-validation (Accuracy, F1, ROC-AUC)

4. **Prediction and submission:**

   * Generate predictions on test set
   * Save results in `submission.csv`

jupyter notebook textSimilarity.ipynb


> Note: If GPU memory is insufficient, embeddings will be computed in smaller batches or fallback to CPU.



## GPU Support

* Embedding computation and XGBoost training are GPU-accelerated if CUDA-compatible GPU is available.
* Automatic fallback to CPU if GPU fails or memory is insufficient.



## Output
Accuracy: 0.767931435355809
Average F1 score: 0.6963082895258764
Average ROC-AUC score: 0.8504585926491656

* `submission.csv` containing:

  * `test_id`: Question pair ID
  * `is_duplicate`: Predicted label (0 or 1)

---

## Folder Structure

textsimilarityApp/
│
├─ data/quora-question-pairs/  # Dataset (ignored in Git)
│  ├─ train.csv
│  ├─ test.csv
│  └─ sample_submission.csv
│
├─ textSimilarity.ipynb         # Main notebook
├─ requirements.txt             # Dependencies
├─ .gitignore                   # Ignore large datasets
└─ README.md                    # Project documentation



## Notes

* The project handles **large datasets** efficiently using chunking for embedding computation to avoid GPU/CPU memory issues.
* GPU memory optimization includes:

  * Processing embeddings in chunks
  * Clearing GPU cache after each batch
  * Dynamic batch size reduction in case of OOM errors


## References

* [Quora Question Pairs Dataset](https://www.kaggle.com/c/quora-question-pairs)
* [Sentence-BERT](https://www.sbert.net/)
* [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)
* [NLTK Documentation](https://www.nltk.org/)



## Author

Monika Chaulagain
github:https://github.com/MonikaChaulagain


