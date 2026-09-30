# RNN-
RNN-based sentiment classifier trained on the IMDB 50k movie reviews dataset. Includes text cleaning, stop word removal with NLTK, a train/test split, and a recurrent neural network that predicts whether a review is positive or negative. Built in Python with a clear, reproducible workflow from raw data to evaluation.**

# IMDB Movie Review Sentiment Analysis with RNN

A deep learning project that classifies movie reviews as **positive** or **negative** using a Recurrent Neural Network (RNN). The project was built and run in **Google Colab** and uses the IMDB 50k movie reviews dataset.

## Overview

This project covers the full workflow of a text classification task:

1. Loading the IMDB dataset
2. Cleaning and preprocessing the review text
3. Removing stop words with NLTK
4. Splitting the data into training and testing sets
5. Building and training an RNN model
6. Evaluating the model on unseen reviews

## Dataset

- **File:** `IMDB_Dataset.csv`
- **Size:** 50,000 movie reviews
- **Columns:**
  - `review`: the text of the movie review
  - `sentiment`: the label (`positive` or `negative`)
- **Task:** binary sentiment classification

## Tech Stack

- Python 3
- Google Colab
- NLTK (stop word removal and text processing)
- Pandas and NumPy (data handling)
- TensorFlow / Keras (RNN model)
- Scikit-learn (train/test split and evaluation)

## Project Workflow

### 1. Data Cleaning
- Lowercased all text
- Removed HTML tags, punctuation, and special characters
- Removed extra whitespace

### 2. Stop Word Removal
- Removed common English stop words using NLTK to keep only meaningful words

### 3. Train/Test Split
- Split the dataset into training and testing sets to measure performance on unseen data

### 4. Text Vectorization
- Tokenized the reviews and padded them to a fixed length so they can be fed into the RNN

### 5. Model
- Embedding layer followed by a recurrent layer (SimpleRNN / LSTM / GRU: update to match your model)
- Dense output layer with sigmoid activation for binary classification

### 6. Evaluation
- Accuracy and loss on the test set
- Final results: train accuracy **94.70%**, test accuracy **83.83%**

## How to Run

1. Open the notebook in Google Colab.
2. Upload `IMDB_Dataset.csv` to your Colab session (or mount Google Drive and update the file path).
3. Run the first cell to install and download the NLTK resources:
   ```python
   import nltk
   nltk.download('stopwords')
   ```
4. Run all remaining cells in order.

## Repository Structure

```
├── IMDB_Dataset.csv      # Dataset (or link to it if too large)
├── notebook.ipynb        # Google Colab notebook with the full code
└── README.md             # Project documentation
```

## Results

| Metric | Score |
|--------|-------|
| Train accuracy | 94.70% (0.9470) |
| Test accuracy | 83.83% (0.8383) |

The gap between training and test scores shows some overfitting, which the improvements below aim to reduce.

## Future Improvements

- Try LSTM or GRU layers for better long-range context
- Use pretrained word embeddings such as GloVe or Word2Vec
- Tune hyperparameters (epochs, batch size, embedding size)
- Add dropout to reduce overfitting

## License

This project is open source and available under the MIT License.
