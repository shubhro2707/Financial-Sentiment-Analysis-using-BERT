Sentiment Analysis of Corporate Reports using FinBERT

Project Overview

This project focuses on sentiment analysis of corporate reports using NLP, Machine Learning, and Deep Learning techniques. The goal was to fine-tune the FinBERT model on the Financial PhraseBank dataset to classify financial texts as positive, negative, or neutral. The model was then benchmarked against other state-of-the-art NLP models.

Working Process

Data Preparation:

The Financial PhraseBank dataset was used as the primary source.

Text data was cleaned and tokenized using FinBERT’s tokenizer.

Labels were assigned for sentiment classification.

Model Fine-Tuning:

FinBERT, a BERT model pre-trained on financial text, was used for transfer learning.

The model was fine-tuned on the Financial PhraseBank dataset using TensorFlow.

The dataset was split into training and test sets for evaluation.

Training & Evaluation:

The model was trained using the Adam optimizer with categorical cross-entropy loss.

Performance was evaluated using accuracy and F1-score.

The fine-tuned model achieved 94% accuracy.

Benchmarking:

The performance of FinBERT was compared with other NLP models like LSM, ELMo, and ULMFit.

FinBERT outperformed others, achieving an F1-score of 0.88.

Dataset

Financial PhraseBank: A dataset containing financial text labeled as positive, negative, or neutral.

The dataset was processed using Pandas and split into train and test sets.

Tools and Frameworks Used

Programming Language: Python

Libraries:

NLP: Transformers (Hugging Face), NLTK, SpaCy, Gensim

Machine Learning: Scikit-Learn, NumPy, Pandas

Deep Learning: TensorFlow, PyTorch, Keras

Visualization: Matplotlib, Seaborn

Database: PostgreSQL (for storing processed results)
