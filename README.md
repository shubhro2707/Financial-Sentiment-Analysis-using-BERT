# **Sentiment Analysis of Corporate Reports Using FinBERT**  

## **Project Overview**  
In this project, I worked on **sentiment analysis of corporate reports** using **NLP, Machine Learning, and Deep Learning** techniques. The objective was to **fine-tune the FinBERT model** on the **Financial PhraseBank** dataset to classify financial texts into **positive, negative, or neutral** sentiments. After fine-tuning, I **benchmarked FinBERT** against other state-of-the-art NLP models to evaluate its performance.  

---  

## **Working Process**  

### **1. Data Preparation**  
- I used the **Financial PhraseBank** as the primary dataset for this task.  
- Before training, I **cleaned and tokenized** the text using **FinBERT’s tokenizer**.  
- I then assigned **sentiment labels** to the data, categorizing each text as **positive, negative, or neutral**.  

### **2. Model Fine-Tuning**  
- I leveraged **FinBERT**, a BERT model **pre-trained on financial text**, for **transfer learning**.  
- I fine-tuned the model using the **Financial PhraseBank dataset** with **TensorFlow**.  
- The dataset was split into **training and test sets** for evaluation.  

### **3. Training & Evaluation**  
- I trained the model using the **Adam optimizer** with **categorical cross-entropy loss**.  
- The performance of the model was evaluated, and it achieved:  
  - **Accuracy:** 94%  
  - **F1-score:** 0.88  

### **4. Benchmarking**  
- To validate FinBERT’s effectiveness, I compared its performance with other NLP models like **LSM, ELMo, and ULMFit**.  
- FinBERT **outperformed** all other models, achieving an **F1-score of 0.88**, making it the best choice for financial sentiment classification.  

---  

## **Dataset**  
- The **Financial PhraseBank** dataset was used, containing financial text labeled as **positive, negative, or neutral**.  
- I processed the dataset using **Pandas** and split it into **train and test sets** for training and evaluation.  

---  

## **Tools and Frameworks Used**  

### **Programming Language**  
- **Python**  

### **Libraries**  
- **NLP:** Transformers (Hugging Face), NLTK, SpaCy, Gensim  
- **Machine Learning:** Scikit-Learn, NumPy, Pandas  
- **Deep Learning:** TensorFlow, PyTorch, Keras  
- **Visualization:** Matplotlib, Seaborn  
- **Database:** PostgreSQL (for storing processed results)  

This project allowed me to explore **NLP in financial analysis**, fine-tune **deep learning models**, and benchmark them effectively for real-world applications.

![image](https://github.com/user-attachments/assets/5fd7add6-1560-4aa4-a4a9-e5ccf6529b2f)

