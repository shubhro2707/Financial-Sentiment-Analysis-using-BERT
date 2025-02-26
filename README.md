# **Sentiment Analysis of Corporate Reports Using FinBERT**  

## **Project Overview**  
This project focuses on sentiment analysis of corporate reports using **NLP, Machine Learning, and Deep Learning** techniques. The goal was to **fine-tune the FinBERT model** on the **Financial PhraseBank** dataset to classify financial texts as **positive, negative, or neutral**. The model was then **benchmarked** against other state-of-the-art NLP models.  

---  

## **Working Process**  

### **1. Data Preparation**  
- **Dataset:** Financial PhraseBank (primary source)  
- **Preprocessing:**  
  - Text cleaning and tokenization using **FinBERT’s tokenizer**  
  - Sentiment labels assigned (**positive, negative, neutral**)  

### **2. Model Fine-Tuning**  
- **FinBERT** (BERT model pre-trained on financial text) used for **transfer learning**  
- Fine-tuned on **Financial PhraseBank dataset** using **TensorFlow**  
- Dataset split into **training and test sets** for evaluation  

### **3. Training & Evaluation**  
- Trained using **Adam optimizer** with **categorical cross-entropy loss**  
- **Performance Metrics:**  
  - **Accuracy:** 94%  
  - **F1-score:** 0.88  

### **4. Benchmarking**  
- Compared **FinBERT** with other NLP models: **LSM, ELMo, ULMFit**  
- FinBERT **outperformed** other models with an **F1-score of 0.88**  

---  

## **Dataset**  
- **Financial PhraseBank:** Financial text labeled as **positive, negative, or neutral**  
- Processed using **Pandas** and split into **train/test sets**  

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

![image](https://github.com/user-attachments/assets/5fd7add6-1560-4aa4-a4a9-e5ccf6529b2f)

