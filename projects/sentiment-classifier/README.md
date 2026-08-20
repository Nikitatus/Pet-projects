# Sentiment Comment Classification

This is a **pet project** where I built a sentiment classifier for tweets.  
The goal is to classify a comment into one of the three categories:
- **Negative**
- **Neutral**
- **Positive**

The project demonstrates two approaches:
1. **Custom trained word embeddings + BiLSTM (from scratch)**  
   - Training word embeddings to use as an input to the network.  
   - A bidirectional LSTM classifier implemented fully from scratch using PyTorch.  

2. **Modern pretrained models (LLMs)**  
   - Uses Hugging Face `transformers` to fine-tune an existing model.  
   - Provides a benchmark against the custom-built pipeline.  

---

## Features
- Full ML workflow: data preprocessing, embedding training, model building, and evaluation.  
- Implementation of BiLSTM from scratch in PyTorch.  
- UMAP visualization of embeddings.  
- Comparison with transformer-based models.
- REST API for the custom model  
- Follows good code practices.  

---

## Results
- First approach -73% accuracy
- Second approach -78% accuracy

The custom model shows the results which are not far from transformer models

For more metrics check the notebook.

---

Open notebook in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nikitatus/Sentiment-Comment-Classification/blob/main/sentiment_classification.ipynb)