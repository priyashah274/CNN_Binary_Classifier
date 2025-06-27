# 🐶🐱 Cat vs Dog Image Classifier using Convolutional Neural Networks (CNN)

This project is a deep learning-based binary image classifier that distinguishes between images of **cats** and **dogs** using a custom-built Convolutional Neural Network (CNN) implemented in PyTorch. It provides a complete end-to-end pipeline — from data preprocessing and model training to evaluation, visualization, and deployment via a web interface.

## 📌 Overview

The goal of this project is to classify images into two categories: **Cat** or **Dog**. The model is trained on the popular [Kaggle Dogs vs Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/data), and achieves high accuracy using a deep CNN architecture with multiple convolutional and pooling layers.

Users can interact with the model through a **Streamlit web application**, where they can upload their own image and get instant predictions.

---

## 🧠 Model Architecture

- **Input size**: 224x224 RGB image  
- **Architecture**:  
  - 5 Convolutional Layers with ReLU + BatchNorm + MaxPooling  
  - 3 Fully Connected Layers with Dropout  
- **Output**: Binary classification (Cat or Dog)  
- **Loss Function**: CrossEntropyLoss  
- **Optimizer**: Adam

---

## 📊 Training & Evaluation

- Trained on ~25,000 labeled images  
- Test accuracy: **~92%**  
- Includes training curves for loss and accuracy  
- Model saved as `binary_classifier.pth` for inference

---

## 🌐 Live Web App (Optional)

A **Streamlit app** is included that allows users to upload an image and get real-time classification.

To run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
