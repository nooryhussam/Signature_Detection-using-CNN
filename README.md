#✍️ Signature_Detection-using-CNN

📌 Overview
This project implements a Convolutional Neural Network (CNN) model to detect whether a handwritten signature is genuine or forged. Signature forgery is a serious issue in document verification, banking, and legal processes — and this system helps automate and improve accuracy in signature authentication.

🧠 Project Objective
To build a deep learning-based system that:

Distinguishes between original (genuine) and forged signatures

Learns key visual features from images using CNN

Achieves high accuracy in binary classification

🛠️ Tech Stack
Python

TensorFlow / Keras

OpenCV

NumPy, Matplotlib

Google Colab

🏗️ Model Architecture
Input: Grayscale signature image (e.g., 128x128 pixels)

Convolutional layers + ReLU + MaxPooling

Fully connected (Dense) layers

Sigmoid output layer for binary classification

🎯 Results
Achieved high accuracy (>95%) on validation set

Model effectively distinguishes forgery patterns

Confusion matrix shows strong separation between classes

📊 Evaluation Metrics
Accuracy

Precision / Recall

Confusion Matrix

Loss / Accuracy Curves

📎 Future Enhancements
Add Siamese Network for signature matching

Integrate OCR for ID + signature matching

Deploy as a web or mobile application for real-time verification

