# Garbage Type Detection using CNN

A deep learning project that classifies waste images into **Renewable**, **Non-Renewable**, and **Biodegradable** categories using a **Convolutional Neural Network (CNN)** based on the **MobileNet** architecture. The project includes a **Streamlit web app** for real-time image classification; users can upload a waste image and get predictions instantly through a browser interface.

---

## Overview

This project was developed to explore how **AI and Computer Vision** can help with **environmental sustainability**. It automates waste type detection and supports smarter waste separation and eco-friendly practices. 

---

## Project Workflow

1. **Data Preprocessing**: Cleaning, resizing, and augmenting image datasets (`data_preprocessing.py`)  
2. **Model Training**: Building and training the CNN (MobileNet) model (`train_model.py`)  
3. **Model Saving**: Storing the trained model (`final_mobilenet_model.keras`)  
4. **Deployment**: Hosting the model via **Streamlit** for web-based interaction (`app.py`)

---

## Tech Stack

- Python  
- TensorFlow & Keras  
- OpenCV  
- NumPy  
- Streamlit  

---

## Quick Start

1. Clone the repo  
   ```bash
   git clone https://github.com/<your-username>/garbage-type-detection-cnn.git
   cd garbage-type-detection-cnn
   ```

2. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app  
   ```bash
   streamlit run app.py
   ```

4. Upload an image of waste and view its classification result instantly.



Model Summary
1.Architecture: MobileNet (transfer learning)
2.Loss Function: Categorical Cross-Entropy
3.Optimizer: Adam
4.Metrics: Accuracy

Future Scope
1.Real-time camera detection
2.Larger dataset for better accuracy
3.Cloud or mobile deployment for scalability
