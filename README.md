# Face Age Prediction using Deep Learning

A deep learning-based application that predicts a person's **age group from facial images** using a pre-trained CNN model. The system is built using **TensorFlow/Keras** and can be deployed via a **Flask web application**.

---

## Features

- Upload any facial image  
- Predict age group (0–10, 11–20, …)  
- Fast inference using transfer learning  
- Simple web interface (Flask)  

---

## Model Architecture

- **Base Model:** EfficientNetB0 (pre-trained on ImageNet)  
- **Approach:** Transfer Learning  

### Training Strategy:
- **Phase 1:** Freeze base model, train custom layers  
- **Phase 2:** Fine-tune last layers  

### Input:
- Image size: **224 × 224**

### Output:
- Age group classification (10 classes)

---

## Dataset

- Dataset used: [UTKFace Dataset](https://www.kaggle.com/datasets/jangedoo/utkface-new)  
- Converted into **age buckets (10-year intervals)**
- I have customized it and given the customized dataset

---

# Project Setup

Follow the steps below to set up and run the project locally.

## 1. System Requirements
	
- Python 3.9+ 
- pip (Python package manager) 
- Git 
- (Recommended) Virtual Environment 

## 2. Clone the Repository

git clone [https://github.com/your-username/face-age-prediction.git ](https://github.com/ankit7828/Face-age-prediction.git)   
cd face-age-prediction

## 3. Create Virtual Environment (Recommended)

**Windows:**   
python -m venv venv  
venv\Scripts\activate

**Linux / Mac:**  
python3 -m venv venv  
source venv/bin/activate  

## 4. Install Dependencies

pip install -r requirements.txt

## 5. Dataset Setup

**Option 1:** Use Provided Dataset  
I have placed dataset inside project folder:   

											dataset/
											 	train/
									  		 	val/
**Option 2:** Use UTKFace Dataset
Download UTKFace dataset from Kaggle 

Preprocess and organize into age groups:   
					
										dataset/
											train/
												0/
												1/
												2/
												...
											val/
												0/
												1/
												2/
												...

Each folder represents an age group (10-year gap)

## 6. Train the Model

If you want to train from scratch: python train.py  
Model will be saved in:   

						models/
					  		best_model.keras
					  		final_model.keras

## 7. Run the Web Application

python app.py

## 8. Open in Browser

http://127.0.0.1:5000/

## 9. Upload & Predict

Upload a face image 
Model predicts age group instantly 

## Evaluation Metrics we can use
| Metric            | Description                     |
|------------------|---------------------------------|
| Accuracy         | Exact prediction correctness    |
| Top-2 Accuracy   | Near-correct predictions        |
| MAE              | Average age error               |
| Confusion Matrix | Misclassification analysis      |
