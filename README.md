# Face-age-prediction
Face Age Prediction with Small Dataset - (A deep learning model that predicts a person’s age group from a face image using transfer learning on a small dataset.)

# Face Age Group Prediction using Deep Learning

A deep learning-based web application that predicts a person's age group from facial images using a pre-trained CNN model. The system is built using TensorFlow/Keras and deployed via a Flask web app.

#Features
*You can upload any face image 
*Model predict age group (0–10, 11–20, …) 
*Fast inference using pre-trained model 
*Simple web interface (Flask) 
*Evaluation using: 
*Accuracy 
*Top-2 Accuracy 
*Mean Absolute Error (MAE) 
*Confusion Matrix


Model Details
Architecture: Transfer Learning (EfficientNetB0) 
Training Strategy: 
Phase 1: Frozen base model 
Phase 2: Fine-tuning 
Input Size: 224 x 224 
Output: Age group classification

Dataset
Dataset used: UTKFace Dataset (from Kaggle = https://www.kaggle.com/datasets/jangedoo/utkface-new) 
I have converted into age buckets (10-year intervals) 
I have customized it and given the customized dataset

Project Setup
Follow the steps below to set up and run the project locally.

1. System Requirements
Python 3.9+ 
pip (Python package manager) 
Git 
(Recommended) Virtual Environment 

2. Clone the Repository
git clone https://github.com/your-username/face-age-group-project.git
cd face-age-group-project

3. Create Virtual Environment (Recommended)
Windows: 
python -m venv venv
venv\Scripts\activate

Linux / Mac:
python3 -m venv venv
source venv/bin/activate

4. Install Dependencies
pip install -r requirements.txt

5. Dataset Setup
Option 1: Use Provided Dataset
I have placed dataset inside project folder: 
dataset/
		 train/
  		 val/

Option 2: Use UTKFace Dataset
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

6. Train the Model
If you want to train from scratch: python train.py
Model will be saved in: 
models/
  		best_model.keras
  		final_model.keras

7. Run the Web Application
python app.py

8. Open in Browser
http://127.0.0.1:5000/

9. Upload & Predict
Upload a face image 
Model predicts age group instantly 

Evaluation Metrics we can use
Metric	Description
Accuracy	Exact prediction correctness
Top-2 Accuracy	Near-correct predictions
MAE	Average age error
Confusion Matrix	Misclassification analysis
