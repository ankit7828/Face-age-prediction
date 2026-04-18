import numpy as np
import cv2
from tensorflow.keras.applications.efficientnet import preprocess_input

# Must match training classes (0–9 folders)
AGE_CLASSES = [
    '0-10', '11-20', '21-30', '31-40', '41-50',
    '51-60', '61-70', '71-80', '81-90', '91-100'
]

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    
    if img is None:
        raise ValueError("Image not found or invalid path")
    
    img = cv2.resize(img, (224, 224))
    
    # ✅ IMPORTANT: same preprocessing as training
    img = preprocess_input(img)
    
    img = np.expand_dims(img, axis=0)
    return img

def predict_age(model, image_path):
    img = preprocess_image(image_path)
    
    preds = model.predict(img)[0]
    
    # ✅ Better prediction (reduces middle-class bias)
    ages = np.array([5,15,25,35,45,55,65,75,85,95])
    predicted_age = np.sum(preds * ages)
    
    class_index = int(predicted_age // 10)
    class_index = min(class_index, 9)
    
    confidence = float(np.max(preds))
    
    return AGE_CLASSES[class_index], confidence