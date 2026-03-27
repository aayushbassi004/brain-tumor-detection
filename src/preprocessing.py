import cv2
import numpy as np
import os

IMG_SIZE = 224

# 🔹 CLAHE (Contrast Limited Adaptive Histogram Equalization)
def apply_clahe(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    return enhanced

# 🔹 Skull Stripping (simple version)
def skull_stripping(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold
    _, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
    
    # Remove noise
    thresh = cv2.medianBlur(thresh, 5)
    
    # Find largest contour (brain area)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)
        img = cv2.bitwise_and(img, img, mask=mask)
    
    return img

# 🔹 Full preprocessing pipeline
def preprocess_image(img_path):
    img = cv2.imread(img_path)

    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Skull stripping
    img = skull_stripping(img)

    # CLAHE enhancement
    img = apply_clahe(img)

    # Normalize
    img = img / 255.0

    # Convert to 3 channel (important!)
    img = np.stack((img,)*3, axis=-1)

    return img


# 🔹 Load dataset
def load_data(data_dir):
    images = []
    labels = []
    
    classes = os.listdir(data_dir)
    
    for label, class_name in enumerate(classes):
        class_path = os.path.join(data_dir, class_name)
        
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            
            try:
                img = preprocess_image(img_path)
                images.append(img)
                labels.append(label)
            except:
                continue
    
    return np.array(images), np.array(labels)