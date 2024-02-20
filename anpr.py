import cv2
import numpy as np
import tensorflow as tf

# Load trained CNN model
model = tf.keras.models.load_model('anpr_cnn_model.h5')

# Load image
img = cv2.imread('car.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# Apply morphological transformations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Find contours
contours, hierarchy = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find license plate region
plate = None
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h
    if 2.5 < aspect_ratio < 5 and 120 < w < 400 and 30 < h < 100:
        plate = img[y:y+h, x:x+w]
        break

if plate is not None:
    # Convert license plate region to grayscale
    gray_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(gray_plate, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Apply morphological transformations
    plate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    plate_morph = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, plate_kernel)

    # Find contours of individual characters
    char_contours, char_hierarchy = cv2.findContours(plate_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours from left to right
    char_contours = sorted(char_contours, key=lambda x: cv2.boundingRect(x)[0])

    # Recognize characters using CNN
    chars = []
    for char_contour in char_contours:
        x, y, w, h = cv2.boundingRect(char_contour)
        char_roi = plate_morph[y:y+h, x:x+w]
        char_roi_resized = cv2.resize(char_roi, (28, 28))
        char_roi_normalized = char_roi_resized / 255.0
        char_roi_reshaped = char_roi_normalized.reshape(1, 28, 28, 1)
        char_pred = model.predict(char_roi_reshaped)
        char_label = np.argmax(char_pred)
        char = chr(char_label + 65) if char_label < 26 else chr(char_label + 22)
        chars.append(char)

    # Print recognized text
    text = ''.join(chars)
    print(text)