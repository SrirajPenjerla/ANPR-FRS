import cv2
import dlib
import numpy as np
from imutils import face_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Load ANPR models
lp_detector = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
lp_recognizer = load_model('lp_recognizer.h5')

# Load FRS models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
svm_clf = pickle.load(open('svm_clf.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale and detect license plates
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lps = lp_detector.detectMultiScale(gray, 1.3, 5)

    # Loop through license plates and recognize them using ANPR
    for (x, y, w, h) in lps:
        lp = frame[y:y+h, x:x+w]
        lp = cv2.resize(lp, (128, 64))
        lp_gray = cv2.cvtColor(lp, cv2.COLOR_BGR2GRAY)
        lp_gray = lp_gray.astype('float') / 255.0
        lp_gray = np.expand_dims(lp_gray, axis=-1)
        lp_gray = np.expand_dims(lp_gray, axis=0)
        lp_text = lp_recognizer.predict(lp_gray)
        lp_text = np.argmax(lp_text, axis=-1)[0]

        # Draw box and label around the license plate
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, 'LP: ' + str(lp_text), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Detect faces in the frame
    rects = detector(gray, 0)

    # Loop through faces and align them using facial landmarks
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        x, y, w, h = face_utils.rect_to_bb(rect)
        face_aligned = cv2.resize(frame[y:y+h, x:x+w], (128, 128))
        face_aligned_gray = cv2.cvtColor(face_aligned, cv2.COLOR_BGR2GRAY)

        # Extract facial features from the aligned face
        hog = cv2.HOGDescriptor()
        features = hog.compute(face_aligned_gray)
        features = np.array(features).reshape((1, -1))

        # Recognize the face using SVM classifier
        label = svm_clf.predict(features)
        name = label_encoder.inverse_transform(label)[0]

        # Draw box and label around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Name: ' + str(name), (x,    y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the resulting frame
cv2.imshow('ANPR-FRS', frame)

# Exit when 'q' key is pressed
if cv2.waitKey(1) & 0xFF == ord('q'):
    break
    cap.release()
cv2.destroyAllWindows()