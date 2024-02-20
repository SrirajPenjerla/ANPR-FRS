import cv2
import dlib
import numpy as np
from imutils import face_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

# Load face detector and shape predictor models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load trained SVM classifier and label encoder
svm_clf = pickle.load(open('svm_clf.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale and detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Facial Recognition System', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close window
cap.release()
cv2.destroyAllWindows()