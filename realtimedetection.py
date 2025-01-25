# import cv2
# from keras.models import model_from_json
# import numpy as np
# # from keras_preprocessing.image import load_img
# json_file = open("facialemotionmodel.json", "r")
# model_json = json_file.read()
# json_file.close()
# model = model_from_json(model_json)

# model.load_weights("facialemotionmodel.h5")
# haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# face_cascade=cv2.CascadeClassifier(haar_file)

# def extract_features(image):
#     feature = np.array(image)
#     feature = feature.reshape(1,48,48,1)
#     return feature/255.0

# webcam=cv2.VideoCapture(0)
# labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'neutral', 5 : 'sad', 6 : 'surprise'}
# while True:
#     i,im=webcam.read()
#     gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#     faces=face_cascade.detectMultiScale(im,1.3,5)
#     try: 
#         for (p,q,r,s) in faces:
#             image = gray[q:q+s,p:p+r]
#             cv2.rectangle(im,(p,q),(p+r,q+s),(255,0,0),2)
#             image = cv2.resize(image,(48,48))
#             img = extract_features(image)
#             pred = model.predict(img)
#             prediction_label = labels[pred.argmax()]
#             # print("Predicted Output:", prediction_label)
#             # cv2.putText(im,prediction_label)
#             cv2.putText(im, '% s' %(prediction_label), (p-10, q-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,2, (0,0,255))
#         cv2.imshow("Output",im)
#         cv2.waitKey(27)
#     except cv2.error:
#         pass

import cv2
from keras.models import model_from_json, Sequential
import numpy as np
import tensorflow as tf

# Load the model from the JSON file
# model.save("facialemotionmodel.h5")
# model = load_model("facialemotionmodel.h5")
json_file = open("emotiondetector.json")
model_json = json_file.read()
json_file.close()

# Deserialize the model from JSON
model = model_from_json(model_json)

# Load the model weights
model.load_weights("emotiondetector.h5")

# Load the Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to preprocess the image for prediction
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Open a connection to the webcam
webcam = cv2.VideoCapture(0)

# Define the emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    # Read a frame from the webcam
    ret, im = webcam.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    try:
        # Process each detected face
        for (p, q, r, s) in faces:
            image = gray[q:q+s, p:p+r]
            cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)

            # Resize and preprocess the face image
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)

            # Make a prediction using the model
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]

            # Display the prediction on the frame
            cv2.putText(im, '% s' % (prediction_label), (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

        # Show the frame with the prediction
        cv2.imshow("Output", im)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except cv2.error as e:
        print(f"Error: {e}")
        pass

# Release the webcam and close windows
webcam.release()
webcame.stop()
cv2.destroyAllWindows()
