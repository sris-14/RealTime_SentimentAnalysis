
import cv2
from tensorflow.keras.models import load_model
import numpy as np

# Load the saved model (Make sure it's in the correct path)
model = load_model('sequential_saved_model.h5')

# Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Feature extraction function
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)  # Reshape for model input
    return feature / 255.0  # Normalize

# Initialize webcam
webcam = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not webcam.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Skip the first few frames to allow the camera to initialize
for _ in range(5):
    i, im = webcam.read()
    if not i:
        print("Error: Failed to read the initial frame.")
        break

# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    i, im = webcam.read()

    # Check if the frame was successfully captured
    if not i or im is None:
        print("Error: Failed to capture image.")
        break

    # Debugging: Check the shape of the captured image
    print(f"Captured frame shape: {im.shape if im is not None else 'None'}")

    try:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        faces = face_cascade.detectMultiScale(im, 1.3, 5)  # Detect faces
        if len(faces) == 0:
            print("No faces detected.")
        
        for (p, q, r, s) in faces:
            image = gray[q:q + s, p:p + r]
            cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)  # Draw rectangle around face
            image = cv2.resize(image, (48, 48))  # Resize to model's expected input size
            img = extract_features(image)
            pred = model.predict(img)  # Get model prediction
            prediction_label = labels[pred.argmax()]  # Get the predicted emotion label

            # Display the predicted emotion label
            cv2.putText(im, '% s' % (prediction_label), (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

        # Display the output frame with the predicted label
        cv2.imshow("Output", im)

        # Exit the loop when pressing 'ESC' key (27)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    except Exception as e:
        print(f"Error processing frame: {e}")
        pass

# Cleanup
webcam.release()
cv2.destroyAllWindows()
    
    