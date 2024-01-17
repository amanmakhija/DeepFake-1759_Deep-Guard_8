from flask import Flask, render_template, request, jsonify
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input

app = Flask(__name__)

# Load the pre-trained model
model = load_model('xception_deepfake_image.h5')

import cv2
import numpy as np

def crop_face_from_array(image_array):
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Check if any faces are detected
    if len(faces) == 0:
        print("No faces found")
        return None

    # Assume only one face is present in the image (you may modify this if necessary)
    x, y, w, h = faces[0]

    # Crop the face from the image
    cropped_face = image_array[y:y + h, x:x + w]

    return cropped_face


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    print("Request received")
    file = request.files['image']
    print("Request received")

    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    file.save(os.path.join(os.path.curdir, file.filename))

    if file:
        try:
            # Read and preprocess the image 
            img_array = cv2.imread(filename=file.filename)
            # img_array = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

            img = crop_face_from_array(img)
            print("Image received")

            # Ensure the image is successfully decoded
            if img is None:
                return jsonify({'error': 'Failed to decode image'})

            # If the image has an alpha channel, remove it
            if img.shape[-1] == 4:
                img = img[:, :, :3]

            # Ensure the image has 3 channels (for compatibility with preprocess_input)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.shape[-1] == 1 else img

            img = cv2.resize(img, (224, 224))
            img = preprocess_input(img.reshape(1, 224, 224, 3))

            # Make prediction
            prediction = model.predict(img)

            # Convert the prediction to a human-readable label
            label = 'FAKE' if prediction[0][0] > 0.5 else 'REAL'
            print(label)
            return jsonify({'prediction': label})
        
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)